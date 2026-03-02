// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "helper.h"

// Unified kernel merging speculate_update +
// speculate_set_value_by_flags_and_idx into a single kernel launch.
//
// IMPORTANT: step_output_ids and step_output_len are READ-ONLY in this kernel.
// save_output reads them AFTER this kernel, so they must not be modified here.
// The original speculate_set_value_by_flags_and_idx cleared accept_num and
// seq_lens_decoder for stopped seqs, but that happened AFTER save_output.
// In the new flow, that cleanup is no longer needed here.
//
// Variable naming uses dual-semantic convention for Phase 2 unification:
//   step_input_ids  (was: draft_tokens)       - decode input tokens per step
//   step_output_ids (was: accept_tokens)      - verified output tokens per step
//   step_output_len (was: accept_num)         - output length per batch (spec:
//   N, naive: 1) adaptive_step_input_len (was: actual_draft_token_nums) -
//   adaptive input length, for dynamic draft token for future has_running_seqs
//   (was: not_need_stop)     - whether any sequences are still running
//   is_paused (was: is_block_step)            - whether batch is paused/blocked

template <int THREADBLOCK_SIZE>
__global__ void unified_update_model_status_kernel(
    // Sequence state (read-write)
    int *seq_lens_encoder,
    int *seq_lens_decoder,
    bool *has_running_seqs,
    int *mask_rollback,
    // Step input (read-write)
    int64_t *step_input_ids,
    int *adaptive_step_input_len,
    // Step output (READ-ONLY — save_output reads these after this kernel)
    const int64_t *step_output_ids,
    const int *step_output_len,
    // Control flags (read-only)
    const bool *stop_flags,
    int *seq_lens_this_time,
    const bool *is_paused,
    // History (read-write)
    int64_t *pre_ids,
    const int64_t *step_idx,
    // Dimension parameters
    const int real_bsz,
    const int max_bsz,
    const int max_step_tokens,
    const int pre_ids_len,
    const bool is_naive_mode) {
  const int bid = threadIdx.x;
  const int output_len = step_output_len[bid];
  int stop_flag_now_int = 0;

  if (!(is_paused[bid] || bid >= real_bsz)) {
    if (stop_flags[bid]) {
      // [from speculate_update] Stopped sequence — only set rollback
      stop_flag_now_int = 1;
      mask_rollback[bid] = 0;
      // Cleanup seq_lens_decoder for sequences that were already stopped in a
      // previous round (output_len==0 because verify initialized accept_num to
      // 0). Newly-stopped sequences (output_len>0) keep their seq_lens_decoder
      // for save_output to read after this kernel.
      if (output_len == 0) {
        seq_lens_decoder[bid] = 0;
      }
    } else if (seq_lens_encoder[bid] == 0) {
      // === Decoder phase ===
      // [from speculate_update] Advance seq_lens_decoder
      seq_lens_decoder[bid] += output_len;
      mask_rollback[bid] = seq_lens_this_time[bid] - output_len;
    } else {
      // === Encoder (prefill) phase ===
      mask_rollback[bid] = 0;
    }

    // [from speculate_update] Encoder → decoder transition
    if (seq_lens_encoder[bid] != 0) {
      seq_lens_decoder[bid] += seq_lens_encoder[bid];
      seq_lens_encoder[bid] = 0;
    }

    // [from speculate_set_value_by_flags_and_idx] Write output to history.
    // MUST be AFTER encoder→decoder transition so that prefill sequences
    // (which just had seq_lens_encoder zeroed above) also get their pre_ids
    // written. This matches the original two-kernel execution order where
    // speculate_set_value_by_flags_and_idx ran after speculate_update.
    if (!stop_flags[bid] && step_idx[bid] > 0) {
      int64_t *history = pre_ids + bid * pre_ids_len;
      const int64_t *output = step_output_ids + bid * max_step_tokens;
      for (int i = 0; i < output_len; i++) {
        history[step_idx[bid] - i] = output[output_len - 1 - i];
      }
    }

    // [from speculate_update] Write back next step's first input token.
    // Guard output_len > 0 to avoid OOB read when accept_num was zeroed
    // for stopped sequences by verify kernel's initialization.
    if (output_len > 0) {
      step_input_ids[bid * max_step_tokens] =
          step_output_ids[bid * max_step_tokens + output_len - 1];
    }

    // Reset seq_lens_this_time for next step's decode.
    // Only in naive mode (no proposer), where this is the final value.
    // In spec mode (MTP/Ngram), the proposer overwrites this afterward,
    // so we must NOT touch it here — downstream kernels
    // (draft_model_preprocess, eagle_get_hidden_states) depend on
    // the original value before the proposer runs.
    if (is_naive_mode) {
      seq_lens_this_time[bid] = stop_flags[bid] ? 0 : 1;
    }

  } else if (bid >= real_bsz && bid < max_bsz) {
    stop_flag_now_int = 1;
    // Cleanup: slot exited scheduling, may have stale seq_lens_decoder
    // from when it was active. Zero it so new sequences in this slot
    // get correct encoder→decoder transition.
    seq_lens_decoder[bid] = 0;
  }

  // [from speculate_update] Reduce to determine has_running_seqs
  __syncthreads();
  typedef cub::BlockReduce<int64_t, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int64_t stop_sum = BlockReduce(temp_storage).Sum(stop_flag_now_int);

  if (threadIdx.x == 0) {
    has_running_seqs[0] = stop_sum < max_bsz;
  }
}

void UnifiedUpdateModelStatus(const paddle::Tensor &seq_lens_encoder,
                              const paddle::Tensor &seq_lens_decoder,
                              const paddle::Tensor &has_running_seqs,
                              const paddle::Tensor &step_input_ids,
                              const paddle::Tensor &adaptive_step_input_len,
                              const paddle::Tensor &step_output_ids,
                              const paddle::Tensor &step_output_len,
                              const paddle::Tensor &stop_flags,
                              const paddle::Tensor &seq_lens_this_time,
                              const paddle::Tensor &is_paused,
                              const paddle::Tensor &mask_rollback,
                              const paddle::Tensor &pre_ids,
                              const paddle::Tensor &step_idx,
                              const bool is_naive_mode) {
  const int real_bsz = seq_lens_this_time.shape()[0];
  const int max_bsz = stop_flags.shape()[0];
  const int max_step_tokens = step_input_ids.shape()[1];
  const int pre_ids_len = pre_ids.shape()[1];

  constexpr int BlockSize = 512;

  // Copy has_running_seqs to GPU for kernel reduction
  auto has_running_seqs_gpu =
      has_running_seqs.copy_to(stop_flags.place(), false);

  unified_update_model_status_kernel<BlockSize>
      <<<1, BlockSize, 0, step_output_ids.stream()>>>(
          const_cast<int *>(seq_lens_encoder.data<int>()),
          const_cast<int *>(seq_lens_decoder.data<int>()),
          const_cast<bool *>(has_running_seqs_gpu.data<bool>()),
          const_cast<int *>(mask_rollback.data<int>()),
          const_cast<int64_t *>(step_input_ids.data<int64_t>()),
          const_cast<int *>(adaptive_step_input_len.data<int>()),
          step_output_ids.data<int64_t>(),
          step_output_len.data<int>(),
          stop_flags.data<bool>(),
          const_cast<int *>(seq_lens_this_time.data<int>()),
          is_paused.data<bool>(),
          const_cast<int64_t *>(pre_ids.data<int64_t>()),
          step_idx.data<int64_t>(),
          real_bsz,
          max_bsz,
          max_step_tokens,
          pre_ids_len,
          is_naive_mode);

  // Copy has_running_seqs back to CPU (synchronous)
  auto has_running_seqs_cpu =
      has_running_seqs_gpu.copy_to(has_running_seqs.place(), true);
  bool *has_running_seqs_data =
      const_cast<bool *>(has_running_seqs.data<bool>());
  has_running_seqs_data[0] = has_running_seqs_cpu.data<bool>()[0];
}

PD_BUILD_STATIC_OP(unified_update_model_status)
    .Inputs({"seq_lens_encoder",
             "seq_lens_decoder",
             "has_running_seqs",
             "step_input_ids",
             "adaptive_step_input_len",
             "step_output_ids",
             "step_output_len",
             "stop_flags",
             "seq_lens_this_time",
             "is_paused",
             "mask_rollback",
             "pre_ids",
             "step_idx"})
    .Attrs({"is_naive_mode: bool"})
    .Outputs({"seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "has_running_seqs_out",
              "step_input_ids_out",
              "adaptive_step_input_len_out",
              "seq_lens_this_time_out",
              "mask_rollback_out",
              "pre_ids_out"})
    .SetInplaceMap({{"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"has_running_seqs", "has_running_seqs_out"},
                    {"step_input_ids", "step_input_ids_out"},
                    {"adaptive_step_input_len", "adaptive_step_input_len_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"mask_rollback", "mask_rollback_out"},
                    {"pre_ids", "pre_ids_out"}})
    .SetKernelFn(PD_KERNEL(UnifiedUpdateModelStatus));
