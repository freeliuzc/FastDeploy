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

// Unified kernel handling ALL state management after verification:
//   Phase A: EOS / max_dec_len detection + step_idx increment + stop_flags
//   Phase B: seq_lens_decoder / encoder / mask_rollback / pre_ids / etc.
//   Phase C: has_running_seqs reduction
//
// This kernel is shared by both speculative and non-speculative (naive) paths.
// Non-spec path skips speculate_verify entirely and goes directly here.
//
// Variable naming uses dual-semantic convention:
//   step_input_ids  (was: draft_tokens)       - decode input tokens per step
//   step_output_ids (was: accept_tokens)      - verified output tokens per step
//   step_output_len (was: accept_num)         - output length per batch
//   adaptive_step_input_len (was: actual_draft_token_nums)
//   has_running_seqs (was: not_need_stop)
//   is_paused (was: is_block_step)

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
    // Step output (read-write: may truncate on EOS/max_dec_len)
    int64_t *step_output_ids,
    int *step_output_len,
    // Control flags (read-write: may set stop_flags on EOS/max_dec_len)
    bool *stop_flags,
    int *seq_lens_this_time,
    const bool *is_paused,
    // History (read-write)
    int64_t *pre_ids,
    int64_t *step_idx,
    // EOS / stop detection params (new for Task 4)
    const int64_t *end_tokens,
    const int64_t *max_dec_len,
    // Dimension parameters
    const int real_bsz,
    const int max_bsz,
    const int max_step_tokens,
    const int pre_ids_len,
    const int num_end_tokens,
    const bool is_naive_mode,
    const bool prefill_one_step_stop) {
  const int bid = threadIdx.x;
  int stop_flag_now_int = 0;

  // === Phase A: EOS / max_dec_len detection + step_idx update ===
  // Must run BEFORE Phase B because it may truncate output_len and set
  // stop_flags, which Phase B's branching depends on.
  int output_len = step_output_len[bid];

  if (!(is_paused[bid] || bid >= real_bsz) && !stop_flags[bid]) {
    bool eos_stopped = false;
    for (int i = 0; i < output_len; i++) {
      step_idx[bid]++;
      int64_t token = step_output_ids[bid * max_step_tokens + i];
      bool is_eos = is_in_end(token, end_tokens, num_end_tokens);

      if (is_eos || step_idx[bid] >= max_dec_len[bid]) {
        if (!is_eos) {
          // max_dec_len hit — force end token
          step_output_ids[bid * max_step_tokens + i] = end_tokens[0];
        }
        output_len = i + 1;  // truncate
        step_output_len[bid] = output_len;
        stop_flags[bid] = true;
        eos_stopped = true;
        break;
      }
    }
    if (!eos_stopped && prefill_one_step_stop && seq_lens_encoder[bid] != 0) {
      // prefill_one_step_stop: stop after first prefill token
      stop_flags[bid] = true;
    }
  }

  // === Phase B: State updates (original unified_update logic) ===
  if (!(is_paused[bid] || bid >= real_bsz)) {
    if (stop_flags[bid]) {
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
      seq_lens_decoder[bid] += output_len;
      mask_rollback[bid] = seq_lens_this_time[bid] - output_len;
    } else {
      // === Encoder (prefill) phase ===
      mask_rollback[bid] = 0;
    }

    // Encoder -> decoder transition
    if (seq_lens_encoder[bid] != 0) {
      seq_lens_decoder[bid] += seq_lens_encoder[bid];
      seq_lens_encoder[bid] = 0;
    }

    // Write output to history (pre_ids).
    // MUST be AFTER encoder->decoder transition so that prefill sequences
    // also get their pre_ids written.
    if (!stop_flags[bid] && step_idx[bid] > 0) {
      int64_t *history = pre_ids + bid * pre_ids_len;
      const int64_t *output = step_output_ids + bid * max_step_tokens;
      for (int i = 0; i < output_len; i++) {
        history[step_idx[bid] - i] = output[output_len - 1 - i];
      }
    }

    // Write back next step's first input token.
    if (output_len > 0) {
      step_input_ids[bid * max_step_tokens] =
          step_output_ids[bid * max_step_tokens + output_len - 1];
    }

    // Reset seq_lens_this_time for next step's decode.
    // Only in naive mode (no proposer), where this is the final value.
    // In spec mode (MTP/Ngram), the proposer overwrites this afterward.
    if (is_naive_mode) {
      seq_lens_this_time[bid] = stop_flags[bid] ? 0 : 1;
    }

  } else if (bid >= real_bsz && bid < max_bsz) {
    stop_flag_now_int = 1;
    seq_lens_decoder[bid] = 0;
  }

  // === Phase C: Reduce to determine has_running_seqs ===
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
                              const paddle::Tensor &end_tokens,
                              const paddle::Tensor &max_dec_len,
                              const bool is_naive_mode,
                              const bool prefill_one_step_stop) {
  const int real_bsz = seq_lens_this_time.shape()[0];
  const int max_bsz = stop_flags.shape()[0];
  const int max_step_tokens = step_input_ids.shape()[1];
  const int pre_ids_len = pre_ids.shape()[1];
  const int num_end_tokens = end_tokens.shape()[0];

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
          const_cast<int64_t *>(step_output_ids.data<int64_t>()),
          const_cast<int *>(step_output_len.data<int>()),
          const_cast<bool *>(stop_flags.data<bool>()),
          const_cast<int *>(seq_lens_this_time.data<int>()),
          is_paused.data<bool>(),
          const_cast<int64_t *>(pre_ids.data<int64_t>()),
          const_cast<int64_t *>(step_idx.data<int64_t>()),
          end_tokens.data<int64_t>(),
          max_dec_len.data<int64_t>(),
          real_bsz,
          max_bsz,
          max_step_tokens,
          pre_ids_len,
          num_end_tokens,
          is_naive_mode,
          prefill_one_step_stop);

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
             "step_idx",
             "end_tokens",
             "max_dec_len"})
    .Attrs({"is_naive_mode: bool", "prefill_one_step_stop: bool"})
    .Outputs({"seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "has_running_seqs_out",
              "step_input_ids_out",
              "adaptive_step_input_len_out",
              "step_output_ids_out",
              "step_output_len_out",
              "stop_flags_out",
              "seq_lens_this_time_out",
              "mask_rollback_out",
              "pre_ids_out",
              "step_idx_out"})
    .SetInplaceMap({{"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"has_running_seqs", "has_running_seqs_out"},
                    {"step_input_ids", "step_input_ids_out"},
                    {"adaptive_step_input_len", "adaptive_step_input_len_out"},
                    {"step_output_ids", "step_output_ids_out"},
                    {"step_output_len", "step_output_len_out"},
                    {"stop_flags", "stop_flags_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"mask_rollback", "mask_rollback_out"},
                    {"pre_ids", "pre_ids_out"},
                    {"step_idx", "step_idx_out"}})
    .SetKernelFn(PD_KERNEL(UnifiedUpdateModelStatus));
