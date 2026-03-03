// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

// Pure verification kernel — outputs accept_tokens + accept_num only.
// All state management (step_idx, stop_flags, EOS/max_dec_len detection)
// is handled by unified_update_model_status, so that both spec and non-spec
// paths share the same state update logic.

#include <curand_kernel.h>
#include "helper.h"  // NOLINT

// Persistent curand state — allocated once, reused across calls
static curandState_t *dev_curand_states = nullptr;
static int allocated_bsz = 0;
static uint64_t seed = 0;
static uint64_t offset = 0;

__device__ inline bool is_in(const int64_t *candidates,
                             const int64_t draft,
                             const int candidate_len) {
  for (int i = 0; i < candidate_len; i++) {
    if (draft == candidates[i]) {
      return true;
    }
  }
  return false;
}

__device__ int64_t topp_sampling_kernel(const int64_t *candidate_ids,
                                        const float *candidate_scores,
                                        curandState_t *curand_states,
                                        const int candidate_len,
                                        const float topp) {
  const int tid = threadIdx.x;

  float sum_scores = 0.0f;
  float rand_top_p = curand_uniform(curand_states + tid) * topp;
  for (int i = 0; i < candidate_len; i++) {
    sum_scores += candidate_scores[i];
    if (rand_top_p <= sum_scores) {
      return candidate_ids[i];
    }
  }
  return candidate_ids[0];
}

__global__ void setup_kernel(curandState_t *state,
                             const uint64_t seed,
                             const uint64_t offset,
                             const int bs,
                             const bool need_batch_random) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < bs; i += gridDim.x * blockDim.x) {
    if (need_batch_random) {
      curand_init(seed, i, offset, &state[i]);
    } else {
      curand_init(seed, 0, offset, &state[i]);
    }
  }
}

// Helper: write accepted token and check if it is an EOS token.
// Returns true if the token is EOS (caller should break the verify loop).
// Does NOT modify any global state (step_idx, stop_flags) — that is
// unified_update_model_status's responsibility.
__device__ inline bool accept_and_check_eos(int bid,
                                            int i,
                                            int64_t accept_token,
                                            int64_t *accept_tokens,
                                            const int64_t *end_tokens,
                                            int end_length,
                                            int max_draft_tokens) {
  accept_tokens[bid * max_draft_tokens + i] = accept_token;
  return is_in_end(accept_token, end_tokens, end_length);
}

// Pure verification kernel — only outputs accept_tokens and accept_num.
// No writes to step_idx or stop_flags.
__global__ void speculate_verify(
    const int64_t *sampled_token_ids,
    int64_t *accept_tokens,
    int *accept_num,
    const bool *stop_flags,
    const int *seq_lens_encoder,
    const int64_t *draft_tokens,
    curandState_t *curand_states,
    const float *topp,
    const int *seq_lens_this_time,
    const int64_t *verify_tokens,
    const float *verify_scores,
    const int64_t *end_tokens,
    const bool *is_block_step,
    const int *cu_seqlens_q_output,
    const int *actual_candidate_len,
    const int *reasoning_status,
    const int real_bsz,
    const int max_bsz,
    const int max_draft_tokens,
    const int end_length,
    const int max_seq_len,
    const int max_candidate_len,
    const int verify_window,
    // Strategy parameters (from SpeculativeConfig, no longer from env vars)
    const bool enable_topp,
    const bool use_topk,
    const bool use_target_sampling,
    const bool benchmark_mode,
    const bool accept_all_drafts) {
  const int bid = threadIdx.x;
  int accept_num_now = 1;
  bool stopped = false;  // local flag for EOS early-exit

  // Initialize accept_num to 0 for ALL slots (0..max_bsz), including slots
  // beyond real_bsz that may have stale data from previous rounds.
  // Active sequences will overwrite this with the correct value below.
  if (bid < max_bsz) {
    accept_num[bid] = 0;
  }

  if (!(is_block_step[bid] || bid >= real_bsz)) {
    const int start_token_id = cu_seqlens_q_output[bid];

    if (!stop_flags[bid]) {
      auto *verify_tokens_now =
          verify_tokens + start_token_id * max_candidate_len;
      auto *draft_tokens_now = draft_tokens + bid * max_draft_tokens;
      auto *actual_candidate_len_now = actual_candidate_len + start_token_id;
      auto *sampled_token_id_now = sampled_token_ids + start_token_id;

      // Phase 1: Verify draft tokens one by one
      int i = 0;
      for (; i < seq_lens_this_time[bid] - 1; i++) {
        if (benchmark_mode || seq_lens_encoder[bid] != 0 ||
            reasoning_status[bid] == 1) {
          break;
        }

        bool accepted = false;

        if (accept_all_drafts) {
          // Force accept all draft tokens
          if (accept_and_check_eos(bid,
                                   i,
                                   draft_tokens_now[i + 1],
                                   accept_tokens,
                                   end_tokens,
                                   end_length,
                                   max_draft_tokens)) {
            // EOS detected — keep this token, mark stopped, break
            accept_num_now++;
            stopped = true;
            break;
          }
          accept_num_now++;
          continue;
        } else if (use_target_sampling) {
          // Target sampling: compare sampled token with draft token
          accepted = (sampled_token_id_now[i] == draft_tokens_now[i + 1]);
        } else if (use_topk) {
          // Top-K: check if top-1 verify token matches draft
          accepted = (verify_tokens_now[i * max_candidate_len] ==
                      draft_tokens_now[i + 1]);
        } else {
          // Top-P: check if draft is in candidate set
          auto actual_cand_len = actual_candidate_len_now[i] > max_candidate_len
                                     ? max_candidate_len
                                     : actual_candidate_len_now[i];
          accepted = is_in(verify_tokens_now + i * max_candidate_len,
                           draft_tokens_now[i + 1],
                           actual_cand_len);

          if (!accepted) {
            // Top-K verify_window fallback: if top-2 matches, check
            // verify_window consecutive top-1 matches ahead
            int ii = i;
            if (max_candidate_len >= 2 &&
                verify_tokens_now[ii * max_candidate_len + 1] ==
                    draft_tokens_now[ii + 1]) {  // top-2 match
              int j = 0;
              ii += 1;
              for (; j < verify_window && ii < seq_lens_this_time[bid] - 1;
                   j++, ii++) {
                if (verify_tokens_now[ii * max_candidate_len] !=
                    draft_tokens_now[ii + 1]) {
                  break;
                }
              }
              if (j >= verify_window) {
                // Bulk accept: top-2 + verify_window consecutive top-1 matches
                // Write all accepted tokens and check EOS along the way
                for (; i < ii; i++) {
                  auto accept_token = draft_tokens_now[i + 1];
                  accept_tokens[bid * max_draft_tokens + i] = accept_token;
                  accept_num_now++;
                  if (is_in_end(accept_token, end_tokens, end_length)) {
                    stopped = true;
                    break;
                  }
                }
                if (stopped) break;
                // Continue outer loop from position ii
                // (i is already at ii after the inner for-loop)
              }
            }
            break;  // reject: exit verify loop
          }
        }

        if (accepted) {
          if (accept_and_check_eos(bid,
                                   i,
                                   draft_tokens_now[i + 1],
                                   accept_tokens,
                                   end_tokens,
                                   end_length,
                                   max_draft_tokens)) {
            // EOS detected — keep this token, mark stopped, break
            accept_num_now++;
            stopped = true;
            break;
          }
          accept_num_now++;
        } else {
          break;  // reject: exit verify loop
        }
      }

      // Phase 2: Sample a token for the rejected/last position
      // Skip if EOS was already detected in Phase 1
      if (!stopped) {
        int64_t accept_token;
        const float *verify_scores_now =
            verify_scores + start_token_id * max_candidate_len;

        if (use_target_sampling) {
          accept_token = sampled_token_id_now[i];
        } else if (enable_topp) {
          auto actual_cand_len = actual_candidate_len_now[i] > max_candidate_len
                                     ? max_candidate_len
                                     : actual_candidate_len_now[i];
          accept_token =
              topp_sampling_kernel(verify_tokens_now + i * max_candidate_len,
                                   verify_scores_now + i * max_candidate_len,
                                   curand_states,
                                   actual_cand_len,
                                   topp[bid]);
        } else {
          accept_token = verify_tokens_now[i * max_candidate_len];
        }

        accept_tokens[bid * max_draft_tokens + i] = accept_token;
        // EOS detection for Phase 2 token is handled by unified_update
      }
      accept_num[bid] = accept_num_now;
    }
  }
}

void SpeculateVerify(const paddle::Tensor &sampled_token_ids,
                     const paddle::Tensor &accept_tokens,
                     const paddle::Tensor &accept_num,
                     const paddle::Tensor &stop_flags,
                     const paddle::Tensor &seq_lens_encoder,
                     const paddle::Tensor &draft_tokens,
                     const paddle::Tensor &seq_lens_this_time,
                     const paddle::Tensor &verify_tokens,
                     const paddle::Tensor &verify_scores,
                     const paddle::Tensor &end_tokens,
                     const paddle::Tensor &is_block_step,
                     const paddle::Tensor &cu_seqlens_q_output,
                     const paddle::Tensor &actual_candidate_len,
                     const paddle::Tensor &topp,
                     const paddle::Tensor &reasoning_status,
                     int max_seq_len,
                     int verify_window,
                     bool enable_topp,
                     bool benchmark_mode,
                     bool accept_all_drafts,
                     bool use_topk,
                     bool use_target_sampling) {
  auto bsz = accept_tokens.shape()[0];
  int real_bsz = seq_lens_this_time.shape()[0];
  auto max_draft_tokens = draft_tokens.shape()[1];
  auto end_length = end_tokens.shape()[0];
  auto max_candidate_len = verify_tokens.shape()[1];

  constexpr int BlockSize = 512;
  auto stream = accept_tokens.stream();

  // Persistent curand state: allocate once, reuse across calls
  if (dev_curand_states == nullptr || bsz > allocated_bsz) {
    if (dev_curand_states) cudaFree(dev_curand_states);
    cudaMalloc(&dev_curand_states, sizeof(curandState_t) * bsz);
    allocated_bsz = bsz;
  }
  setup_kernel<<<1, BlockSize, 0, stream>>>(
      dev_curand_states, seed, offset, bsz, true);
  seed++;
  offset++;

  // Single kernel launch
  speculate_verify<<<1, BlockSize, 0, stream>>>(
      sampled_token_ids.data<int64_t>(),
      const_cast<int64_t *>(accept_tokens.data<int64_t>()),
      const_cast<int *>(accept_num.data<int>()),
      stop_flags.data<bool>(),
      seq_lens_encoder.data<int>(),
      draft_tokens.data<int64_t>(),
      dev_curand_states,
      topp.data<float>(),
      seq_lens_this_time.data<int>(),
      verify_tokens.data<int64_t>(),
      verify_scores.data<float>(),
      end_tokens.data<int64_t>(),
      is_block_step.data<bool>(),
      cu_seqlens_q_output.data<int>(),
      actual_candidate_len.data<int>(),
      reasoning_status.data<int>(),
      real_bsz,
      bsz,  // max_bsz
      max_draft_tokens,
      end_length,
      max_seq_len,
      max_candidate_len,
      verify_window,
      enable_topp,
      use_topk,
      use_target_sampling,
      benchmark_mode,
      accept_all_drafts);
}

PD_BUILD_STATIC_OP(speculate_verify)
    .Inputs({"sampled_token_ids",
             "accept_tokens",
             "accept_num",
             "stop_flags",
             "seq_lens_encoder",
             "draft_tokens",
             "seq_lens_this_time",
             "verify_tokens",
             "verify_scores",
             "end_tokens",
             "is_block_step",
             "cu_seqlens_q_output",
             "actual_candidate_len",
             "topp",
             "reasoning_status"})
    .Outputs({"accept_tokens_out", "accept_num_out"})
    .Attrs({"max_seq_len: int",
            "verify_window: int",
            "enable_topp: bool",
            "benchmark_mode: bool",
            "accept_all_drafts: bool",
            "use_topk: bool",
            "use_target_sampling: bool"})
    .SetInplaceMap({{"accept_tokens", "accept_tokens_out"},
                    {"accept_num", "accept_num_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateVerify));
