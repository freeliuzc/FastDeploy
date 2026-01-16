"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import numpy as np
import paddle
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig

from .base import Proposer

try:
    from arctic_inference.suffix_decoding import SuffixDecodingCache
except ImportError:
    SuffixDecodingCache = None


class SuffixProposer(Proposer):
    """
    Proposer for Suffix Decoding method.

    Uses SuffixDecodingCache to generate draft tokens based on suffix tree matching.
    """

    def __init__(self, fd_config: FDConfig):
        super().__init__(fd_config)

        if SuffixDecodingCache is None:
            raise ImportError(
                "arctic_inference.suffix_decoding is not available. " "Please install arctic-inference package."
            )

        # Initialize SuffixDecodingCache
        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=self.speculative_config.suffix_decoding_max_tree_depth,
            max_cached_requests=self.speculative_config.suffix_decoding_max_cached_requests,
        )

        self.max_tree_depth = self.speculative_config.suffix_decoding_max_tree_depth
        self.max_spec_factor = self.speculative_config.suffix_decoding_max_spec_factor
        self.min_token_prob = self.speculative_config.suffix_decoding_min_token_prob

        # Track active requests: req_id -> idx mapping
        self.req_id_to_idx = {}
        self.idx_to_req_id = {}
        self.context_tokens = paddle.full(
            shape=[self.max_num_seqs, self.max_model_len],
            fill_value=-1,
            dtype="int64",
        ).cpu()

    def start_request(self, idx: int, req_id: str, prompt_token_ids: list[int]):
        """
        Start a new request in the suffix cache.

        Args:
            req_id: Request identifier
            prompt_token_ids: List of prompt token IDs
        """
        if req_id in self.suffix_cache.active_requests:
            # Request already active, skip
            return

        # Convert to numpy array (int32, contiguous)
        prompt_array = np.array(prompt_token_ids, dtype=np.int32)
        if not prompt_array.flags["CONTIGUOUS"]:
            prompt_array = np.ascontiguousarray(prompt_array)
        logger.info(f"Starting request {req_id}. Prompt: {prompt_array}")

        self.context_tokens[idx, :] = -1
        self.context_tokens[idx, : len(prompt_token_ids)] = prompt_array
        self._update_request_mapping(idx, req_id)
        self.suffix_cache.start_request(req_id, prompt_array)

    def stop_request(self, req_id: str):
        """
        Stop a request in the suffix cache.

        Args:
            req_id: Request identifier
        """
        if req_id in self.suffix_cache.active_requests:
            self.suffix_cache.stop_request(req_id)

        # Clean up mappings
        if req_id in self.req_id_to_idx:
            idx = self.req_id_to_idx[req_id]
            del self.req_id_to_idx[req_id]
            if idx in self.idx_to_req_id:
                del self.idx_to_req_id[idx]

    def add_active_response(self, req_id: str, token_ids: list[int]):
        """
        Add newly sampled tokens to the suffix cache for a request.

        Args:
            req_id: Request identifier
            token_ids: List of newly sampled token IDs
        """
        if req_id not in self.suffix_cache.active_requests:
            return

        # Convert to numpy array (int32, contiguous)
        token_array = np.array(token_ids, dtype=np.int32)
        if not token_array.flags["CONTIGUOUS"]:
            token_array = np.ascontiguousarray(token_array)

        self.suffix_cache.add_active_response(req_id, token_array)

    def _run_impl(self, share_inputs):
        draft_tokens = share_inputs["draft_tokens"]
        seq_lens_this_time = share_inputs["seq_lens_this_time"]

        stop_flags = share_inputs["stop_flags"].cpu().numpy().flatten()
        accept_tokens = share_inputs["accept_tokens"].cpu()
        accept_num = share_inputs["accept_num"].cpu()
        seq_lens_encoder = share_inputs["seq_lens_encoder"].cpu().numpy().flatten().astype(np.int32)
        seq_lens_decoder = share_inputs["seq_lens_decoder"].cpu().numpy().flatten().astype(np.int32)

        total_lens = seq_lens_encoder + seq_lens_decoder
        batch_size = seq_lens_this_time.shape[0]

        draft_tokens_cpu = draft_tokens.cpu()

        for bid in range(batch_size):
            # ---------- 1. stop 优先级最高 ----------
            if stop_flags[bid]:
                req_id = self.idx_to_req_id.get(bid)
                if req_id is not None and req_id in self.suffix_cache.active_requests:
                    self.stop_request(req_id)

                seq_lens_this_time[bid, 0] = 0
                draft_tokens_cpu[bid, :] = -1
                continue

            # ---------- 2. 非 stop：兜底处理 ----------
            req_id = self.idx_to_req_id.get(bid)
            if req_id is None:
                # 没有映射，但还在跑：不给 speculate，保证行为可预期
                seq_lens_this_time[bid, 0] = 1
                draft_tokens_cpu[bid, 1:] = -1
                continue

            # ---------- 3. accept ----------
            acc_n = int(accept_num[bid])
            if acc_n > 0 and req_id in self.suffix_cache.active_requests:
                token_ids = accept_tokens[bid, :acc_n]
                ctx_start = seq_lens_decoder[bid] - acc_n
                self.context_tokens[bid, ctx_start : ctx_start + acc_n] = token_ids
                self.add_active_response(req_id, token_ids)

            seq_lens_this_time[bid, 0] = 1

            # ---------- 4. 非 active request ----------
            if req_id not in self.suffix_cache.active_requests:
                draft_tokens_cpu[bid, 1:] = -1
                continue

            num_tokens = total_lens[bid]
            if num_tokens >= self.max_model_len:
                draft_tokens_cpu[bid, 1:] = -1
                continue

            # ---------- 5. context ----------
            start = max(0, num_tokens - self.max_tree_depth)
            ctx = self.context_tokens[bid, start:num_tokens].numpy()
            ctx = ctx[ctx >= 0]

            if ctx.size == 0:
                draft_tokens_cpu[bid, 1:] = -1
                continue

            # if not ctx.flags["CONTIGUOUS"]:
            ctx = np.ascontiguousarray(ctx, dtype=np.int32)
            # else:
            #     ctx = ctx.astype(np.int32, copy=False)

            max_spec_tokens = min(
                self.max_draft_token_num,
                self.max_model_len - num_tokens - 1,
            )
            if max_spec_tokens <= 1:
                draft_tokens_cpu[bid, 1:] = -1
                continue

            # ---------- 6. speculate ----------
            draft = self.suffix_cache.speculate(
                req_id,
                ctx,
                max_spec_tokens=max_spec_tokens,
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )

            token_ids = draft.token_ids
            n = min(len(token_ids), self.max_draft_token_num)

            if n > 0:
                draft_tokens_cpu[bid, 1 : 1 + n] = paddle.to_tensor(token_ids[:n], dtype="int64")
                draft_tokens_cpu[bid, 1 + n :] = -1
                seq_lens_this_time[bid, 0] = 1 + n
            else:
                draft_tokens_cpu[bid, 1:] = -1

        share_inputs["draft_tokens"][:] = draft_tokens_cpu.cuda()

    def _update_request_mapping(self, idx: int, req_id: str):
        """
        Update the mapping between request ID and batch index.

        Args:
            req_id: Request identifier
            idx: Batch index
        """
        # Clean up old mapping if exists
        if idx in self.idx_to_req_id:
            old_req_id = self.idx_to_req_id[idx]
            if old_req_id in self.req_id_to_idx:
                del self.req_id_to_idx[old_req_id]

        # Set new mapping
        self.req_id_to_idx[req_id] = idx
        self.idx_to_req_id[idx] = req_id

        logger.info(f"self.req_id_to_idx: {self.req_id_to_idx}, self.idx_to_req_id: {self.idx_to_req_id}")
