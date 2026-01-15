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
                "arctic_inference.suffix_decoding is not available. "
                "Please install arctic-inference package."
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

    def start_request(self, req_id: str, prompt_token_ids: list[int]):
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
        """
        Generate draft tokens using suffix decoding.
        
        Args:
            share_inputs: Dictionary containing input tensors
        """
        draft_tokens = share_inputs["draft_tokens"].cpu()
        seq_lens_this_time = share_inputs["seq_lens_this_time"].cpu()
        seq_lens_encoder = share_inputs["seq_lens_encoder"].cpu()
        seq_lens_decoder = share_inputs["seq_lens_decoder"].cpu()
        input_ids_cpu = share_inputs["input_ids_cpu"].cpu()
        stop_flags = share_inputs["stop_flags"].cpu()
        accept_tokens = share_inputs["accept_tokens"].cpu()
        accept_num = share_inputs["accept_num"].cpu()
        logger.info(f"suffix _run_impl:")

        batch_size = draft_tokens.shape[0]
        
        # Process stop_flags first - stop requests that are finished
        # Use vectorized operations to find stopped requests
        stop_flags_np = stop_flags.numpy().flatten()
        for bid in range(batch_size):
            if stop_flags_np[bid]:
                req_id = self.idx_to_req_id.get(bid, None)
                if req_id is not None and req_id in self.suffix_cache.active_requests:
                    self.stop_request(req_id)
            elif (accept_num[bid] > 0):
                req_id = self.idx_to_req_id.get(bid, None)
                if req_id is not None and req_id in self.suffix_cache.active_requests:
                    token_ids = accept_tokens[bid, :accept_num[bid]]
                    self.add_active_response(req_id, [token_ids])
                
        #     for bid in range(batch_size):
        #         if not stop_flags[bid]:
        #             req_id = self.proposer.idx_to_req_id.get(bid, None)
        #             if req_id is not None and req_id in self.proposer.suffix_cache.active_requests:
        #                 token_id = int(sampled_ids[bid, 0])
        #                 self.proposer.add_active_response(req_id, [token_id])
        # Pre-compute valid batch indices to reduce loop iterations
        # Vectorized operations for sequence lengths
        encoder_lens = seq_lens_encoder.numpy().flatten().astype(np.int32)
        decoder_lens = seq_lens_decoder.numpy().flatten().astype(np.int32)
        total_lens = encoder_lens + decoder_lens
        seq_lens_this_time_np = seq_lens_this_time.numpy().flatten()
        
        # Filter valid batch indices: not stopped, has mapping, active, valid length
        valid_bids = []
        for bid in range(batch_size):
            if stop_flags_np[bid] or seq_lens_this_time_np[bid] <= 0:
                continue
            req_id = self.idx_to_req_id.get(bid, None)
            if req_id is None or req_id not in self.suffix_cache.active_requests:
                continue
            if total_lens[bid] >= self.max_model_len:
                continue
            valid_bids.append(bid)
        logger.info(f"valid_bids: {valid_bids}")

        # Process valid batches
        for bid in valid_bids:
            num_tokens = total_lens[bid]
            
            # Extract context from the end of the sequence (up to max_tree_depth)
            start = max(0, num_tokens - self.max_tree_depth)
            context_slice = input_ids_cpu[bid, start:num_tokens]
            context_tokens = context_slice.numpy()
            
            # Remove padding tokens (-1) using vectorized operation
            valid_mask = context_tokens >= 0
            context_tokens = context_tokens[valid_mask].astype(np.int32)
            
            # Skip if context is empty
            if len(context_tokens) == 0:
                draft_tokens[bid, 1:] = -1
                seq_lens_this_time[bid, 0] = 1
                continue
            
            # Ensure contiguous
            if not context_tokens.flags["CONTIGUOUS"]:
                context_tokens = np.ascontiguousarray(context_tokens)
            
            # Calculate max_spec_tokens (limited by remaining length)
            max_spec_tokens = min(
                self.max_draft_token_num,
                self.max_model_len - num_tokens - 1
            )
            
            if max_spec_tokens <= 0:
                draft_tokens[bid, 1:] = -1
                seq_lens_this_time[bid, 0] = 0
                continue
            
            # Generate draft tokens using suffix decoding
            req_id = self.idx_to_req_id[bid]
            # try:
            draft = self.suffix_cache.speculate(
                req_id,
                context_tokens,
                max_spec_tokens=max_spec_tokens,
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )
            logger.info(f"req: {req_id}")
            logger.info(f"context_tokens: {context_tokens}")
            logger.info(f"max_spec_tokens: {max_spec_tokens}")
            logger.info(f"self.max_spec_factor,: {self.max_spec_factor,}")

            logger.info(f"self.min_token_prob,: {self.min_token_prob,}")
            
            logger.info(f"draft,: {draft}")

            draft_token_list = draft.token_ids
            num_draft_tokens = len(draft_token_list)
            
            if num_draft_tokens > 0:
                # Write draft tokens back to share_inputs
                num_to_write = min(num_draft_tokens, self.max_draft_token_num)
                draft_tokens[bid, :num_to_write] = paddle.to_tensor(
                    draft_token_list[:num_to_write], dtype="int64"
                )
                if num_to_write < self.max_draft_token_num:
                    draft_tokens[bid, num_to_write:] = -1
                seq_lens_this_time[bid, 0] = num_to_write
            else:
                # No draft tokens generated
                draft_tokens[bid, 1:] = -1
                seq_lens_this_time[bid, 0] = 1
                    
            # except Exception:
            #     # If speculation fails, set empty draft tokens
            #     draft_tokens[bid, :] = -1
            #     seq_lens_this_time[bid, 0] = 0
        
        # Copy results back to GPU
        share_inputs["draft_tokens"][:] = draft_tokens.cuda()
        share_inputs["seq_lens_this_time"][:] = seq_lens_this_time.cuda()

    def update_request_mapping(self, req_id: str, idx: int):
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

