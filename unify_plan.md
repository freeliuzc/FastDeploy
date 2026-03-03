# FastDeploy 阶段一 Spec 分支重构 — 后处理优化 + Verify 重构

## Context

FastDeploy 中 Speculative Decoding 的后处理流程当前调用 4 个独立 CUDA kernel（`speculate_set_stop_value_multi_seqs` → `speculate_update` → `speculate_save_output` → `speculate_set_value_by_flags_and_idx`），存在：

1. **多次 kernel launch 开销**：3 个可合并的 kernel 导致不必要的同步和调度开销
2. **accept_num/step_idx 在多处修改/回退**：`speculate_update` 修改 `seq_lens_decoder`、`mask_rollback`，`speculate_set_value_by_flags_and_idx` 再次读写 `accept_num`、`seq_lens_decoder`，存在隐式数据依赖
3. **Verify kernel 代码分支复杂**：4 个模板组合（`ENABLE_TOPP × USE_TOPK`）+ `use_target_sampling` + `accept_all_drafts` + `benchmark_mode`，调用方重复 4 次 launch 代码
4. **Verify 中每次 `cudaMalloc/cudaFree` curand state**：性能浪费

本方案的目标是：

- 合并 `speculate_update` + `speculate_set_value_by_flags_and_idx` 为 `unified_update_model_status`
- 重构 `speculate_verify.cu`：整合分支、规范代码架构、消除运行时 `cudaMalloc`
- 支持 `method="naive"` 走 spec 路径退化为普通解码
- 仅涉及 GPU (CUDA) 平台

## 设计原则

1. **变量命名兼容双语义**：命名需同时满足 spec 与非 spec 的语义，为后期 unify（阶段二）做准备。不再使用 spec-only 术语（如 `accept_tokens`、`draft_tokens`），而是使用两种场景都成立的中性名称
2. **Spec 分支支持 method="naive"**：当 `speculative_config.method = "naive"` 时，走 spec 代码路径但 `num_speculative_tokens=0`，无 proposer，系统自然退化为普通解码。这验证了设计文档的核心原则："Naive 是 Spec 的特例"
3. **save_output 异步逻辑保护**：后续需要支持 Spec 分支的 overlap scheduling（延迟 save_output），因此合并后处理算子时，必须保持 `save_output` 的调用位置和数据依赖关系，为异步化留出空间
4. **seq_lens_this_time 条件化写入**：`unified_update_model_status` 通过 `is_naive_mode` Attr 控制是否重置 `seq_lens_this_time`。MTP/Ngram 模式（`is_naive_mode=false`）不修改此值，保持与原始 `speculate_update`（`const int*`）一致的行为；Naive 模式（`is_naive_mode=true`）重置为 1/0
5. **生成设计文档**：实现完成后输出详细设计文档

## save_output 异步约束（关键）

当前 save_output 的执行时序有严格约束：

**Spec 路径**（`post_process_specualate`）：

```
speculate_set_stop_value_multi_seqs  → 确定 stop_flags/accept_num
unified_update_model_status          → 更新 seq_lens/pre_ids/seq_lens_this_time（消费 accept_num）
save_output                          → 读取 accept_tokens/accept_num 发送输出
```

save_output **必须在** unified_update_model_status 之后执行（因为 `accept_num` 和 `stop_flags` 已 finalized），但**必须在** 下一步的 pre_ids 写回之前（合并后 pre_ids 写入在 unified_update 内完成）。

**当前实现**：Spec 分支**不支持 overlap scheduling**（`enable_overlap_schedule = False`），save_output 是同步内联在 `post_process_specualate` 中。

**本次实现**：save_output 保持原位调用不变，但确保数据依赖清晰，不引入新的隐式依赖。

------

## 任务一：后处理算子合并 → `unified_update_model_status`（已完成）

### 1.1 合并范围

将 `speculate_update` + `speculate_set_value_by_flags_and_idx` 合并为单次 kernel launch。同时承担 `seq_lens_this_time` 的重置职责。

`speculate_set_stop_value_multi_seqs`（stop 检查）和 `speculate_save_output`（输出发送）保持独立。

### 1.2 Kernel 设计

**文件**：`custom_ops/gpu_ops/speculate_decoding/unified_update_model_status.cu`

```cpp
template <int THREADBLOCK_SIZE>
__global__ void unified_update_model_status_kernel(
    // == 序列状态（读写）==
    int *seq_lens_encoder,
    int *seq_lens_decoder,
    bool *has_running_seqs,               // 原 not_need_stop
    int *mask_rollback,
    // == 输入 / 输出（读写）==
    int64_t *step_input_ids,              // 原 draft_tokens
    int *adaptive_step_input_len,         // 原 actual_draft_token_nums
    // == 本步输出（只读）==
    const int64_t *step_output_ids,       // 原 accept_tokens
    const int *step_output_len,           // 原 accept_num
    // == 控制标志 ==
    const bool *stop_flags,
    int *seq_lens_this_time,              // 读写：读取当前值，写入下一步值
    const bool *is_paused,               // 原 is_block_step
    // == 历史记录（读写）==
    int64_t *pre_ids,
    const int64_t *step_idx,
    // == 维度参数 ==
    const int real_bsz, const int max_bsz,
    const int max_step_tokens, const int pre_ids_len,
    const bool is_naive_mode)               // 控制是否写入 seq_lens_this_time
```

每个线程处理一个 batch element，逻辑：

1. **Stopped**：`mask_rollback = 0`
2. **Decoder 阶段**（`seq_lens_encoder == 0`）：
   - `seq_lens_decoder += output_len`
   - `mask_rollback = seq_lens_this_time - output_len`
   - 自适应调整 `adaptive_step_input_len`
3. **Encoder (prefill) 阶段**：`mask_rollback = 0`
4. **Encoder→Decoder 转换**：`seq_lens_decoder += seq_lens_encoder; seq_lens_encoder = 0`
5. **写入 pre_ids 历史**（**必须在步骤 4 之后**）：将 `step_output_ids` 写入 `pre_ids`。条件 `!stop_flags && step_idx > 0`。此步骤放在 encoder→decoder 转换之后，确保刚完成 prefill 的序列也能写入历史（匹配原 `speculate_set_value_by_flags_and_idx` 在 `speculate_update` 之后执行的语义）
6. **写回下一步输入**：`step_input_ids[0] = step_output_ids[output_len-1]`
7. **条件重置 seq_lens_this_time**（仅 `is_naive_mode=true`）：`seq_lens_this_time[bid] = stop_flags[bid] ? 0 : 1`
   - MTP/Ngram 模式（`is_naive_mode=false`）：不修改，保留原始值供下游 `draft_model_preprocess` / `eagle_get_hidden_states` 读取
   - Naive 模式（`is_naive_mode=true`）：直接使用此值，无 proposer 覆盖
8. **has_running_seqs 归约**：CUB BlockReduce

**关键设计**：
- `step_output_ids` / `step_output_len` 为 `const` 只读，save_output 在 kernel 之后安全读取
- `seq_lens_this_time` 的写入受 `is_naive_mode` 控制。原始 `speculate_update` 用 `const int*` 不修改此值，MTP 下游 kernel（`draft_model_preprocess`、`eagle_get_hidden_states`）依赖未修改的原始值。无条件写入会导致接受率降至 ~1.07
- `pre_ids` 写入必须在 encoder→decoder 转换之后执行（Bug fix: 原实现将 pre_ids 写入放在 decoder 分支内，导致 prefill 序列的历史记录缺失，造成接受率异常）

### 1.3 Python 端修改

**文件**：`fastdeploy/model_executor/pre_and_post_process.py`

`post_process_specualate()` 中替换调用：

```python
# 原来 3 步 → 现在 2 步
# Step 1: unified_update_model_status（合并 speculate_update + set_value_by_flags_and_idx）
unified_update_model_status(
    model_output.seq_lens_encoder,       # seq_lens_encoder
    model_output.seq_lens_decoder,       # seq_lens_decoder
    model_output.not_need_stop,          # has_running_seqs
    model_output.draft_tokens,           # step_input_ids
    model_output.actual_draft_token_num, # adaptive_step_input_len
    model_output.accept_tokens,          # step_output_ids (read-only)
    model_output.accept_num,             # step_output_len (read-only)
    model_output.stop_flags,             # stop_flags
    model_output.seq_lens_this_time,     # seq_lens_this_time (条件读写)
    model_output.is_block_step,          # is_paused
    model_output.mask_rollback,          # mask_rollback
    model_output.pre_ids,               # pre_ids
    model_output.step_idx,              # step_idx
    is_naive_mode,                       # is_naive_mode: MTP→False, naive→True
)
# Step 2: save_output（保持不变）
```

### 1.4 Op 注册

**PD_BUILD_STATIC_OP**（`unified_update_model_status.cu`）：

- Inputs: 13 个 Tensor
- Attrs: `is_naive_mode: bool`
- Outputs: 8 个（seq_lens_encoder, seq_lens_decoder, has_running_seqs, step_input_ids, adaptive_step_input_len, seq_lens_this_time, mask_rollback, pre_ids）
- InplaceMap: 8 对

**cpp_extensions.cc**：

- 前向声明 `UnifiedUpdateModelStatus(...)` 13 个 Tensor 参数 + 1 个 bool 参数
- `m.def("unified_update_model_status", &UnifiedUpdateModelStatus, ...)`（pybind11 自动匹配新签名）

### 1.5 `seq_lens_this_time` 更新链路

这是 naive 模式能正确工作的关键。完整的更新链路：

```
insert_tasks_v1()     → prefill: seq_lens_this_time = prompt_length
                        decode: 不写（沿用上一步值）
speculate_verify()    → 只读，不修改
unified_update()      → is_naive_mode=true 时写入 1/0；is_naive_mode=false 时不修改（与原 speculate_update const int* 行为一致）
proposer.run()        → MTP: draft_model_postprocess 覆盖为 draft_count+1
                        Ngram: ngram_match 覆盖为 match_count+1
                        Naive: proposer=None，不运行，使用 kernel 写入的 1
speculate_step_*()    → 仅 block management 场景修改（steal→0, recover→seq_len）
```

**is_naive_mode 根因修复**：原始 `speculate_update` 将 `seq_lens_this_time` 声明为 `const int*`（line 26），不会修改。`unified_update_model_status` 如果无条件写入 `seq_lens_this_time = stop_flags ? 0 : 1`，会在 MTP proposer 调用 `draft_model_preprocess` / `eagle_get_hidden_states` 之前破坏原始值，导致接受率降至 ~1.07。通过 `is_naive_mode` Attr 条件化控制，MTP 模式传 `false` 不写入，同时修复接受率问题并支持 naive 模式。

### 1.6 文件清单

| 操作 | 文件 | 状态 |
|---|---|---|
| 新增 | `custom_ops/gpu_ops/speculate_decoding/unified_update_model_status.cu` | ✅ 完成 |
| 修改 | `custom_ops/setup_ops.py` — 编译注册 | ✅ 完成 |
| 修改 | `custom_ops/gpu_ops/cpp_extensions.cc` — 前向声明 + pybind11 注册 | ✅ 完成 |
| 修改 | `fastdeploy/model_executor/pre_and_post_process.py` — import + 调用替换 | ✅ 完成 |

------

## 任务二：Verify 重构（CUDA + Python + Config 三层联动）（已完成）

### 2.1 Config 层：统一验证策略到 `SpeculativeConfig`

**文件**：`fastdeploy/config.py`

新增 3 个字段：

```python
self.verify_strategy: str = "topp"       # "topk" | "topp" | "target_sampling"
self.prefill_one_step_stop: bool = False  # 原 PREFILL_NODE_ONE_STEP_STOP 环境变量
self.accept_policy: str = "normal"        # "normal" | "accept_all" | "reject_all"
```

**向后兼容**：`reset()` 中读取环境变量作为 fallback：

```python
if os.environ.get("SPECULATE_VERIFY_USE_TOPK", "0") == "1":
    self.verify_strategy = "topk"
if os.environ.get("SPECULATE_VERIFY_USE_TARGET_SAMPLING", "0") == "1":
    self.verify_strategy = "target_sampling"
if os.environ.get("PREFILL_NODE_ONE_STEP_STOP", "0") == "1":
    self.prefill_one_step_stop = True
```

**参数校验**：`check_legality_parameters()` 验证 `verify_strategy` 和 `accept_policy` 取值。

### 2.2 Python Sampler 层

**文件**：`fastdeploy/model_executor/layers/sample/sampler.py`

`SpeculativeSampler.__init__` 从 config 读取策略：

```python
self.use_topk = (spec_config.verify_strategy == "topk")
self.use_target_sampling = (spec_config.verify_strategy == "target_sampling")
self.enable_topp = (spec_config.verify_strategy in ("topp", "topk"))
self.prefill_one_step_stop = spec_config.prefill_one_step_stop
self.config_accept_all = (spec_config.accept_policy == "accept_all")
self.config_reject_all = (spec_config.accept_policy == "reject_all")
```

`forward_cuda` 传参：

```python
final_accept_all = self.config_accept_all or accept_all_drafts
final_reject_all = self.config_reject_all or reject_all_drafts or self.speculative_benchmark_mode

speculate_verify(
    ...,
    self.enable_topp,              # 原来硬编码 True
    final_reject_all,              # 合并 benchmark_mode + config + 参数
    final_accept_all,              # 合并 config + 参数
    self.use_topk,                 # 新增
    self.use_target_sampling,      # 新增
    self.prefill_one_step_stop,    # 新增
)
```

### 2.3 CUDA Kernel 层

**文件**：`custom_ops/gpu_ops/speculate_decoding/speculate_verify.cu`

| 改动 | 说明 |
|---|---|
| 消除模板 | `template <ENABLE_TOPP, USE_TOPK>` → 单个非模板 kernel，runtime bool |
| 消除 getenv | 删除 3 个 `getenv()` 调用，参数从 Python config 传入 |
| 持久化 curand | `static curandState_t*` 持久分配，不再每次 `cudaMalloc/cudaFree` |
| 简化 dispatch | 4 段重复 kernel launch → 1 段 |
| 抽取 helper | `accept_and_check_stop()` device inline 函数 |
| 新增 Attrs | `use_topk: bool`, `use_target_sampling: bool`, `prefill_one_step_stop: bool` |

**代码量**：531 行 → 406 行（−125 行），消除约 200 行重复参数传递。

**Host 函数签名**：

```cpp
void SpeculateVerify(...,
    int max_seq_len, int verify_window,
    bool enable_topp, bool benchmark_mode, bool accept_all_drafts,
    bool use_topk, bool use_target_sampling, bool prefill_one_step_stop);
```

**cpp_extensions.cc**：前向声明同步更新，新增 3 个 bool 参数。

### 2.4 文件清单

| 操作 | 文件 | 状态 |
|---|---|---|
| 修改 | `fastdeploy/config.py` — 新增字段 + 环境变量 fallback + 校验 | ✅ 完成 |
| 修改 | `fastdeploy/model_executor/layers/sample/sampler.py` — init + forward_cuda | ✅ 完成 |
| 重写 | `custom_ops/gpu_ops/speculate_decoding/speculate_verify.cu` | ✅ 完成 |
| 修改 | `custom_ops/gpu_ops/cpp_extensions.cc` — 签名更新 | ✅ 完成 |

------

## 任务三：method="naive" 支持（已完成）

### 3.1 设计

当 `speculative_config.method = "naive"` 时：

- `speculative_decoding = True`，走 spec 代码路径
- `proposer = None`，不产生 draft tokens
- `is_naive_mode = True` 传入 `unified_update_model_status` kernel
- kernel 内 `seq_lens_this_time` 被重置为 1（active decode）或 0（stopped）
- `speculate_verify` 内循环 `for i < 1-1` 不执行 → 直接走 Phase 2 采样
- `step_output_len` 恒为 1，行为等同于普通 Sampler

### 3.2 Config 层

**文件**：`fastdeploy/config.py`

- `method_list` 添加 `"naive"`
- `check_legality_parameters()` 中 `method="naive"` 跳过 `num_speculative_tokens` 范围校验

### 3.3 gpu_model_runner 修改

**文件**：`fastdeploy/worker/gpu_model_runner.py`

`_init_speculative_proposer()`：

```python
elif self.speculative_method == "naive":
    self.proposer = None  # naive 模式无 proposer
```

`_postprocess()` proposer 调用加 None guard，并传递 `is_naive_mode`：

```python
# post_process 传入 is_naive_mode
post_process_specualate(
    ...,
    is_naive_mode=(self.proposer is None),
)

if self.speculative_decoding and self.proposer is not None:
    if self.speculative_method == "mtp":
        self.proposer.run(...)
    else:
        self.proposer.run(share_inputs=self.share_inputs)
# naive 模式: proposer=None, seq_lens_this_time 已在 unified_update kernel 内重置为 1
```

### 3.4 pre_and_post_process.py 修改

**文件**：`fastdeploy/model_executor/pre_and_post_process.py`

`post_process_specualate()` 签名新增 `is_naive_mode: bool = False`，透传给 `unified_update_model_status()` 调用。

### 3.5 文件清单

| 操作 | 文件 | 状态 |
|---|---|---|
| 修改 | `custom_ops/gpu_ops/speculate_decoding/unified_update_model_status.cu` — `is_naive_mode` Attr + 条件写入 | ✅ 完成 |
| 修改 | `fastdeploy/config.py` — method_list + check_legality | ✅ 完成 |
| 修改 | `fastdeploy/model_executor/pre_and_post_process.py` — `is_naive_mode` 参数透传 | ✅ 完成 |
| 修改 | `fastdeploy/worker/gpu_model_runner.py` — proposer init + None guard + `is_naive_mode` 传递 | ✅ 完成 |

------

任务四：Verify 职责拆分 — 状态管理下沉到 unified_update（已完成）
Context
当前 speculate_verify 同时承担两个职责：

纯验证：draft token 是否被 target 模型接受 → 输出 accept_tokens、accept_num
状态管理：step_idx++、EOS/max_dec_len 检测 → 设 stop_flags、替换 end_tokens[0]
对于阶段二 unify（spec + 非 spec 共用 unified_update_model_status），非投机解码不经过 speculate_verify，但必须共用状态管理逻辑。因此需将 EOS/stop/step_idx 等状态管理下沉到 unified_update_model_status。

speculate_set_stop_value_multi_seqs（多 token 停止序列）保持独立 kernel，不纳入此次变更。

当前流程

speculate_verify  → 验证 + EOS/stop + step_idx++ + stop_flags 写入
  ↓
speculate_set_stop_value_multi_seqs → 多 token 停止序列匹配（可能截断 accept_num，设 stop_flags）
  ↓
unified_update_model_status → seq_lens_decoder/encoder/pre_ids/mask_rollback/has_running_seqs
  ↓
save_output → 读取 accept_tokens/accept_num
目标流程

speculate_verify  → 纯验证：输出 accept_tokens + accept_num（不写 step_idx/stop_flags）
  ↓
speculate_set_stop_value_multi_seqs → 多 token 停止序列匹配（保持不变）
  ↓
unified_update_model_status → EOS/max_dec_len + step_idx + stop_flags + 原有状态更新
  ↓
save_output → 读取 accept_tokens/accept_num（可能被 unified_update 截断）
改动 1：speculate_verify.cu
1.1 accept_and_check_stop() → accept_and_check_eos()
重命名并简化为：仅写 accept_token、检测 EOS 返回 bool（用于 break loop），不修改任何全局状态。

// 旧：写 step_idx, stop_flags, accept_tokens, end_tokens[0] override
// 新：只写 accept_tokens, 返回 is_eos
__device__ inline bool accept_and_check_eos(int bid, int i, int64_t accept_token,
                                            int64_t *accept_tokens,
                                            const int64_t *end_tokens,
                                            int end_length,
                                            int max_draft_tokens) {
  accept_tokens[bid * max_draft_tokens + i] = accept_token;
  return is_in_end(accept_token, end_tokens, end_length);
}
删除：

step_idx[bid]++（line 87）
stop_flags[bid] = true（line 92）
accept_tokens[...] = end_tokens[0] max_dec_len override（line 94-95）
stop_flag_now_int = 1（line 93）
step_idx 和 stop_flags 参数
max_dec_len 参数
1.2 Bulk accept (verify_window) 路径
Lines 223-241 — 同样移除 step_idx/stop_flags 写入：

// 旧
step_idx[bid] += verify_window + 1;
for (; i < ii; i++) {
  accept_tokens[bid * max_draft_tokens + i] = accept_token;
  if (is_in_end(...) || step_idx[bid] >= max_dec_len[bid]) {
    stop_flags[bid] = true;
    stop_flag_now_int = 1;
    accept_tokens[...] = end_tokens[0];
    accept_num_now--;
    step_idx[bid]--;
    break;
  }
}

// 新 — 只写 accept_tokens，EOS 时 break 并调整 accept_num_now
for (; i < ii; i++) {
  auto accept_token = draft_tokens_now[i + 1];
  accept_tokens[bid * max_draft_tokens + i] = accept_token;
  if (is_in_end(accept_token, end_tokens, end_length)) {
    // EOS hit during bulk accept — keep this token but stop
    accept_num_now = i + 1;  // 实际接受数（含 EOS token）
    goto phase2_skip;        // 跳过 Phase 2 采样
  }
}
注意：原来 bulk accept 时 accept_num_now 是先 += verify_window + 1 再在 break 时 -- 修正。新逻辑直接设为 i + 1（当前已写入的 token 数）。

1.3 Phase 2 采样
Lines 268-304 — 移除 step_idx/stop_flags 写入：

// 旧
step_idx[bid]++;
...
if (prefill_one_step_stop) { stop_flags[bid] = true; }
if (is_in_end(...) || step_idx[bid] >= max_dec_len[bid]) {
  stop_flags[bid] = true;
  stop_flag_now_int = 1;
  accept_tokens[...] = end_tokens[0];
}

// 新 — 只写 accept_token，不管 stop
accept_tokens[bid * max_draft_tokens + i] = accept_token;
// EOS 和 prefill_one_step_stop 全部交给 unified_update 处理
1.4 stop_flag_now_int 归约
问题：verify 当前用 stop_flag_now_int 做 CUB BlockReduce 来更新 not_need_stop。但这个归约现在不准确了（verify 不再设置 stop_flags）。

方案：verify 不再做 has_running_seqs 归约。这本来就在 unified_update 里做了（line 151-160）。verify 只需要正确输出 accept_tokens 和 accept_num。

删除：verify kernel 末尾的 BlockReduce 和 has_running_seqs 写入（如果有的话）。

根据代码，verify 没有做 has_running_seqs 归约（那是 unified_update 的职责）。verify 里的 stop_flag_now_int 只是用来控制 Phase 2 是否跳过（line 268 if (!stop_flag_now_int)）。

新方案：用一个 local bool stopped 代替 stop_flag_now_int，仅控制 Phase 2 跳过逻辑。当 EOS 被检测到时 stopped = true，跳过 Phase 2。

1.5 Kernel 签名变更

// 移除：step_idx, max_dec_len, stop_flags（从 read-write 改为 read-only）
// stop_flags 保留为 const 只读（用于跳过已停止的序列）
__global__ void speculate_verify(
    const int64_t *sampled_token_ids,
    int64_t *accept_tokens,
    int *accept_num,
    // int64_t *step_idx,          // 移除
    const bool *stop_flags,        // 从 bool* 改为 const bool*
    const int *seq_lens_encoder,
    // const int *seq_lens_decoder,  // 已经不用了，考虑移除
    const int64_t *draft_tokens,
    // const int *actual_draft_token_nums,  // 已经不用了，考虑移除
    curandState_t *curand_states,
    const float *topp,
    const int *seq_lens_this_time,
    const int64_t *verify_tokens,
    const float *verify_scores,
    // const int64_t *max_dec_len,  // 移除
    const int64_t *end_tokens,
    const bool *is_block_step,
    const int *cu_seqlens_q_output,
    const int *actual_candidate_len,
    const int *reasoning_status,
    ...
    // bool prefill_one_step_stop  // 移到 unified_update
);
1.6 PD_BUILD_STATIC_OP 更新
Outputs 从 {accept_tokens_out, accept_num_out, step_idx_out, stop_flags_out} 改为 {accept_tokens_out, accept_num_out}。

InplaceMap 相应减少。

从 Inputs 移除 step_idx。stop_flags 保留在 Inputs 但不在 Outputs/InplaceMap 中。

考虑是否也移除 max_dec_len、seq_lens_decoder、actual_draft_token_nums（均未在 kernel 中使用）。

1.7 Attrs 更新
移除 prefill_one_step_stop（移到 unified_update）。

改动 2：unified_update_model_status.cu
2.1 新增输入
参数  类型  说明
end_tokens  const int64_t*  EOS token 列表
max_dec_len const int64_t*  每个序列的最大解码长度
num_end_tokens  int (dim param) end_tokens 数量
prefill_one_step_stop   bool (Attr) prefill 完成后立即停止
2.2 原有参数变更
参数  旧   新
step_output_ids (accept_tokens) const int64_t*  int64_t*（读写：可能替换为 end_tokens[0]）
step_output_len (accept_num)    const int*  int*（读写：可能截断）
stop_flags  const bool* bool*（读写：设置 stop）
step_idx    const int64_t*  int64_t*（读写：递增）
注意 step_idx 原来在 unified_update 中是只读的（只在 pre_ids 写入时读取）。现在变成读写。

2.3 新增逻辑（在现有状态更新之前）

// === EOS / max_dec_len 检测 + step_idx 更新 ===
// 必须在 seq_lens_decoder 更新之前执行，因为可能截断 output_len
int output_len = step_output_len[bid];  // 可能被截断

if (!stop_flags[bid] && !(is_paused[bid] || bid >= real_bsz)) {
  bool stopped = false;
  for (int i = 0; i < output_len; i++) {
    step_idx[bid]++;
    int64_t token = step_output_ids[bid * max_step_tokens + i];
    bool is_eos = is_in_end(token, end_tokens, num_end_tokens);

```
if (is_eos || step_idx[bid] >= max_dec_len[bid]) {
  if (!is_eos) {
    // max_dec_len hit, force end token
    step_output_ids[bid * max_step_tokens + i] = end_tokens[0];
  }
  output_len = i + 1;  // truncate
  step_output_len[bid] = output_len;
  stop_flags[bid] = true;
  stopped = true;
  break;
}
```
  }
  if (!stopped && prefill_one_step_stop && seq_lens_encoder[bid] != 0) {
    // prefill_one_step_stop: stop after first prefill token
    stop_flags[bid] = true;
  }
}
关键：is_in_end helper 需要从 verify 中提取为共享函数（或在 unified_update.cu 中重新定义）。

2.4 原有逻辑调整
output_len 现在可能被截断，后续 seq_lens_decoder += output_len 使用截断后的值
stop_flags[bid] 可能在新逻辑中被设为 true，后续 if (stop_flags[bid]) 分支正确处理
step_output_ids 可能被修改（end_tokens[0] 替换），save_output 读取修改后的值
执行顺序：

1. EOS/max_dec_len 检测 + step_idx 更新 + 可能截断 output_len + 设 stop_flags
2. if (stop_flags) → mask_rollback = 0, cleanup seq_lens_decoder
3. else if (decoder) → seq_lens_decoder += output_len, mask_rollback, adaptive adjust
4. else → encoder phase
5. encoder→decoder transition
6. pre_ids 写入
7. step_input_ids 写回
8. seq_lens_this_time 条件重置
9. has_running_seqs 归约
2.5 PD_BUILD_STATIC_OP 更新
新增 Inputs: end_tokens, max_dec_len
新增 Attrs: prefill_one_step_stop: bool
修改 Outputs/InplaceMap: 新增 step_output_ids(accept_tokens), step_output_len(accept_num), stop_flags, step_idx

2.6 InplaceMap 全量

.SetInplaceMap({
    {"seq_lens_encoder", "seq_lens_encoder_out"},
    {"seq_lens_decoder", "seq_lens_decoder_out"},
    {"has_running_seqs", "has_running_seqs_out"},
    {"step_input_ids", "step_input_ids_out"},
    {"adaptive_step_input_len", "adaptive_step_input_len_out"},
    {"step_output_ids", "step_output_ids_out"},       // 新增
    {"step_output_len", "step_output_len_out"},       // 新增
    {"stop_flags", "stop_flags_out"},                 // 新增
    {"seq_lens_this_time", "seq_lens_this_time_out"},
    {"mask_rollback", "mask_rollback_out"},
    {"pre_ids", "pre_ids_out"},
    {"step_idx", "step_idx_out"},                     // 新增
})
改动 3：cpp_extensions.cc
3.1 SpeculateVerify 前向声明
移除 step_idx 参数。stop_flags 保留。移除 max_dec_len（如果 verify 不再需要）。移除 prefill_one_step_stop。

3.2 UnifiedUpdateModelStatus 前向声明
新增 end_tokens、max_dec_len Tensor 参数，新增 prefill_one_step_stop bool 参数。

改动 4：Python 侧
4.1 sampler.py — speculate_verify 调用
移除传递 step_idx、max_dec_len 参数。移除 prefill_one_step_stop 参数。

4.2 pre_and_post_process.py — unified_update_model_status 调用
新增传递 end_tokens、max_dec_len、prefill_one_step_stop 参数。

post_process_specualate 签名新增 prefill_one_step_stop: bool = False。

4.3 gpu_model_runner.py
传递 prefill_one_step_stop 到 post_process。

改动 5：共享 is_in_end helper
当前 is_in_end 定义在 speculate_verify.cu 中（line 62-72）。unified_update_model_status.cu 也需要用。

方案：在 unified_update_model_status.cu 中重新定义一份（简单的 device inline，5 行代码）。或提取到 helper.h。

推荐直接在 unified_update_model_status.cu 顶部重新定义，避免改动 helper.h 影响其他文件编译。

### 文件清单

| 操作 | 文件 | 改动 | 状态 |
|---|---|---|---|
| 修改 | `custom_ops/gpu_ops/speculate_decoding/speculate_verify.cu` | 移除 step_idx/stop_flags 写入、max_dec_len 检查、prefill_one_step_stop；accept_and_check_stop→accept_and_check_eos；Outputs 从 4→2 | ✅ 完成 |
| 修改 | `custom_ops/gpu_ops/speculate_decoding/unified_update_model_status.cu` | 新增 Phase A: EOS/max_dec_len/step_idx 逻辑；新增 end_tokens/max_dec_len/prefill_one_step_stop 参数；Inputs 13→15, Outputs 8→12, InplaceMap 8→12 | ✅ 完成 |
| 修改 | `custom_ops/gpu_ops/cpp_extensions.cc` | SpeculateVerify 移除 4 个参数，UnifiedUpdateModelStatus 新增 end_tokens/max_dec_len/prefill_one_step_stop | ✅ 完成 |
| 修改 | `fastdeploy/model_executor/layers/sample/sampler.py` | verify 调用移除 step_idx/max_dec_len/seq_lens_decoder/actual_draft_token_num/prefill_one_step_stop | ✅ 完成 |
| 修改 | `fastdeploy/model_executor/pre_and_post_process.py` | unified_update 调用新增 end_tokens/max_dec_len/prefill_one_step_stop；post_process wrapper 新增 is_naive_mode/prefill_one_step_stop 透传 | ✅ 完成 |
| 修改 | `fastdeploy/worker/gpu_model_runner.py` | post_process 调用新增 is_naive_mode/prefill_one_step_stop | ✅ 完成 |
验证
编译通过: bash build.sh
MTP spec decode: 接受率与修改前一致（verify 的 EOS 早停 + unified_update 的 EOS 检测 = 等价行为）
max_dec_len 截断: 验证达到 max_dec_len 时 token 被替换为 end_tokens[0]
prefill_one_step_stop: 验证 prefill 后第一步即停止
naive 模式: 验证 is_naive_mode=true 时 EOS 检测正常工作

## 变量命名对照表

| 旧名称 | 新名称 | 双语义说明 |
|---|---|---|
| `accept_tokens` | `step_output_ids` | spec: 验证通过的 tokens / naive: 采样的 token |
| `accept_num` | `step_output_len` | spec: 验证通过数量 / naive: 恒为 1 |
| `draft_tokens` | `step_input_ids` | spec: 候选 tokens / naive: 上一步输出 |
| `actual_draft_token_num` | `adaptive_step_input_len` | spec: 自适应候选长度 / naive: 恒为 1 |
| `not_need_stop` | `has_running_seqs` | 中性：是否还有活跃序列 |
| `is_block_step` | `is_paused` | 中性：该 batch 是否暂停 |
| `seq_lens_this_time` | **不改** | — |
| `mask_rollback` | **不改** | — |
| `pre_ids` | **不改** | — |
| `step_idx` | **不改** | — |
| `stop_flags` | **不改** | 语义已明确 |

新名称仅在新建/修改的文件内部使用。`share_inputs` 字典 key 保持不变，边界处做转换。

------

## 全部修改文件汇总

| 操作 | 文件 | 改动内容 |
|---|---|---|
| **修改** | `custom_ops/gpu_ops/speculate_decoding/unified_update_model_status.cu` | 合并 kernel + `is_naive_mode` 条件化 seq_lens_this_time 写入 + EOS/max_dec_len/step_idx 状态管理 |
| **修改** | `custom_ops/gpu_ops/speculate_decoding/speculate_verify.cu` | 消除模板/getenv/cudaMalloc，新增 3 个 Attr，移除状态管理（纯验证） |
| **修改** | `custom_ops/gpu_ops/cpp_extensions.cc` | 前向声明 + pybind11 注册 unified_update（含 is_naive_mode/prefill_one_step_stop）+ verify 签名精简 |
| **修改** | `custom_ops/setup_ops.py` | 编译注册新 .cu 文件 |
| **修改** | `fastdeploy/config.py` | SpeculativeConfig 新增 verify_strategy/prefill_one_step_stop/accept_policy，method_list 加 naive |
| **修改** | `fastdeploy/model_executor/layers/sample/sampler.py` | SpeculativeSampler 从 config 读取策略并传参 |
| **修改** | `fastdeploy/model_executor/pre_and_post_process.py` | import + post_process_specualate 调用替换 + `is_naive_mode` 参数 |
| **修改** | `fastdeploy/worker/gpu_model_runner.py` | proposer init naive case + None guard + `is_naive_mode` 传递 |

------

## 验证策略

### 单元测试

1. `unified_update_model_status`：构造已知状态 tensor，验证所有 8 个 inplace 输出（含 seq_lens_this_time 重置）
2. `speculate_verify`：分别测试 topk/topp/target_sampling × accept_all/reject_all/normal

### 集成测试

```bash
# MTP Spec Decode 测试
python -m fastdeploy.entrypoints.openai.api_server \
    --model <model_path> \
    --speculative_method mtp \
    --speculative_model_num_speculative_tokens 5

# Naive 模式测试（通过 spec 分支走 naive 推理）
python -m fastdeploy.entrypoints.openai.api_server \
    --model <model_path> \
    --speculative-config '{"method":"naive","num_speculative_tokens":1}'

# 对比 method=None（原有 naive 路径）的输出结果
```

### 性能验证

- 后处理延迟：对比合并前后 kernel 总耗时（预期减少 2 次 launch 开销）
- Verify 延迟：对比重构前后（预期相当或略优，消除 cudaMalloc 开销）
