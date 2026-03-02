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
| **新增** | `custom_ops/gpu_ops/speculate_decoding/unified_update_model_status.cu` | 合并 kernel + `is_naive_mode` 条件化 seq_lens_this_time 写入 |
| **修改** | `custom_ops/gpu_ops/speculate_decoding/speculate_verify.cu` | 消除模板/getenv/cudaMalloc，新增 3 个 Attr |
| **修改** | `custom_ops/gpu_ops/cpp_extensions.cc` | 前向声明 + pybind11 注册 unified_update（含 is_naive_mode）+ verify 签名更新 |
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
