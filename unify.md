一、需求背景与痛点
双重实现，人力翻倍
1. 人力/维护成本双倍：FD里每个功能均是两套实现(Spec && Naive)，相似功能无法复用
2. 典型功能：cudaGraph / prefixCahe / logprob / zmq 通讯 / multi_stops / token_processor / .....
3. 痛苦案例：功能开发在 Naive 分支，上线是 Spec 分支；快上线时发现 Spec 分支功能未实现，紧急人力支持

语义割裂，认知成本高
1. Token语义映射映射不一致：Spec 与 Naive 分支对 token_ids 的 padding / offset 管理完全不同
2. 抽象过重，Spec 上手难：引入对 output_padding_offset 等抽象管理，理解、上手难度高
3. 行为语义不一致，本质是两套系统：Naive后处理是以 Batch 维度为核心，Spec后处理以 Token 维度为核心

新人上手困难
1. 非核心开发者无法在无"口口相传"的情况下独立开发
2. DraftToken / AcceptToken 概念理解困难，输入/输出易绕晕
3. 同名功能语义不同，行为不可预期

二、目标与愿景
2.1 架构统一
构建 一套统一的推理架构与算子体系，Spec 与 Naive 不再并列存在。
目标收益：
1. 减少 30% ~ 50% 的算子数量
2. 显著降低核心代码体量
3. 减少长期维护与人力成本

核心原则：
1. 推理侧只保留一套算子

   * 不再出现 if speculative_decoding

2. 服务侧对投机解码无感知

   * 所有 token_ids 统一通过 for-loop 处理
   * Spec 细节完全收敛在推理内部

2.2 3A 架构愿景
愿景：AnyOne 可以在 AnyContext 理解 AnyCode
新架构尽努力满足 3A 原则：
维度
目标
AnyOne
非核心作者、无历史背景的人也可参与开发
AnyContext
无需口头传承或隐性知识即可理解代码
AnyCode
代码本身即可表达设计意图
具体体现为：
* 变量命名、使用方式符合直觉
* 模块职责清晰，算子边界明确
* 降低模块间耦合，避免“牵一发而动全身”
* 按照开发范式，无需关注是否投机解码场景，即可正确开发

2.3 Overlap 原生支持
目前已有一版非投机解码的 overlap 实现 （by 孙欣 康康）；
但由于投机解码架构差异较大，无法直接复用，需在重构、解决算子间耦合后，再重新设计
架构需满足：
1. 支持 Spec 分支的 0 维 Tensor 的 model_forward
2. 变量状态可支撑延迟 save_output
3. not_need_stop / accept_tokens 异步拷贝，延迟一个 step
    1. 最优删除 not_need_stop

2.4 动态投机解码系统
TODO:
* 留好异步接口，待统一框架完成后再进行开发

投机解码动态架构（ADHM）

三、技术方案
3.1 核心设计理念
Naive 是 Spec 的特例：当 num_speculative_tokens = 0 时，系统自然退化为普通解码模式。
┌─────────────────────────────────────────────────────────────┐
│                    统一推理架构 (Unified)                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │  前处理      │ -> │  模型推理    │ -> │  后处理      │      │
│  │  (Unified)  │    │  (Unified)  │    │  (Unified)  │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│         │                                                   │
│         v                                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Spec 模式: num_speculative_tokens > 0               │  │
│  │  Naive 模式: num_speculative_tokens = 0 (自然退化)    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

3.2 变量体系重构
3.2.1 设计思路
现有背景
1. Naive 模式：使用 input_ids 传入 prompt/decode，输出 sampled_token_ids（单 token）；输入写入 input_ids[0]
2. Spec 模式：使用 draft_tokens 作为候选，输出 accept_tokens + accept_num（多 token）;输入写入 draft_tokens
3. 两套变量体系导致必须有 is_spec_mode 判断

核心思想：
1. 不区分 Naive/Spec，统一用「每步输入 N 个 token，输出 M 个 token」的范式
2. 职责分离：Prompt、Decode 输入、已生成历史、本步输出 各自独立存储

3.2.2 新变量体系
# ============ 新变量体系（职责分离 + 复用现有变量）============

# 关键维度定义
max_tokens_per_step = num_speculative_tokens + 1  # 每步最大可能 token 数
                                                   # - Naive: 1
                                                   # - Spec: draft_num + 1（包含 bonus）

# 1. Prompt 存储（Prefill 输入，只读）
prompt_ids: [batch, max_prompt_len]      # 原始 prompt

# 2. Decode 输入（维度小）
step_input_ids: [batch, max_tokens_per_step]   # Decode 阶段的输入
                                          # - Naive: 只用 [:, 0]
                                          # - Spec: 存储 draft tokens
删除 input_ids
seq_lens_this_time: [batch]               # 本次输入长度
                                          # - Prefill: prompt 长度
                                          # - Decode Naive: 1
                                          # - Decode Spec: draft_num + 1

# 3. 已生成历史（仅存储 decoder 产出的 ID，不含 prompt）
history_ids: [batch, max_new_tokens]      # 已确认输出的 tokens，替换pre_ids
                                          # - 不包含 prompt
                                          # - 仅存储 decoder 产出的 ID

# 4. 本步输出
step_output_ids: [batch, max_tokens_per_step]  # 本步的输出 tokens
step_output_len: [batch]                  # 本步输出长度
                                          # - Naive: 恒为 1
                                          # - Spec: 验证通过的数量
* step_input_ids / step_output_ids 分离，原生支持 overlap 方案

关键变量命名说明：
字段
形状
说明
新增/复用
prompt_ids
[batch, max_prompt_len]
Prefill 输入，只读
现有
step_input_ids
[batch, max_tokens_per_step]
Decode 输入
新增
seq_lens_this_time
[batch]
本次输入长度
复用
history_ids
[batch, max_new_tokens]
已生成历史（不含 prompt）
新增
step_output_ids
[batch, max_tokens_per_step]
本步输出
新增
step_output_len
[batch]
本步输出长度
新增
prompt_len
[batch]
Prompt 长度
现有
3.2.3 新旧字段
变量对照表：
旧变量 (Naive)
旧变量 (Spec)
新统一变量
说明
input_ids +
next_tokens
input_ids + draft_tokens
prompt_ids + step_input_ids
职责分离
pre_ids
pre_ids
history_ids（只存已生成）
语义更清晰
seq_lens_this_time
seq_lens_this_time
seq_lens_this_time（复用）
本步输入长度
sampled_token_ids
accept_tokens
step_output_ids
本步输出
(隐式=1)
accept_num
step_output_len
本步输出长度
3.2.4 设计总结
1. 消除 input_ids / draft_tokens / accept_tokens 命名冲突

   * 统一为 step_input_ids 和 step_output_ids

2. 消除 is_spec_mode 判断

   * 所有算子都接收 step_input_len 和 step_output_len
   * 算子内部根据长度自动处理，无需外部判断

3. Proposer 是可选组件

   * 有 Proposer：准备多 token 输入（step_input_len > 1）
   * 无 Proposer：默认单 token 输入（step_input_len = 1）

4. 原生支持 overlap
    1. step_output_ids 与 step_input_ids 独立，可并行操作

3.3 数据流设计
3.3.1 统一执行流程
┌───────────────────────────────────────────────────────────────────────────┐
│                     职责分离的数据流                                        │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   数据存储布局：                                                            │
│   ┌─────────────────────────────────────────────────────────────────────┐│
│   │  prompt_ids:      [batch, max_prompt_len]         # Prefill 输入       ││
│   │  step_input_ids:  [batch, max_tokens_per_step]    # Decode 输入         ││
│   │  history_ids:     [batch, max_decode_len]         # 已生成（不含prompt） ││
│   │  step_output_ids: [batch, max_tokens_per_step]           # 本步输出            ││
│   └─────────────────────────────────────────────────────────────────────┘│
│                                                                           │
│   Step 1: 前处理 - 构造 ids_remove_padding                                │
│   ┌─────────────────────────────────────────────────────────────────────┐│
│   │  ids_remove_padding = []                                            ││
│   │  for each batch bi:                                                 ││
│   │      if seq_lens_encoder[bi] > 0:  # Prefill                       ││
│   │          ids_remove_padding += prompt_ids[bi, :seq_lens_encoder]   ││
│   │      else:  # Decode                                               ││
│   │          ids_remove_padding += step_input_ids[bi, :step_input_len] ││
│   │                                                                     ││
│   │  # 和老方案一样，只是数据来源不同                                     ││
│   └─────────────────────────────────────────────────────────────────────┘│
│                              │                                            │
│                              v                                            │
│   Step 2: Model Forward                                                   │
│   ┌─────────────────────────────────────────────────────────────────────┐│
│   │  hidden_states = model(ids_remove_padding, cu_seqlens_q, ...)       ││
│   │  logits = compute_logits(hidden_states)                             ││
│   └─────────────────────────────────────────────────────────────────────┘│
│                              │                                            │
│                              v                                            │
│   Step 3: Sample & Verify                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐│
│   │  unified_sample_verify(logits, ...) -> step_output_ids, step_output_len│
│   │                                                                     ││
│   │  # 输出写入 step_output_ids（独立buffer，支持 Overlap）              ││
│   └─────────────────────────────────────────────────────────────────────┘│
│                              │                                            │
│                              v                                            │
│   Step 4: 更新 history_ids / seq_lens_decoder / stop_flags ...（追加本步输出)│
│   ┌─────────────────────────────────────────────────────────────────────┐│
│   │  for each batch bi:                                                 ││
│   │      pos = seq_lens_decoder[bi]                                     ││
│   │      for j in range(step_output_len[bi]):                           ││
│   │          history_ids[bi, pos + j] = step_output_ids[bi, j]          ││
│   │      seq_lens_decoder[bi] += step_output_len[bi]                    ││
│   └─────────────────────────────────────────────────────────────────────┘│
│                              │                                            │
│                              v                                            │
│   Step 5: Save Output（可 Overlap）                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐│
│   │  # step_output_ids 独立，可异步保存                                  ││
│   │  unified_save_output(step_output_ids, step_output_len)              ││
│   └─────────────────────────────────────────────────────────────────────┘│
│                              │                                            │
│                              v                                            │
│   Step 6: 准备下一步 Decode 输入                                           │
│   ┌─────────────────────────────────────────────────────────────────────┐│
│   │  for each batch bi:                                                 ││
│   │      if proposer exists:                                            ││
│   │          proposer.fill(step_input_ids[bi], step_input_len[bi])     ││
│   │      else:                                                          ││
│   │          # Naive: 用本步输出的最后一个 token                           ││
│   │          step_input_ids[bi, 0] = step_output_ids[bi, 0]             ││
│   │          step_input_len[bi] = 1                                     ││
│   └─────────────────────────────────────────────────────────────────────┘│
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
3.3.2 混合 Prefill/Decode 场景示例
假设 batch_size = 4:
* batch[0]: Prefill (新请求)
* batch[1]: Decode Naive (老请求)
* batch[2]: Decode Spec (老请求)
* batch[3]: Prefill (新请求)

seq_lens_encoder: [128, 0, 0, 256]   # >0 表示 Prefill

ids_remove_padding 来源：
* batch[0]: prompt_ids[0, :128]          # 从 prompt_ids 读取
* batch[1]: step_input_ids[1, :1]        # 从 step_input_ids 读取
* batch[2]: step_input_ids[2, :n+1]      # 从 step_input_ids 读取
* batch[3]: prompt_ids[3, :256]          # 从 prompt_ids 读取

关键点：Prefill 和 Decode 从不同的 buffer 读取，不区分是否投机解码

3.3 [Spec分支]Token 表达范式优化
3.3.1 当前问题
原有方式需要理解复杂的 padding 机制：
# 原来的方式（需要理解 padding 机制）
bi = (token_idx + output_padding_offset[token_idx]) / max_seq_len
query_start_token_idx = bi * max_seq_len - output_cum_offsets[bi]
relative_pos_in_batch = token_idx - query_start_token_idx
3.3.2 优化方案
采用更直观的 cu_seqlens_q 体系：
# 优化后的方式（直观易懂）
bi = batch_id_per_output_token[token_idx]
query_start_token_idx = output_cu_seqlens_q[bi]
relative_pos_in_batch = token_idx - query_start_token_idx
3.3.3 新变量语义
变量名
语义
形状
说明
batch_id_per_output_token
每个输出 token 所属的 batch id
[total_output_tokens]
直接索引，无需计算
output_cu_seqlens_q
输出序列的累积长度
[batch_size + 1]
标准 cu_seqlens 格式
seq_lens_output
每个 batch 的输出 token 数
[batch_size]
Naive=1, Spec=accept_num
3.3.4 关键映射关系
# 统一的 Token 索引映射
for token_idx in range(total_output_tokens):
    batch_id = batch_id_per_output_token[token_idx]
    seq_start = output_cu_seqlens_q[batch_id]
    relative_pos = token_idx - seq_start  # 在该 batch 内的相对位置
3.4 [Spec分支]前处理优化
3.4.1  当前状态
* 算子数量：3 个 custom op + 2 个散 op
* 耗时：~200us

3.4.2 优化目标
* 算子数量：1 个 custom op + 少量散 op
* 耗时：~20us（10x 提升）

3.4.3 算子合并方案
待合并算子清单：
原算子
功能
合并后
get_padding_offset
计算 padding offset
合并
speculate_get_padding_offset
Spec 版本
合并
speculate_get_seq_lens_output
Spec 版本
合并
散 op  * n
辅助计算
合并
新统一算子：unified_prepare_inputs
// 输入
* prompt_ids
* input_token_ids: [batch_size * (max_draft_tokens+1)]
* seq_lens_encoder: [batch_size]
* seq_lens_decoder: [batch_size]
* seq_lens_this_time: [batch_size]  // Naive=1, Spec=num_draft+1
// 输出
* ids_remove_padding: [total_tokens]
* cu_seqlens_q: [batch_size + 1]
* cu_seqlens_k: [batch_size + 1]
* cu_seqlens_q_output: [total_output_tokens + 1]
* batch_id_per_token: [total_tokens]
* batch_id_per_output_token: [total_output_tokens]
3.4.4 init_forward_meta 优化
* 目前多步MTP会多次触发此功能，耗时多次
* 优化为一次，后面的 step 复用前面的 step

3.5 [Spec分支]后处理重构
3.5.1 当前问题
* accept_num / step_idx 会在多处修改/回退
* 后处理分为多个耦合的算子，update / clear_accept / set_value 三个算子可合并

3.5.2 优化方案
算子合并：
unified_update_model_status = speculate_update + clear_accept + set_value
* 更新 seq_lens* / stop_flags / pre_ids / next_step_input_tokens / accept*

3.6 [Spec分支]Verify 重构
3.5.1 当前问题
* 分支复杂，trick offset 计算多
* 代码架构不规范
* 性能有优化空间

3.5.2 优化方案
* 整合分支，规范代码架构
* 使用严谨的逻辑推理代替 trick offset 计算
* 性能优化

四、MileStone规划
[图片]
阶段一：Spec 分支重构（预计 15 天）
子任务
预计工时
负责人
状态
前处理优化
3 天
慧聪
Doing
Token 表达范式优化
2 天
康康
Done
后处理/MTP 联动优化
3 天
子畅
待开始
Verify 重构
2 天
子畅
待开始
logprob 性能优化
3 天
彦澎
待开始
ZMQ 完善
1天
彦澎
待开始
padding_samper_param + seed 性能优化
2天
彦澎
待开始
非 Spec 分支回归摸底
5天
慧聪
Done
阶段性目标：
* 结构清晰易懂，确保大家可正常在此分支开发
* 对变量 / 功能 / 算子语义做优化，合理的支持 Naive 场景
* 使用 NaiveProposer 性能基本对齐非 Spec 分支

阶段二：统一分支（预计 20 天）
子任务
预计工时
负责人
Naive 模式回归
5 天

Spec 功能回归
5 天

model_forward 统一
3 天

Config 重构
1 天

InferEngine 统一
3 天

Serving 统一
2 天

阶段性目标：
* 完全删除 FD 中(不包含 EB5)

功能回归清单
* [ ] cudaGraph
* [ ] prefixCache
* [ ] logprob
* [ ] zmq 通讯
* [ ] multi_stops
* [ ] token_processor
* [ ] chunked_prefill
* [ ] MTP 投机解码
* [ ] suffix 投机解码
