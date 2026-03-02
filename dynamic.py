Adaptive Drafting with Hybrid Mechanism(ADHM)

背景
* 竞品：
    * 目前 Vllm、Sglang、Trtllm 等开源架构在投机解码方向的方案各有千秋，例如均有 DraftTree 功能，TrtLLM 具有多层 MTP功能，Sglang 具有PD分离功能
    * 但目前所有框架均是静态的：即服务启动后，每个 Step 均是固定数量的 Draft Token Num、Draft Model Step

* 缺陷：
    * 在不同并发下（例如1/32/128），最优策略差异很非常大，需要手动遍历测试，找出最优配置
    * 在实际线上环境，同样有流量的高峰、低估，但目前都是按照一种理想的并发下设置的超参；在并发波动较大时，无法取得最优性能



核心点
框架描述
1. 在任意并发下，结合 接收率 自动计算每个 step 的最优 Token 数量，以及选取最优 Draft Token
2. 结合 MTP(Eagle) / Ngram / Draft Tree 方法，动态生成每个 Step 的 Draft Token，即每个 Query 的 Draft Token可能产出自不同方法（例如1个 MTP 产出的 Token，5个 Ngram 匹配的 Token）

如何“动态”
1. 首先 offline（或服务启动时）Profiler 一份在不同 Token 数量下的推理性能，例如逐 Step 耗时，记为$Profiler_{step}$
2. 模型运行中，结合接受率（$AcceptRatio_{query} / AcceptRatio_{batch} $）以及 $Profiler_{step}$ 计算出 Step 最优 Token 数量配置 $BestTokenNum_{step}$
    1. 首先保证每个 Batch 中的 Query 运行相同次数的 Draft Model Step；
    2. 第一阶段可用 NgramMatch 补足 Token，或一些特殊任务后期也可直接使用 NgramMatch
    3. 第二阶段用 Draft Tree 补足 Token


依赖路径
1. 模型侧：多层 MTP 或自回归下精度较好的 Eagle 系列模型
    1. 目前更倾向自回归 Eagle

2. Profiler 梯度式统计
3. Draft Tree 功能
    1. 框架侧 Tree 的建立管理
    2. Verify 改造
    3. Attention 改造

4. Ngram 从 CPU 版升级到 GPU版

动态设计
变量定义
$T_{step}:一次推理step的时延，单位是s$
$AcceptLen_{avg}: 一个Step中平均每个Query接收的Token数量$
$AcceptLen_{total}: 一个Step所有Query接收的所有Token数量$
$NumToken_{per\_{step}}：一个Step的Token总量$
四种高性能模式
[图片]
1. 自适应动态模式
    1. 一定范围内解码速度、OTPS的相对最优值

2. 吞吐模式
    1. $OPTS = \frac{\text{Total Accepted Tokens in One Step}}{\text{One-step Inference Latency (s)}} = \frac{{AcceptLen_{total}}}{T_{step}}$
    2. $\Rightarrow \max \, \text{OTPS} = max\  \frac{{AcceptLen_{total}}}{T_{step}}$

3. 低时延模式
    1. $\text{Decoding-speed(Token/s)}=\frac{\text{Avg Accept Len Per Query}}{\text{One-step Inference Latency(s)}} = \frac{{AcceptLen_{avg}}}{T_{step}}$
    2. $\Rightarrow \max \, \text{Decoding-speed} = max\  \frac{{AcceptLen_{avg}}}{T_{step}}$

4. 控制解码速度
    1. $Fixed(\frac{\text{Avg Accept Len Per Query}}{\text{One-step Inference Latency(s)}}) = Fixed(\frac{{AcceptLen_{avg}}}{T_{step}})$



从公式可以看出：
1. 所有的指标均可以抽象化为$AcceptLen\ 和\ T_{step}$的函数
2. $T_{step}$只和每个 step 的 token 总量有关，即$NumToken_{per\_{step}}$
    1. 可以在模型启动时进行 Profile，是固定的 mapping

3. $AcceptLen$则是在 $NumToken_{per\_{step}}$限制的范围内，生成质量最高的 DraftToken，最大化$AcceptLen$

不同并发下的推理耗时
[图片]
1. 在 Token 数量较少时，可以看做 Memory-bound 场景；适量增加 DraftToken 数量不会引起推理时延激增，从而提高比值
    1. $AcceptLen$提高，$T_{step}$略微增加，$\frac{{AcceptLen}}{T_{step}}$几乎一定是提高的，在各个场景下均有提升

2. 在中间的区间时，DraftToken 与推理时延都会线性增长，基于不同接受率、不同模型的架构有不同的增长曲线，需找到那个最优解
    1. $AcceptLen$与$T_{step}$同时增加，需根据不同场景找到最优比值

3. 在 Token 数量过大时，DraftToken 的增加完全与推理时延的增加成正比，则会导致负向收益
    1. $DraftLen$与$T_{step}$几乎成线性增加，而$AcceptLen$< $DraftLen$，因此是负向收益


[图片]
