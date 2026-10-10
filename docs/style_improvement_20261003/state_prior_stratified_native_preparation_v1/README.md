# q98a：Warehouse4／15 最小准备与ABI调查

使用已独立审计／root封存的q终态`98a49058…`，2000updates／16.384M transitions；固定旧DP`63e7cad0…`、alpha0、guide20、单候选、0.5秒重规划、200秒及原Warehouse4／15全部场景和routes参数。[完整部署数组](commands.json)逐token对比原19f实际preflight，仅AMP路径与output变化；q terminal/source准入记录另绑定新身份。

## 生产loader结论

`EvaluationActor.from_checkpoint → ReconstructionAMPInference.from_checkpoint → validate_bundle_identity`的参数是checkpoint，不接收外部auditor JSON；不存在“q auditor schema不能被production185接受”的生产缺陷。外部owner／reader的旧19f terminal格式、model SHA、183source身份及fixture质量门槛需要薄metadata适配；不能把q审计JSON改名伪装成19f格式。

[ABI证据](ABI_source_review.json)记录6个支持模块在production185与实际q935中逐字节一致：EvaluationActor、checkpoint contract、settings、state-prior初始化、trainer、terrain statistics。`amp_inference.py`并非整文件相同；[精确delta](inference_source_delta.patch)是原完整model／RMS严格载入后裁掉未执行网络，使用同一task_encoder＋actor计算原Gaussian mean，省去distribution构造。q实际auditor inference probe使用旧完整network；本准备未再加载production185的q网络，不能把source支持判断说成新的实际tensor／GPU验证。

评估分支不配置expert sampler，不执行训练resume的sampling/reward比较。因此可以保留19f原native public profile：其中sampling为null不替换已载入q actor的saved configuration／完整训练provenance。q私有RNG和训练采样计数不在inference恢复。

## 已实际执行的CPU检查

- [terminal＋fixture metadata](actual_CPU_terminal_fixture_metadata.json)实际PASS／`.023s`，没有Torch／NumPy／SDK／模型载入。terminal直接委托原已独立审阅q executor：935source、2019D、AMP RMS24,776,000、public prior、protected suffix、q registry／private RNG、8,192,000 windows／2000batches全部按其原真实契约核验。
- 同一检查绑定已真实闭合raw owner `6ed65649…`、raw report `5af611af…`及sidecar `331aafd9…`：完整6000／200s／4人、0跌倒／0static／0peer，raw来源与AMP98a一致。真实CPU owner是root的tools session／completion ABI；没有杜撰Popen child字段。未重新跑raw／90／45数学。
- [native4／15真实SDK-free preflight](actual_native_preflights_closed.json)均closed0（`.504s`／`.514s`）。parsed仅resume／output改变，native_scene_request整体等于19f模板；actor-only请求仅checkpoint身份改变。读权重仅opaque SHA，不启动SDK，不反序列化模型，不建立native输出。

## 待完成的薄运行接线

本记录给出direct deployment argv，**还没有新的native进程owner／raw reader source adapter登记**。下一步复用原owner的真实wait／cleanup与原native raw／90／45数学，绑定current185、AMP98a、新terminal、output及实际闭合记录；4人实际raw成功后再允许15人。旧19f owner的21arrivals／inclusive90启动门槛不沿用，native启动只以真实完整fixture零跌倒／零static／零peer及来源闭合准入，MMD不作为launch filter。

所有native失败、接触、reset和idle保留；没有读取未来native capture或sealed34，没有GPU/native物理效果结论，不自动推广默认，不修改production185／q935／旧冻结。
