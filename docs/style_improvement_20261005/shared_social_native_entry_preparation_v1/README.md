# 原生Social AMP入口：迁移与实际CPU预检

新统一入口：[evaluate_shared_social_native.py](../../../scripts/evaluate_shared_social_native.py)。
详细范围和源码结构见[原生接入文档](../../SHARED_SOCIAL_NATIVE_EVALUATION_ZH.md)。

## 实际检查

| 检查 | 外部工具实际退出 | 证据 |
| --- | --- | --- |
| 3人DP源码检查 | `8da894`，0 | DP_N3_source_checks_v2.log |
| 15人SFM源码检查 | `9df061`，0 | SFM_N15_source_checks_v2.log |
| 3人DP严格S1资源预检 | `f94322`，0 | DP_N3_preflight_v2.json |
| 15人SFM严格S1资源预检 | `487e24`，0 | SFM_N15_preflight_v2.json |
| 最终30项数值/来源/别名检查 | `55b038`，0 | tests_final_v2.log |
| Ruff | `87c743`，0 | ruff_final.log |

源码检查没有导入Torch或启动SDK。预检明确选择同一个S1 Pop-Art checkpoint与formalDP40k配对资源，
验证已有实际训练退出、完整SHA、native scene/schema、forwarded参数和reader选择；没有加载网络或启动SDK。
输入场景为新固定资源队列的Warehouse N3及N15，配置与其原生资源均显式绑定。

原生观测使用独立模块合成已有TorchScript来源检查；生命周期源码从现有N4版本作11项可逆替换。
单articulation、24 SDK sensors和当前SDK释放方法保持原实现，peer/floor数量成为N+1。
来源/别名夹具检查通过，不表示原生PhysX views实际已释放。

## 未验证部分

GPU仍由S3长预算训练独占。新入口尚未实际运行原生30步/200秒，未证明当前Pop-Art模型在原生场景可用。
完整World或producer尚未构造成功时的异常清理也未认证。接下来必须串行执行真实SDK smoke，
记录实际退出、native接触/全身原始数组和释放结果，再补齐原生初态配对与独立指标。
当前seed固定7，八场景名单与作者计数定义仍未恢复；没有达到/超越论文效果的声明。

本目录同时保留v1源码检查及资源预检。v2仅补充预检中的当前迁移/Pop-Art准入元数据，控制和指标数学不变。
最终只清除生命周期源码的一处行尾空白，整个AST一致（实际工具`e93800`退出0）；
30项测试时的完整原文件保留为`native_navigation_lifecycle_tested_v2.py`。
当前源码的两臂最终检查`7c6211`／`503b19`与Ruff `232e6c`均实际退出0；v3日志保留。
