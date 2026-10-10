# 近时域权重 ETA 0.25：固定三组导航风格比较

时间：2026-10-02T06:32:50.518938+00:00。仅CPU，4 threads，无GPU/Kit。业务、资源、权重、历史报告和evaluator均未修改。

## 结论

**当前没有ETA 0.25整体优于guidance20的证据。** ETA整体MMD²略差、速度分布距离几乎不变，且出现2次跌倒；部分0.1–1m/s速度bin有改善。该结论仅基于本次固定单seed200秒实测，不能据此判断所有近时域权重。

guidance40运动更快、路线顶点更多、speed W更小，但MMD²和稳定性未优于guidance20。路线收益或局部bin改善不足以抹掉真实跌倒。

## 固定结果

| 配置 | 跌倒 | 顶点总数 | inclusive MMD² | upright MMD² | speed W m/s | 物理位移速度均值 m/s |
|---|---:|---:|---:|---:|---:|---:|
| guidance20 | 0 | 21 | 0.072381 | 0.072421 | 0.353368 | 0.560623 |
| guidance40 | 1 | 29 | 0.086353 | 0.085461 | 0.217233 | 0.697314 |
| eta025 | 2 | 20 | 0.077826 | 0.075019 | 0.354131 | 0.559891 |

三个来源各4人、6000控制步/200秒，全部fall frames保留。只按冻结实现排除跨reset的导数interval/triplet；同episode interval数量分别23996/23995/23994，acceleration triplet数量23992/23990/23988。upright子集是次要诊断，不能替换inclusive接受条件。

原guidance20/guidance40两个run字典与历史frozen_style逐值完全相同。固定reference SHA256：`c274348fc2d58862fa47e7fdb00b136d71e806fc7ac9ef182e87c0f0249f0564`。controller和Diffusion模型完整文件记录在三run之间完全一致。

## 物理速度区间的MMD²

| 区间 m/s | guide20 | guide40 | ETA .25 |
|---|---:|---:|---:|
| [0.0, 0.1) | 0.162037 | 0.119738 | 0.170187 |
| [0.1, 0.5) | 0.045008 | 0.119049 | 0.036592 |
| [0.5, 1.0) | 0.026610 | 0.019699 | 0.018942 |
| [1.0, 1.5) | 0.051330 | 0.051907 | 0.053774 |
| [1.5, 2.0) | 0.076919 | 0.121370 | 0.109633 |
| [2.0, 3.0) | unsupported | unsupported | unsupported |
| [3.0, ∞) | unsupported | unsupported | unsupported |

2m/s以上参考支持不足，未强评分。ETA对[.1,.5)与[.5,1)较好，但最低速及1m/s以上较差；总MMD²是有支持bin的等权平均，不是所有帧简单加权。

## Body-origin脚部代理

| 配置 | 近地toe速度均值 m/s | >.1m/s fraction |
|---|---:|---:|
| guidance20 | 0.397334 | 0.207178 |
| guidance40 | 0.545471 | 0.289661 |
| eta025 | 0.406129 | 0.213275 |

这是高度/垂速条件下的toe body-origin代理，不是真实脚掌支撑点滑移或接触。native sole/COM/contact/clock与near-stall诊断由独立ground-support审查负责，此报告不重新计算它们。

## 配置与证据范围

ETA相对guide20的runtime参数差异为新增显式 `near_horizon_waypoint_mass=.25` 与输出目录；guide40相对guide20为 `guidance_weight=40` 与输出目录。历史guide20没有near-horizon CLI字段；旧行为未施加该额外近时域质量。新报告额外带有 `expert_sampling=None` 字段，这是元数据字段存在性差异，未启用专家采样。全部逐字段差异与实际计数保存在JSON；没有把相同seed称为相同physical trajectory。

90D指标使用局部位置+速度、speed-bin Gaussian biased MMD²、sigma1。作者精确实现/阈值未知。validation已在开发中反复使用，不是盲测；窗口与邻近帧相关，没有IID置信区间。当前带平面障碍布局和显式部署适配，不等同于作者八场景实验。

## 文件

- [冻结evaluator原始结果](evaluation.json)
- [机器比较与实际参数/model身份](comparison.json)
- [reference原字节副本](heldout_reference.json)
- [CPU evaluator log](../near_horizon_eta025_style_v1.log)
- [summary log](comparison.log)
- [summary Ruff](ruff.log)
- [前guide20/40历史快照](../navigation_guidance40_comparison_v1/frozen_style/evaluation.json)
