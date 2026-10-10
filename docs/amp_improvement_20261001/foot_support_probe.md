# 独立 native 脚部支撑 QA probe

日期：2026-10-02。脚本 `scripts/probe_paper_foot_support.py`，测试 `tests/test_paper_foot_support_probe.py`。本agent只执行CPU tests / SDK-free实际资源preflight；**尚未启动Kit/GPU，尚未验证原生地面接触**。AMPWorld、PaperPowerSimulator、runner、跟踪基准和训练源码没有修改。

后续root实际v1在legacy SimulationManager lookup失败、无阶段采集；失败工件保留。最新v2已修正到当前Lab manager的公开view，CPU20tests PASS，source冻结，使用新输出目录。请按 [v2 lookup修正与最新命令](foot_support_probe_v2_lookup_fix.md) 执行；下方v1命令保留作首次版本记录，不应覆盖已存在的v1输出。

## 运行方式

先检查实际manifest、container/LFS状态、唯一pose-anchor ID、30Hz/frame范围、MJCF脚部名字和finite几何；该命令不导入Torch/SDK，不创建capture输出目录。

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
../IsaacLab-Protomotions/.venv/bin/python scripts/probe_paper_foot_support.py \
  --output output/diffusion_amp/foot_support_native_qa_v1 --preflight-only
```

root在其独占GPU窗口执行真正的有限QA（不要与正在运行的训练/跟踪混用GPU）：

```bash
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
../IsaacLab-Protomotions/.venv/bin/python scripts/probe_paper_foot_support.py \
  --deterministic-evaluation \
  --output output/diffusion_amp/foot_support_native_qa_v1
```

输出目录必须是新的或空目录；历史结果不可覆盖。成功exit0，失败exit1，traceback与result在关闭Kit前写出。没有影片输出。

## 场景与控制

实际 `AMPWorld(contact_reporting=True,num_envs=1,shared_world=True)` 在初始化前由现有scene_description机制创建一个高空静态墙cuboid，中心 `(12,6,3.4)`、大小 `(.2,4,3)`；起始站立root XY `(6,6)`，地面为真实terrain。不是事后USD变更或视觉假墙。

默认neutral zero-DOF固定PD，root高度1m，root XYZW identity与所有初始线/角/DOF速度0。这里0角速度在child_joint坐标也为0。真实rig限位和现有PACER decoder（含冻结hand/toe）仍生效；可显式选 `--pose-anchor walking_frame`，记录其source frame与decoder裁剪/冻结后的实际固定target。这不是actor、AMP训练或姿态跟踪策略。

每个阶段只在开始时native reset一次；阶段内只有同一个固定normalized PD action，没有posewrite、teleport、额外settle physicsstep或kinematic目标替代。

| 阶段 | 有限控制步数 | 初态与严格证据 |
|---|---|---|
| standing | 默认60×30Hz | neutral PD沉降后末20controls左右脚均有接触的control比例≥.9、全阶段无fall；原始4子步力保留 |
| airborne | 固定4 | root抬高3m、初速0；所有floor force分量abs≤1e-5N，foot body origins z>1m |
| airborne_wall | 固定4 | 同样root抬高3m，将实际初态toe最大X中心放进高空墙表面约2cm，仅initial reset；真实脚-wall filtered force magnitude>1N，同时全部floor force≤1e-5N、foot origins z>1m |

standing的contact按每foot `>1N`，side为ankle/toe任一，stable判定当前采用control内any子步contact；它不是所有子步都支撑的保证。raw substeps、double/no-support fractions和any/all都存储，可独立审查。airborne墙穿入是专用contact-separation QA刺激，不是正常人群行为配置。

墙与地面使用**两个独立exact-path RigidContactView**，只对4个脚rigid bodies过滤；禁止把wall列加入floor view。创建前解析enabled CollisionAPI地面后代与wall collider，检查sensor/filter metadata并按实际路径重排。接口不存在、路径歧义或shape变化会失败，不能输出全零“无接触”。

## 无额外步的采样接口

`ProbeSubstepAdapter` 注入这个独立world的 `PaperPowerSimulator.pedestrian_contacts`：

- `begin_control_step()` 读取实际episode IDs和已存在 `_sim.get_physics_step_count()`。
- 原来的 `_sim.step(render=False)` + `_scene.update(1/120)` 后调用 `sample(dt)`；每control必须恰4个连续counter值，每次读取前后counter不得增加。
- 当前固定Lab的public counter是monotonic **submitted physics-step counter**；不是未知的原生PhysX engine time。没有synthetic counter fallback，也不宣称PhysX/Warp bitwise保证。
- body channels用固定public getter；`body_link_lin_vel_w` 是link actor-origin速度。native body_names/root-view names必须一致，再按完整名字双射转换common顺序。保存 `.detach().clone()`，防止Warp/native缓冲被下步覆盖。
- 同步采集live worldCOM、localCOM、native body masses；检查 `link_pos + R(link_quat_xyzw)*localCOM` 与真实worldCOM误差≤1e-4m，再计算真实质量加权COM。不能用body origin proxy替代COM。
- 当前hook没有其他monitor。脚本拒绝覆盖已经存在的map/peer monitor；不会改变通用AMPWorld的默认行为。

## 输出与失败保存

正常结果包含 `standing.npz`、`airborne.npz`、`airborne_wall.npz` 和 `result.json`。每control的raw floor力为 `[4,1,4,G,3]`，身体位置/朝向/原点速度/角速度/COM/localCOM/masses为4个子步的common-order快照。原生未重排floor矩阵、独立wall原始矩阵、按foot重排的wall矩阵、actual Labcounter before/after、dt、episode、root/termination flags、force约简与body-origin切向速度proxy一起保存。

报告包括live USD、native/common body名字、exact脚/地面/墙路径、实际物理配置、manifest/package声明与source SHA、sourceclip ID、PD target、reset前后counter、阈值、每阶段criterion和capture SHA。Top-level `native_physical_probe_passed=True` 只在三个阶段全部实际通过后写入。

任何SDK/data/partial错误，先保存已完成controls prefix，再逐个保存pending substeps已取得字段。不存在的字段保持缺失，不填0；非有限的故障原始值可以留在标记为failed的partial capture作诊断，不被发布为有效支撑统计。

## 当前CPU证据与仍待验证

15 focused CPU tests PASS，0.85s；Ruff两新文件PASS。测试有4步hook计数器、native/common质量重排、buffer clone、COM90°变换、地面/墙分离、跨episode/reset缺步拒绝、非有限原始物理/force partial保存、阶段criterion独立输入、CWD/CLI import无Torch/SDK、LFS/overwrite/参数门禁。mock clocks和mock接触值不是实际物理证据。

实际默认manifest/MJCF preflight已经PASS：[JSON](foot_support_probe_actual_preflight.json)。测试结果：[JUnit](foot_support_probe_cpu_tests.xml)，源与状态：[JSON](foot_support_probe.json)。真实neutral fixed-PD站立稳定性、原生metadata布局、floor/wall接触载荷和COM读取仍由root运行验证；在取得实际result前不能称native QA完成。该probe即使通过也只证明这些数据通道的有限QA，不证明NavIsaacLab论文步态成功率。
