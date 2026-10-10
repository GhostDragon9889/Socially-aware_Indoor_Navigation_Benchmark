# ProtoMotions / SMPL 公开资源核对与补齐

日期：2026-10-01。核对本地当前工作树、原 NavIsaacLab 参考及公开一手资料。
没有安装依赖、启动 Kit、读取账户凭据或替换现有控制权重。后续父任务明确授权
下载 SMPL 小资源及独立 PACER 参考；PACER 结果单独记录。

## 1. 本次已补齐的真实 SMPL 外观资源

官方来源为 [Meshcapade/SMPL_texture_samples](https://github.com/Meshcapade/SMPL_texture_samples/tree/c87d871c7498fdc98f3cabb7341b5c85c38debaa)，
固定 revision `c87d871c7498fdc98f3cabb7341b5c85c38debaa`。源仓库明确使用
[SMPL-Body CC BY 4.0](https://smpl.is.tue.mpg.de/bodylicense.html)。
原始许可证和 MPI-IS / Meshcapade 归属已保留；这不改变已有参数化 SMPL 模型的许可证。

| 本地资源 | 实际状态 |
|---|---|
| `data/smpl/smpl_uv.npz` | 新增 266,068 字节；`uvs=(28944,2)`，`uv_faces=(13776,3)` |
| `data/smpl/textures/f_01_alb.002.png` | 新增官方原始 albedo，17,457,804 字节，4096×4096 RGB |
| `data/smpl/textures/m_01_alb.002.png` | 新增官方原始 albedo，18,954,277 字节，4096×4096 RGB |
| `data/smpl/meshcapade_source/` | 原始男女 OBJ、MTL 与许可证，均保留源字节/Git blob hash |
| `data/smpl/UV_TEXTURE_ATTRIBUTION.md` | 归属、许可链接、原始文件及 UV 转换说明 |

**拓扑核对很关键。** 官方样例 OBJ 实际是 27,578 顶点、27,576 细分四边形，
不能直接当成 6,890 顶点的原始 SMPL。转换器按共有 face-center 收缩细分拓扑，
恢复 6,876 个四边形和 24 个三角形父面；本地 neutral 的 13,776 个三角面全部
唯一匹配父面，原始顶点编号、面绕序、逐角 UV 接缝均校验通过。男女映射后的
逐角 UV 最大差为 **0**。没有使用最近邻、裁掉后续顶点或直接重用 OBJ 面顺序。

`smpl_uv.npz` 是明确注明修改的衍生资源：保留 UV 值，按本地模型 `f` 重排 UV
角索引；本地 neutral 模型、几何和两张图片没有修改。原始 normal maps 和 clothing
preview 图片没有进入 albedo 池。

真实模型的 CPU 构建结果：现有 `SMPLMeshVisualizer.from_config` 返回
`has_texture=true`、`uv_valid=true`、`texture_pool=2`；4 人按 seed 随机循环分配
这两张纹理。这只是本审查执行的 CPU 几何/配置验证。

父任务随后提供的真实运行结果 `output/resource_validation/textured_pedestrians/result.json`
已记录 `status=passed`、`mesh_appearance=textured`：4 人、240 步、8 秒，捕获 80 帧
10 Hz 视频，两张纹理都已用于人物；最低 root 高度 0.7348 m。本审查没有启动该 Kit。
这是短时外观和物理证据，较长时段的测试与修复结果由主报告单独记录。
证据：

- [UV 几何验证](smpl_uv_geometry_verification.json)
- [纹理大小/hash/图像解码](smpl_texture_installation.json)
- [现有可视化代码 CPU 验证](smpl_textured_cpu_verification.json)
- [可复用离线转换器](install_meshcapade_smpl_uv.py)，默认仅验证；`--write` 排他创建，拒绝覆盖现存 UV。

## 2. 当前 SMPL 权重和配置附件没有缺失

官方 [预训练模型目录](https://nvlabs.github.io/ProtoMotions/getting_started/pretrained_models.html)
列出 SMPL MaskedMimic、SMPL flat tracker、SMPL terrain tracker；它们分别用于稀疏
未来姿势约束、平地参考动作跟踪、地形参考动作跟踪。它们不是同一个导航 AMP 策略。

本次用官方 GitHub revision `7a8417a9f55b1586b0d6e8f46578c5bc9d7ce1f1` 的 LFS
指针元数据核验本地字节。下表每组的 `last.ckpt`、`resolved_configs.pt`、
`resolved_configs_inference.pt` 共 **9 文件**，大小及 SHA-256 全部匹配官方 LFS OID；
本地文件是完整对象，均不是指针。

| 本地目录 | `last.ckpt` 大小 | 已匹配的配置附件 |
|---|---:|---|
| `data/pretrained_models/masked_mimic/smpl` | 229,131,184 | training PT + inference PT |
| `data/pretrained_models/motion_tracker/smpl` | 126,361,575 | training PT + inference PT |
| `data/pretrained_models/motion_tracker/smpl-terrains` | 132,661,223 | training PT + inference PT |

因此无需再次下载这三份大权重，也无需从别的模型库替换配置附件。机器证据见
[官方 LFS 元数据](protomotions_smpl_remote_lfs.json)与[当前完整文件验证](protomotions_smpl_local_verification.json)。
本地另外 5 组 SOMA/G1/GPC 文件也都是实物而非指针；其角色和骨架不等于论文
SMPL 导航 AMP。先前 8 组严格实例化证据见 `docs/full_audit_20261001/resources.json`；
本次没有重复声明对它们进行了新的 GPU 运行测试。

## 3. Hugging Face 旧 MaskedMimic 不能直接覆盖当前版本

[作者的 HF 模型卡](https://huggingface.co/ctessler/MaskedMimic/blob/d2cd712c515bdab7c59502f97b999e43816b1727/README.md)
仍使用 `eval_agent.py` 与 Hydra `+robot=smpl` 入口。匿名 HF CLI 元数据确认仓库
公开、非 gated，revision 为 `d2cd712c515bdab7c59502f97b999e43816b1727`；只有
`config.yaml`、`last.ckpt` 等，没有当前所需的两个序列化 PT 配置附件。
该模型卡标注 NCSL；应按其许可证处理，不能用本地核心包的 Apache 声明替代外部资源许可。

本地是 ProtoMotions 3.1.0：当前官方[推理入口](https://nvlabs.github.io/ProtoMotions/getting_started/quickstart.html)
使用 `inference_agent.py`、`--checkpoint` 及配套 inference PT。只有旧 YAML/ckpt
不能建立当前 agent 类路径、观测、归一化和 context 合同。不要以名字相同为理由
将 HF 旧文件覆盖已经逐字节匹配的当前 SMPL 权重。

HF CLI 使用隔离的空 `HF_HOME`，显式禁用 implicit token；没有 login、账户探测或
模型下载。记录见 [HF 公开 metadata](hf_maskedmimic_public_metadata.json)。

## 4. MotionLib、模型与 Sim 6 / Lab 3 入口

官方 [AMASS 准备说明](https://nvlabs.github.io/ProtoMotions/getting_started/amass_preparation.html)
要求自行获取 AMASS 与 SMPL 1.1.0，并将原始动作转换和打包为 SMPL MotionLib。
本地已有 train 9,565,828,107、validation 678,009,566、test 65,614,188 字节；
当前 neutral 247,186,228 字节已安全接入，已有真实 LBS 与四人短时物理证据。
本次仅重新确认这些文件的存在/大小，没有重下 AMASS 或对 9.6 GB 文件重复计算 hash。

官方文档描述三个性别参数模型。当前 neutral mesh 与已处理 MotionLib 推理使用
neutral；同轮资源任务还安全接入了已有本地 male 247,101,031 字节与 female
247,530,000 字节模型，并通过受限解码核对 6,890 顶点 / 13,776 面 / 24 关节。
证据见 `smpl_male_installation.json`、`smpl_female_installation.json`；模型保持原有
许可，并非新下载或由纹理许可重新授权。SMPL-X 不能替代当前 24 body / 69 DOF SMPL。

**必须显式指定匹配的 SMPL MotionLib。** 本地 flat/terrain tracker 的原始 inference
sidecar 默认 `motion_lib.motion_file` 仍指向 G1 tiny 数据（YAML 第 1579 行）；不能将
G1 的 33 body / 29 DOF 当成 SMPL 输入。当前 CrowdSim runtime 会显式覆盖。

```bash
../IsaacLab-Protomotions/.venv/bin/python protomotions/inference_agent.py \
  --checkpoint data/pretrained_models/motion_tracker/smpl/last.ckpt \
  --motion-file ../Assets/motion/amass_smpl_test.pt \
  --simulator isaaclab --num-envs 1 --headless
```

以上是重现入口示例，本次没有执行。实际使用的 pin 与
[官方迁移说明](https://nvlabs.github.io/ProtoMotions/user_guide/isaaclab3_migration.html)
一致：IsaacLab 源码 `4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8`，产品 `3.0.0` / Python
包 `12.0.0`，Isaac Sim `6.0.0.1`，Python 3.12。MJCF→USD / D6 兼容及 XYZW 边界已在
当前工程迁移；无需下载旧 `smpl_humanoid.usda` 来代替已验证的 MJCF 转换通路。
精确配置见 `env/isaaclab3_sim6/profile.json`。

## 5. 公开 AMP 参考与严格论文复现的边界

本次查到的官方 ProtoMotions 模型 inventory 没有可直接接入当前 CrowdSim 的
SMPL 导航 AMP checkpoint + 轨迹 task/config/normalization 套件。它包含 SOMA FSQ AMP
跟踪器，后者的骨架和观测任务不同。这个结论限定于本次检查的官方目录，不能推断
任何作者从未发布其他资源。

另有公开 [PACER 官方实现](https://github.com/nv-tlabs/pacer/tree/baeb6241f03fc9d78370a64e3c2dcef9d830d97f)：
它提供 SMPL 轨迹跟踪 AMP 参考，并在
[download_data.sh](https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/download_data.sh)
发布 Drive IDs。其原始运行依赖 Python 3.8 / Isaac Gym 与 rl-games，不是当前
ProtoMotions 3.1 的 `last.ckpt`/PT sidecar 格式。父任务要求将其放在独立
`data/pretrained_models/paper_references/pacer/`。本次要求的 **5 项全部下载完成**：

| 隔离参考文件 | 实际字节 | 校验 |
|---|---:|---|
| `output/release/pacer/Humanoid.pth` | 121,648,127 | SHA-256 + 完整 ZIP CRC |
| `output/release/pacer_no_shape/Humanoid.pth` | 119,755,239 | SHA-256 + 完整 ZIP CRC |
| `sample_data/amass_isaac_standing_upright_slim.pkl` | 2,161,089 | SHA-256 / 响应大小 / 原始文件名 |
| `sample_data/amass_isaac_gender_betas_unique.pkl` | 69,402 | 同上 |
| `sample_data/amass_copycat_occlusion_v2.pkl` | 2,733,463 | 同上 |

共 246,367,320 字节。每项来源 ID、原始文件名、服务端大小及实物 SHA-256 见
[PACER 独立 manifest](pacer_download_manifest.json)。官方脚本未发布校验和，因此本地
hash 是后续完整性和来源记录；下载器没有反序列化 pickle 或执行上游安装脚本。
官方 `LICENSE`（CC BY-NC-SA 4.0）与 `RESOURCE_PROVENANCE.md` 已保存在隔离目录；
SMPL / AMASS 的原始许可仍适用。

可复用工具 `scripts/fetch_pacer_reference.py` 只含上述固定官方 ID，默认不使用代理，
不登录或读取 cookie。它支持已校验文件跳过和严格 HTTP 206 断点续传，限制单文件
150 MiB、单批最多 750 MiB、默认 10 分钟；先写临时副本验证，再原子发布且拒绝覆盖
不同内容的现有文件。下载成功不等于已经移植或严格复现 NavIsaacLab 论文策略。

本地论文 IV-A/IV-B/IV-C 指定：H/N/M 条件 U-Net、5 秒未来轨迹，SMPL 导航 AMP
actor/discriminator，以及 2,848 段 locomotion 筛选清单。当前验证链是 SFM + SMPL
MaskedMimic。仍需 matching 权重/任务接入、对应训练数据与清单、评估 manifest；
TRACE/PACER 公开参考不能凭文件名自动补齐这些合同。论文证据及缺口详见
`docs/migration_20261001/reference_comparison.md` 与论文提取
`docs/full_audit_20261001/navisaaclab_paper.txt:396,475`。

## 6. 尚存 LFS 与其他外观来源

本轮纹理安装前的扫描确认 `protomotions/data/assets/usd` 中有 49 个 LFS 指针，
当前 `data/pretrained_models` 中没有指针。它们包含旧预制 body USD、markers、
materials 及机器人分层 USD；其中小的 marker/material 可按实际引用分支补齐，
但 49 个指针不能统一当作默认 SMPL 行人运行的必要缺口。当前主链用真实 MJCF。

[SURREAL 官方](https://github.com/gulvarol/surreal)的 SMPL/动作配套需要申请访问并
遵从其许可；本次没有登录或访问受限内容。
[SMPLitex 作者仓库](https://github.com/dancasas/SMPLitex)公开生成式 SMPL 纹理资料，
但现成 albedo 只需图片与匹配 UV，并不需要下载其 Stable Diffusion 纹理生成权重。
本次采用明确许可并已经验证的 Meshcapade 样例，没有将 SMPL-X atlas 或服装预览
当成 SMPL 全身 albedo。
