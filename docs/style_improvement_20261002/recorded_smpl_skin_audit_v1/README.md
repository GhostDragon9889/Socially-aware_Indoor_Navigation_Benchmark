# 记录回放的 SMPL 外观与真实姿态绑定审查

## 结论

现有回放已经使用**完整 SMPL LBS 蒙皮、6,890 顶点、13,776 三角面和 UV 贴图**，不是把24个物理碰撞体作为最终行人外观。本次无需新增蒙皮渲染器，也没有改动 source181、isolated source931、训练、控制器、物理评估或任何既有记录。

本次检查仅CPU执行，未启动Kit/App/Context、CUDA、渲染或物理。已存在的真实室内回放结果和本次CPU数值核验分别保留，不把CPU核验当新的GPU验证。

## 已有代码与绑定数学

| 环节 | 实际代码 | 合同 |
| --- | --- | --- |
| 记录准入 | `scripts/replay_paper_pedestrians.py:75` | 完整终态、30Hz、XYZW、common24body/69DOF语义顺序、reset时间戳校验；不跨reset插值 |
| 创建人体 | `scripts/replay_paper_pedestrians.py:297` | CPU `SMPLMeshVisualizer`，明确`shape_std=0`，中性体型，现有两张4096² albedo图 |
| 关节重排 | `CrowdSim/tools/smpl_mesh_visualizer.py:232` | 根据69个语义名或D6后缀映射到标准SMPL24关节，不猜PhysX遍历顺序 |
| 局部旋转 | `CrowdSim/tools/smpl_mesh_visualizer.py:652` | 原生exp-map转`R_robot`，用`R_smpl=CᵀR_robot C`；`C=Rz(π/2)Rx(π/2)` |
| SMPL LBS | `CrowdSim/tools/smpl_mesh_visualizer.py:598` | `smplx.SMPL`计算shape/pose blend shapes、关节FK和每顶点24关节加权刚体变换 |
| 世界坐标 | `CrowdSim/tools/smpl_mesh_visualizer.py:626` | 先减SMPL骨盆，转`C`坐标，再用记录根旋转和根位置：`v_world=R_root C(v_LBS−J_pelvis)+p_root`；beta0时floor correction严格0 |
| 逐帧更新 | `scripts/replay_paper_pedestrians.py:328` | 实际记录DOFs/root直接驱动皮肤；XYZW显式重排为WXYZ；写`UsdGeom.Mesh.points`，无策略推理或姿态拟合 |
| UV/材料 | `CrowdSim/tools/smpl_mesh_visualizer.py:562` | `faceVarying` UV保留面角接缝，`UsdPreviewSurface` albedo纹理；几何不是采样OBJ的替代骨架 |
| 原生场景 | `CrowdSim/humanoid/runtime/native_visual_scene.py:151` | 已有App/stage组合原始DynBench视觉USD，禁用源PhysicsScene/碰撞；不建立Physics Context |
| 实时捕获 | `scripts/train_paper_amp.py:715` | 也调用同一个蒙皮adapter，beta0并隐藏物理Robot；与记录回放的执行合同分别报告 |

安装的主实现为`smplx 0.1.28`，`smplx/lbs.py:208`起依次实现`v_shaped`、joint regressor、pose offsets、`batch_rigid_transform`、`T_v=Σ_j w_vj A_j`及齐次顶点变换。本次记录该实现文件SHA，未修改依赖。

## 实际数据核验

源物理记录：`output/diffusion_amp/native_warehouse_clearance08_30s_v2/physical_states.npz`。已完成视觉回放：`output/diffusion_amp/native_warehouse_recorded_replay_indoor_30s_v1/result.json`。该回放实际450帧/15fps/30秒，`mesh_mode=textured`、0物理步、0策略forward、timeline0→0。直接查看已生成`frames/frame_00000.png`可见连续着装人体网格，而非分段碰撞几何；此单帧不能证明全部行人每时刻均可见。

本次重算全900控制帧×4人，无挑选好帧；以同控制帧最后一个120Hz子步的真实`body_link_pos_w/body_link_quat_w`作参照。ground文件内common名字与MJCF逐名一致；最后子步刚体位置与`physical_states.body_pos`逐字节一致，未混入COM或其他时间点。

| 全量CPU核验项 | 实测 |
| --- | --- |
| 实际行人帧 | 3,600 |
| 24关节比较 | 86,400 |
| 实际计算蒙皮顶点 | 24,804,000，57批全部finite |
| 蒙皮关节FK与原生link origin位置差 | 平均0.07998mm，P95 0.19002mm，最大0.23931mm |
| 重构关节世界旋转与原生link rotation差 | 平均0.00003309rad，P95 0.00034279rad，最大0.00070322rad |
| 根与子步时间合同 | 根/身体位置为真实post-control；未重置、未引入GT3D、未拟合原生body位置 |
| beta/floor correction | 全4人beta0、floor correction0 |
| 精确资源绑定 | 物理states/result/ground、MJCF、SMPL模型、UV、纹理均重算完整SHA |

结果详见[actual_cpu_audit.json](actual_cpu_audit.json)。现检查的replay、SMPL visualizer、simulate、rotations、native_visual_scene五份生产源SHA，**全部与这次实际历史回放自身的source_code记录相等**，见[historical_source_identity.json](historical_source_identity.json)；这里没有把当前整树身份当作历史完整import ledger。

## 明确的外观边界

1. **体型/服装多样性有限。** 当前显式beta0与neutral物理rig保持骨长一致，仅两张albedo交替分配。male/female两份许可PKL虽已在本地，回放没有使用它们；女性贴图不等于女性身体骨架。衣服是贴图外观，不是独立衣料/头发/软组织仿真。不能直接开启随机beta或切男女模型并声称仍与同一中性物理rig严格匹配。
2. **皮肤与碰撞表面不同。** 实际3,427/3,600行人帧的最低皮肤顶点z<0，全部行人帧最低顶点z的均值约−2.37mm。该值来自LBS表面，不是脚接触点/穿透力/滑移测量；不能据此判定原生floor接触失效，也不能用逐帧抬根来掩盖差异。小的关节位置/旋转残差如上，没有假称严格逐位相同或确定其物理成因。
3. **视角影响动作可见性。** 已有外部Warehouse镜头曾被roof遮挡，室内镜头已单独完成真实回放；单视角仍可能遮挡或把行人缩得过小。下一视觉验收应使用既有`--camera-eye/--camera-target`另起明确标注的室内近景回放，而不是修改控制器/记录或把渲染成功当作所有人始终可见。
4. **视觉不能修复步态。** 当前管线忠实展示真实记录的动作，没有额外动作平滑、重定向、姿态写入或控制参数。动作自然度仍须由已冻结全22/200秒物理评估和正式统计判断，增加皮肤不等于论文行为质量达标。

## 本地许可与复跑

本地`data/smpl/SMPL_NEUTRAL.pkl`、`SMPL_MALE.pkl`、`SMPL_FEMALE.pkl`为已许可资源的symlink，本次仅读取。UV归档为`uvs[28944,2]`/`uv_faces[13776,3]`；完整模型/纹理SHA在JSON。本地纹理来源和派生UV合同见`data/smpl/UV_TEXTURE_ATTRIBUTION.md`及`data/smpl/textures/LICENSE.md`；样例纹理许可不重新许可参数PKL。

CPU工具：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
PYTHONPATH=/tmp/protomotions-review-tools-20261001:.:src \
../IsaacLab-Protomotions/.venv/bin/python \
docs/style_improvement_20261002/recorded_smpl_skin_audit_v1/audit.py
```

该工具完整执行exit0，Ruff通过。原输出存在时明确拒绝覆盖，因此交付JSON保留真实首次完整执行；复跑应先复制工具到新的独立docs证据目录并保留原结果，不删除或替换历史证据。没有新增需要运行SDK的候选，也没有改变训练/物理参数。
