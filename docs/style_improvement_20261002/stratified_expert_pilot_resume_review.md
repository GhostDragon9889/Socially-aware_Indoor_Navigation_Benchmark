# 真实 4+4 更新 pilot：CPU 恢复核验

核验时间：2026-10-02T06:03:42.277178+00:00。完整结果见 [JSON](stratified_expert_pilot_resume_review.json)。

实际 step4 与 step8 均以封存 trainer 在 CPU 加载；模型、两优化器、四 normalizer、完整 policy replay、runner state、待恢复 Python / NumPy / Torch CPU / CUDA RNG payload 均逐项一致。私有采样器与共享 corpus RNG 的后续抽样确定续接。

实际 pilot 为 8 更新 / 65,536 transitions。前四更新跌倒 5 次；恢复四更新跌倒 1 次；分段合计 6 次。最终 result 的 1 仅覆盖恢复段。

未启动 Kit / GPU，未执行逐步 GPU 轨迹比较，未在设备上恢复 CUDA RNG。PhysX 接触与求解器缓存不在 checkpoint 中，因此逻辑 payload 一致不代表物理逐位等价，也不代表训练收敛或论文复现。
