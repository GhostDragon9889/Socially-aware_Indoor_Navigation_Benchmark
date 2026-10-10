# Social 推理／value 数值检查（未来 CPU，尚未执行）

本目录只新增 `test_social_inference_future_CPU.py`。当前正式训练仍运行，本轮仅编译和 Ruff，不 import 此程序、原 fixture、Torch、NN、checkpoint 或 SDK。没有数值结果；源码审阅／编译不能当作运行通过。

## 单一 fixture 与边界

未来独立进程仅直接调用已封 `test_social_trainer_future_CPU.py`（12,606 B／`98de7ef9…`）的 `FutureSocialTrainerCPU.trained_fixture()`。该缓存方法产生一次原32-example合成 PPO／D update；不运行原四项 unittest，也不调用另一轮 update、fresh 或训练恢复。原 teardown 会清缓存，因此本程序自行保有 fixture 至四项检查结束后再清理，不从已经结束的原 runner 获取它。

数据全部来自该方法的合成 rawself／rawtask／history；其 `physical_rollout=False`，计数1／32是算法 fixture 的计数，不是实际物理或 AMASS 观察。fixture 的明确 `root/root/root_z/exclude_ego` 几何**不同于**真实评估 producer 的 `root/root/static_center_ground_z/exclude_ego` 合同；不得拿它冒称 SDK 接线通过。

## 四项必要检查

1. [冻结 actor](../social_deployment_integration_v1/inference_adapter.py) `05261849…` 从该已更新 trainer 取 owned snapshot。B1／B4 deterministic mean 与 applied57逐 bit 对照同一 trainer 的 deterministic输出，验证输入未改变。构造／推理期间禁用新的 Linear、Conv、RMS、trainer、Adam、replay、Normal构造，挂钩拒绝原 critic／D forward。trainer baseline policy 计算在这些挂钩之外；它本来包含 critic，不能误称 baseline 无 critic。
2. 错误的模型 SHA 必须在 payload load／NN来源 admission 前被拒绝；spy确认 `torch.load` 和 loader `_owned_module` 未调用，不依赖“坏模型恰巧被其它规则拒绝”。
3. 同一 fixture `save` 一次到普通、独立 nlink1临时 snapshot，设0444并算全文件 SHA。只调用一次 [严格 checkpoint loader](../social_checkpoint_inference_staging_v1/checkpoint_inference.py) `cd227f5d…`，检查 `/proc/self/fd/N`／CPU／weights_only／mmap参数及前后身份。允许其明确的一次临时完整 NN初始化来验证所有权重，禁用 trainer／Adam／replay／Normal构造并拒绝临时 critic／D forward；最终 B1／B4与owned snapshot逐 bit一致。验证 caller CPU RNG恢复、checkpoint-declared计数标签、无训练 RNG恢复／device move、最终actor／RMS使用独立存储。临时 snapshot是本次合成 fixture 生成的，不读取生产或外部权重。
4. [冻结 critic](../social_value_adapter_staging_v1/value_adapter.py) `9875c58b…` 只 deepcopy现有 value path。候选为B4×S2×10×2，仅替换20个本地XY输入；绑定的当前self／raster即使原输入之后被修改也保持不变。对照同一 trainer 的原 RMS→共享CNN→critic→value RMS公式，值逐 bit比较，candidate gradient用 `rtol=1e-6, atol=1e-7`比较。保留输入梯度，检查有限／非零梯度；一个刻意超clip5的坐标应梯度零。没有新critic训练、世界坐标转换、选优或guidance质量声称。

每项执行前后都核查 CUDA未初始化、无SDK package、Torch CPU／Python／NumPy全局RNG及四组原 RMS完全未变。fixture初始化／其单次update本来会 seed和抽样，RNG保留检查从 fixture完成之后开始；不能宣称整个程序从训练构造前保留RNG。程序开始／结束核对依赖和自身完整 source SHA。没有新的结果／receipt包装文件。

## 源码来源

程序 `PINS` 固定 trainer `81a6319f…`、fixture `98de7ef9…`、actor `05261849…`、checkpoint reader `cd227f5d…`、critic `9875c58b…` 的完整字节数和SHA；复用这些来源的配置、PPO／D、normalization／forward，不改封存来源或生产185。

## 未来唯一执行命令

仅在正式训练**真实退出0**且所需主比较完成、root明确安排后执行；现在不得执行：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python \
  docs/style_improvement_20261004/social_inference_future_CPU_v1/test_social_inference_future_CPU.py -v
```

未来stdout／实际退出码由root记录。即使四项通过，也只说明该合成 CPU fixture 的数值／所有权兼容性；真实训练的 checkpoint、22动作、四人200s、GPU、Nav场景和论文效果仍需各自实际验证。

## 独立源码审阅与封存

作者静态编译／AST检查 `a5266b`、Ruff `8e67d0` 通过；root完整源码阅读 `4f4f00` 与说明阅读 `5dab8a`，独立审阅者完整来源核对 `9b8442`，Required=[]／Optional=[]。源码17,955 B，SHA256 `159f87419521ed80bb73deb606c0e0aecfbeef72c98ffed2ade86904dc098a5e`；root设0444、确认普通独立nlink1。未执行四项检查或任何NN／SDK。当前训练真实会话50502仍运行，不把静态通过当作数值通过。
