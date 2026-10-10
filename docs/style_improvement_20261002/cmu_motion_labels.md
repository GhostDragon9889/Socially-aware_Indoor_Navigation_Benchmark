# CMU 官方动作说明与本地 AMASS 文件对应

## 已完成的元数据核对

2026-10-02 获取 [CMU 官方完整 motions 页面](https://mocap.cs.cmu.edu/search.php) 的单份 HTML，955,392 bytes，SHA256 `7a5bd8c60b22df22b77c8ab3a0b5d6bf24d93511a01f971d169cd3c5f236312b`。HTTPS 直连及代理出现 TLS EOF，随后官方 HTTP URL 成功；另一独立 HTTP 路由取得的字节及哈希完全相同。HTTP 本身不提供 TLS 身份认证，若后续要求 HTTPS provenance，应重新获取并比对；本次试次说明已与搜索服务缓存的官方页面做交叉抽查。

仅读取完整说明页，没有下载 ASF/AMC/C3D、视频或许可动作数据，也没有遍历每个 subject / trial。官网 [FAQ](https://mocap.cs.cmu.edu/faqs.php) 提醒不要爬取所有动作，并提供批量动作下载入口；本次不使用动作下载入口。

| 覆盖范围 | 数量 |
| --- | ---: |
| 当前官方说明页 trial / subject | 2,435 / 109 |
| 本地 train CMU 实际文件 / 数值 subject+take | 1,919 / 1,881 |
| 本地数值 subject | 98 |
| 具有非空官方试次说明的实际文件 | 1,805 |
| 对应官方 trial 存在，但 description 空白 | 64 |
| 不在当前完整说明页中的实际文件 | 50 |

缺失的 50 行分别为 subject 63 的29行、84的12行、73的9行。主页搜索缓存曾报告2,605 trials；不能将这个总数当作本次实际取得的说明覆盖数，也不能从缺失数量猜动作类别。

## 文件与使用方式

- `cmu_motion_labels.json`：全部1,919本地行、原始试次说明、未知原因、源URL、摘要和限制。
- `cmu_motion_labels_join.json`：供资源筛选脚本 `--labels` 使用，**实际完整 source_path → 非空 description + source_url + provenance**。只含1,805条，不将未知描述伪造成导航标签。
- `cmu_official_motions_20261002.html`：冻结的官方元数据响应。
- `cmu_actual_train_ids_20261002.json`：资源代理从已有 train mmap 的 `motion_files` 导出的实际ID清单。
- `parse_cmu_motion_metadata.py`：离线重建工具，拒绝覆盖既有输出。每个试次同时核对表格 trial、subject 标题、feedback URL 与 AMC **链接文件名**；只解析链接，不获取 AMC 内容。

身份对应使用文件 basename 中的整数 subject / take。例如 `18_19_Justin/18_04_poses.motion` 对应 `(18,4)`；目录名不能代替数值ID。所有1,919个 source_path 唯一，38个数值 trial 的额外目录别名保留为独立本地行。当前官方页面2,435个 feedback记录与解析条数一致；全部 join 行均有非空 description 和官方域名 source_url。工具 Ruff 检查通过。

重建到新的临时输出：

```bash
python docs/style_improvement_20261002/parse_cmu_motion_metadata.py \
  --html docs/style_improvement_20261002/cmu_official_motions_20261002.html \
  --local-ids docs/style_improvement_20261002/cmu_actual_train_ids_20261002.json \
  --output /tmp/cmu_motion_labels_rebuilt.json \
  --join-output /tmp/cmu_motion_labels_join_rebuilt.json
```

## 筛选边界

本产物提供官方动作描述，**没有推断最终导航/style类别**。包含 walk/run 字符串不代表整段是纯导航。官方试次69/68包含走近、拾起并放置物体；86/2混合行走、深蹲、跑步、伸展、跳跃、拳击与饮水。因此资源筛选应独立执行负面动作排除，不能先匹配 walk 就接受整段。stylized / weird / baby / martial walking也应保留字面说明，由显式目标分布决定取舍。

description 空白、官方列表缺失与具有混合动作的试次是不同状态；缺失不得由 session description 补齐。资料不能恢复 NavIsaacLab 作者未公开的确切2,848动作清单；最终筛选不足2,848时应报告真实数量。

CMU [官方首页](https://mocap.cs.cmu.edu/) 说明同一人可对应多个 subject 编号，因此 subject namespace 分离不能证明人的身份分离。首页也提示 toe / hand 动捕噪声，finger / thumb 并非真实捕获；官方动作说明不会自动构成足接触、滑步或物理质量的真值。
