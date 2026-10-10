#!/usr/bin/env python3
"""CPU comparison of preserved200s captures; writes only a new report folder."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_paper_pedestrians import read_capture  # noqa: E402


def source_record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def analyze_run(path):
    arrays, runner, validated = read_capture(path/'physical_states.npz', path/'result.json')
    evaluation = json.loads((path/'pedestrian_metrics/evaluation.json').read_text())
    if evaluation['sources']['physical_states']['sha256'] != source_record(path/'physical_states.npz')['sha256']:
        raise ValueError('Preserved capture differs from the evaluated source')
    root = arrays['root_pos'].astype(np.float64)
    tracking = np.linalg.norm(root[..., :2] - arrays['target_xy'], axis=-1)
    same_episode = arrays['episode_ids'][1:] == arrays['episode_ids'][:-1]
    legal = same_episode & ~arrays['reset_mask'][1:]
    distance = np.linalg.norm(np.diff(root[..., :2], axis=0), axis=-1)
    phase_rows = []
    for start in range(0, len(root), 600):
        end = min(start + 600, len(root))
        valid = legal[start:end-1]
        peer = arrays.get('peer_contact')
        phase_rows.append({'time_s': [start/30, end/30], 'terminal_frames': int(arrays['terminated'][start:end].sum()),
            'tracking_mean_m': float(tracking[start:end].mean()), 'tracking_p95_m': float(np.quantile(tracking[start:end], .95)),
            'mean_xy_speed_mps': float(distance[start:end-1][valid].mean()*30),
            'walked_distance_m_per_actor': np.where(valid, distance[start:end-1], 0).sum(0).tolist(),
            'static_contact_actor_frames': int(arrays['static_contact'][start:end].sum()) if 'static_contact' in arrays else None,
            'peer_contact_pair_frames': int(np.triu(peer[start:end] | peer[start:end].swapaxes(-1, -2), 1).sum()) if peer is not None else None})
    physical = evaluation['rollout_metrics']
    fall_count = int(arrays['terminated'].sum())
    exposure = len(root)*root.shape[1]/30
    result = {'sources': {key: source_record(path/name) for key,name in (
                ('runner','result.json'), ('capture','physical_states.npz'),('evaluation','pedestrian_metrics/evaluation.json'))},
        'capture_validation': validated, 'arguments': runner['arguments'], 'physics': runner['physics'],
        'effective_navigation_guidance': runner['navigation_guidance'], 'diffusion_guidance': runner.get('diffusion_guidance'),
        'runtime_execution': runner['runtime_execution'], 'evaluation_planner_change': runner['evaluation_planner_change'],
        'physical_actor_seconds': exposure, 'terminal_falls': fall_count, 'terminal_falls_per_actor': arrays['terminated'].sum(0).tolist(),
        'falls_per1000_actor_seconds': fall_count/exposure*1000, 'timeout_frames': int(arrays['truncated'].sum()),
        'tracking': {**physical['plan_tracking'], 'p95_xy_error_m': float(np.quantile(tracking, .95))},
        'walked_distance_xy_m_per_actor': physical['walked_distance_xy_m_per_agent'],
        'walked_distance_xy_m_total': physical['walked_distance_xy_m_total'],
        'mean_xy_speed_excluding_resets_mps': float(distance[legal].mean()*30),
        'minimum_pelvis_height_m': physical['minimum_pelvis_ground_height_m'],
        'episode_boundary_disagreements': physical['boundary_source_disagreement_count'],
        'pose_generated': {key: value['generated'] for key,value in evaluation['cluster_scores'].items()},
        'pose_mmd': evaluation['generated_vs_test_gaussian_mmd'],
        'heldout_reference_mmd': evaluation['train_vs_test_gaussian_mmd_reference'],
        'goal_condition_metrics': evaluation['goal_condition_metrics'], 'actual_contacts': physical.get('actual_contacts'),
        'geometry_proxies': evaluation.get('geometry_proxies'), 'phases_20s': phase_rows,
        'elapsed_wall_s': runner['elapsed_s'], 'timing': runner['timing']}
    return result, evaluation, runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=ROOT/'output/diffusion_amp/shared_obstacle_corrected_endpoint_1000')
    parser.add_argument('--new', type=Path, default=ROOT/'output/diffusion_amp/shared_obstacle_clean_sdf_5000_v2')
    parser.add_argument('--output-prefix', type=Path, default=ROOT/'docs/amp_improvement_20261001/clean_5000_physical_comparison')
    args = parser.parse_args()
    baseline, old_eval, old_runner = analyze_run(args.baseline.resolve())
    new, new_eval, new_runner = analyze_run(args.new.resolve())
    if baseline['physical_actor_seconds'] != new['physical_actor_seconds']:
        raise ValueError('Match capture exposure before comparison')
    protocol_keys = ('embedding_choices','reference_sampling','generated_sampling','cluster_choices','train_vs_test_gaussian_mmd_reference')
    protocol_equal = {key: old_eval[key] == new_eval[key] for key in protocol_keys}
    cluster_equal = {str(k): (args.baseline/f'pedestrian_metrics/train_pose_clusters_k{k}.json').read_bytes() ==
                            (args.new/f'pedestrian_metrics/train_pose_clusters_k{k}.json').read_bytes() for k in (50,100)}
    if not all(protocol_equal.values()) or not all(cluster_equal.values()):
        raise ValueError('Pose comparison requires identical references, generated indices and train-only clusters')
    changes = {key: {'baseline': old_runner['arguments'].get(key), 'new': new_runner['arguments'].get(key)}
        for key in old_runner['arguments'].keys() | new_runner['arguments'].keys()
        if old_runner['arguments'].get(key) != new_runner['arguments'].get(key)}
    amp_paths = [Path(r['arguments']['resume']).resolve() for r in (old_runner,new_runner)]
    if amp_paths[0] != amp_paths[1]:
        raise ValueError('This diagnostic expects the same controller checkpoint')
    weights = {'controller': source_record(amp_paths[0])}
    for key, runner in (('baseline_diffusion',old_runner),('new_diffusion',new_runner)):
        weights[key] = source_record(runner['arguments']['diffusion_checkpoint'])
        if weights[key]['sha256'] != runner['evaluation_planner_change']['evaluation_sha256']:
            raise ValueError('A pinned evaluated diffusion checkpoint changed on disk')
    phase_quality = json.loads((args.new/'pedestrian_metrics/phase_quality.json').read_text())
    terminals = phase_quality['terminal_contact_relation']
    contact_relation = {'same_frame_static': sum(x['static_contact_same_frame'] for x in terminals),
                        'same_frame_peer': sum(x['peer_contact_same_frame'] for x in terminals),
                        'terminal_count': len(terminals), 'windows_inclusive_s': {}}
    for window in ('0.5','1.0','2.0'):
        contact_relation['windows_inclusive_s'][window] = {
            'terminals_with_static_contact': sum(x['same_episode_lookback_inclusive_s'][window]['static_contact_actor_frames'] > 0 for x in terminals),
            'terminals_with_peer_contact': sum(x['same_episode_lookback_inclusive_s'][window]['peer_contact_actor_frames'] > 0 for x in terminals),
            'terminals_with_either_contact': sum(any(x['same_episode_lookback_inclusive_s'][window].values()) for x in terminals)}
    report = {'schema': 'clean_diff5000_physical_comparison_v1', 'status': 'cpu_analyzed',
        'baseline': baseline, 'new': new, 'weights': weights, 'argument_changes': changes,
        'pose_protocol_equal': protocol_equal, 'train_cluster_artifacts_byte_identical': cluster_equal,
        'new_terminal_contact_relation': contact_relation, 'new_terminal_events': terminals,
        'new_route_vertex_progress': phase_quality['recorded_route_vertex_progress'],
        'same_controller_checkpoint': True, 'single_factor_causal_claim': False, 'author_metric_equivalence': False,
        'limitations': ['One200s four-actor realization per configuration at declared seed7, not a multi-seed paired statistical experiment.',
            'Diffusion checkpoint, guidance implementation/weights, collision extensions, batch setting and compilation/render/contact processing changed together.',
            'Old capture has no actual contact arrays: baseline collision values remain unknown, never0.',
            'Route reached counts are cumulative observed vertices across resets, not uninterrupted full-route success.',
            'Pose MMD is a45D per-frame local protocol; fixed train-only centers and sample identities are matched, but does not evaluate temporal gait realism.',
            'Fewer falls and lower tracking coexist with lower average movement; contact/slowdown remain actionable.',
            'Terminal lookback contact association does not prove a cause; floor/self/tangential components are excluded.'],
        'analysis_script': source_record(__file__)}
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['# Diff5000 clean guidance：真实四行人200秒对照', '',
        'CPU读取并验证两份6000帧×4人记录；保留历史baseline不重写。AMP控制器相同，planner和运行参数有多项变化，不能单因归因。', '',
        '## 相同暴露的结果', '', '| 指标 | baseline corrected_endpoint_1000 | clean_sdf_5000_v2 |', '| --- | --- | --- |']
    fields = [('terminal falls','terminal_falls'),('每1000 actor-seconds跌倒','falls_per1000_actor_seconds'),
              ('reset不计入的总XY路程(m)','walked_distance_xy_m_total'),('平均XY速度(m/s)','mean_xy_speed_excluding_resets_mps')]
    for title,key in fields:
        lines.append(f'| {title} | {baseline[key]:.6g} | {new[key]:.6g} |')
    for title,key in [('tracking mean(m)','mean_xy_error_m'),('tracking RMS(m)','rmse_xy_m'),('tracking p95(m)','p95_xy_error_m')]:
        lines.append(f'| {title} | {baseline["tracking"][key]:.6g} | {new["tracking"][key]:.6g} |')
    for title,key in [('路线顶点累计到达','route_vertices_reached_total'),('A*失败查询','global_path_failures')]:
        lines.append(f'| {title} | {baseline["goal_condition_metrics"][key]} | {new["goal_condition_metrics"][key]} |')
    lines += [f'| plan endpoint到condition goal mean(m) | {baseline["goal_condition_metrics"]["recorded_plan_endpoint_goal_residual"]["mean_m"]:.6g} | {new["goal_condition_metrics"]["recorded_plan_endpoint_goal_residual"]["mean_m"]:.6g} |',
              f'| Gaussian MMD² / σ1 | {baseline["pose_mmd"]["mmd_squared"]:.9f} | {new["pose_mmd"]["mmd_squared"]:.9f} |']
    for k in ('50','100'):
        lines += [f'| K{k} coverage | {baseline["pose_generated"][k]["coverage"]:.2f} | {new["pose_generated"][k]["coverage"]:.2f} |',
                  f'| K{k} normalized entropy | {baseline["pose_generated"][k]["normalized_pose_entropy"]:.6f} | {new["pose_generated"][k]["normalized_pose_entropy"]:.6f} |']
    contacts = new['actual_contacts']
    static_stats = contacts['static']['actor_frame_statistics']
    peer_stats = contacts['peer']['unordered_pair_statistics']
    peer_actor_stats = contacts['peer']['actor_frame_statistics']
    window2 = contact_relation['windows_inclusive_s']['2.0']
    lines += ['', f'每人falls：旧{baseline["terminal_falls_per_actor"]}，新{new["terminal_falls_per_actor"]}。新route vertices={new["goal_condition_metrics"]["route_vertices_reached"]}；完整路线成功率未知。', '',
        '姿态参考：train/test各2048、seed7、显式15节点45D／root heading+height规范化、σ1 biased Gaussian MMD；生成2048个同一time-major索引。两份train-only K50/100文件逐字节完全相同，AMASS样本IDs、fit和embedding协议完全相同。train/test参考MMD²=.035753752。K50覆盖下降、K100略升，不能称所有姿态指标统一改善。', '',
        '## 新run真实法向接触', '',
        '| 接触统计 | 接触帧／全部观测帧 | 比例 | 连续事件 | 最大单body-pair力(N) |',
        '| --- | --- | --- | --- | --- |',
        f'| Static actor-frames | {static_stats["observed_flagged_frames"]}/{static_stats["observed_frames"]} | {100*static_stats["observed_flagged_frames"]/static_stats["observed_frames"]:.5f}% | {static_stats["observed_event_starts"]} actor incidences | {contacts["static"]["maximum_body_pair_force_n"]:.2f} |',
        f'| Peer unordered pair-frames | {peer_stats["observed_contact_pair_frames"]}/{peer_stats["observed_pair_frames"]} | {100*peer_stats["observed_contact_pair_frames"]/peer_stats["observed_pair_frames"]:.5f}% | {peer_stats["contiguous_pair_contact_incidences"]} pair incidences | {contacts["peer"]["maximum_body_pair_force_n"]:.2f} |',
        f'| Peer actor-frames | {peer_actor_stats["observed_flagged_frames"]}/{peer_actor_stats["observed_frames"]} | {100*peer_actor_stats["observed_flagged_frames"]/peer_actor_stats["observed_frames"]:.5f}% | {peer_actor_stats["observed_event_starts"]} actor incidences | 同上peer力 |', '',
        '严格normal force>1N，控制帧聚合全部4物理子步最大body-pair幅度，排除floor/self；不包含摩擦切向力或全身总冲量。旧run没有这些数组，所以历史actual collision保持未知；不能据新数据断言碰撞改善。', '',
        f'{len(terminals)}个terminal当帧static={contact_relation["same_frame_static"]}、peer={contact_relation["same_frame_peer"]}；同episode前2s有static{window2["terminals_with_static_contact"]}／peer{window2["terminals_with_peer_contact"]}，合并{window2["terminals_with_either_contact"]}个。其余{len(terminals)-window2["terminals_with_either_contact"]}个近2s无已记录接触。这里只提供时间关系。', '',
        '## 分段与残留问题', '', '| 时段(s) | 旧/新falls | 旧/新tracking mean(m) | 新static actor-frames | 新peer pair-frames | 新速度(m/s) |', '| --- | --- | --- | --- | --- | --- |']
    for old_phase,new_phase in zip(baseline['phases_20s'],new['phases_20s'],strict=True):
        lines.append(f'| {new_phase["time_s"]} | {old_phase["terminal_frames"]}/{new_phase["terminal_frames"]} | {old_phase["tracking_mean_m"]:.4f}/{new_phase["tracking_mean_m"]:.4f} | {new_phase["static_contact_actor_frames"]} | {new_phase["peer_contact_pair_frames"]} | {new_phase["mean_xy_speed_mps"]:.4f} |')
    lines += ['', 'Actor2无fall，但80–100s仅走.804m，180–200s仅.421m；末20s actor0/2共343个接触pair-frames。新actor1从2次fall增至5次；总体减少并不意味着每人都改善。低tracking误差不能代替速度、全局路线和身体接触质量。', '',
        '## 同时变化的模型、采样与参数', '',
        f'两run相同controller：`{weights["controller"]["path"]}`，SHA256 `{weights["controller"]["sha256"]}`。', '',
        f'Diffusion旧1000 SHA `{weights["baseline_diffusion"]["sha256"]}` → 新5000 SHA `{weights["new_diffusion"]["sha256"]}`。', '',
        '旧为legacy posterior guidance weight1；新为官方clean目标端梯度（独立actor求和、actor内waypoint均值）weight20＋peer5/radius.4＋本地signed-AABB几何10/radius.5。新参数speed/acceleration penalties仍为0，不能声称推理严格限速限加速度。', '',
        '旧planner batch64、compiled forward／render+400帧视频；新batch8、eager／physics-only、真实contact monitor。四人仍整组处理；代码路径与数值、运行耗时不能当完全相同。实际DDPM100步、FP32、15控制步重规划、endpoint guidance与seed7保持。', '',
        '场景cuboids、bounds、起点声明、路线、到达半径和A*尺寸一致；新配置文件额外声明timed_path/recovery.5，但本run CLI仍endpoint且recovery0，不能把它当timed guidance效果。旧physics未显式保存terrain_extent字段；新记录64m，不自行把缺失值当不同时钟或地面变更。', '',
        f'墙钟旧{baseline["elapsed_wall_s"]:.2f}s／新{new["elapsed_wall_s"]:.2f}s；本对照是物理效果，不是纯planner性能benchmark。', '',
        '单seed、多项变化、episode censoring与PhysX非逐位恢复限制因果结论。后续要固定新Diff5000/guidance/物理与起点，比较独立低层课程transfer→500→1000；保持同接触与姿态协议并增加seed。', '',
        '[新逐帧／phase图与terminal证据](/project/EXP/SangforAndSaut/ProtoMotions/output/diffusion_amp/shared_obstacle_clean_sdf_5000_v2/pedestrian_metrics/phase_quality.md)。完整source hashes、参数diff、测量与采样合同见相邻JSON。', '', '![分段对照](clean_5000_physical_comparison.png)']
    args.output_prefix.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(3,1,figsize=(10,8),sharex=True,constrained_layout=True)
    x = np.arange(len(new['phases_20s']))*20+10
    for run,label in ((baseline,'Baseline: Diff1000 / legacy'),(new,'Diff5000 / clean + collision guidance')):
        axes[0].plot(x,[p['terminal_frames'] for p in run['phases_20s']],marker='o',label=label)
        axes[1].plot(x,[p['tracking_mean_m'] for p in run['phases_20s']],marker='o')
        axes[2].plot(x,[p['mean_xy_speed_mps'] for p in run['phases_20s']],marker='o')
    axes[0].set_ylabel('Falls per20s')
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Tracking mean (m)')
    axes[2].set_ylabel('Root XY speed (m/s)')
    axes[2].set_xlabel('Physical time interval midpoint (s)')
    axes[0].set_title('Same controller, multiple planner/parameter changes; descriptive one-seed comparison')
    for axis in axes:
        axis.grid(alpha=.2)
    fig.savefig(args.output_prefix.with_suffix('.png'),dpi=160)
    plt.close(fig)
    print(json.dumps({'status':'cpu_analyzed','output_prefix':str(args.output_prefix),'falls':[baseline['terminal_falls'],new['terminal_falls']],
                      'mmd_squared':[baseline['pose_mmd']['mmd_squared'],new['pose_mmd']['mmd_squared']]}))


if __name__ == '__main__':
    main()
