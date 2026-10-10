#!/usr/bin/env python3
"""Rebuild the exact selected 2,848 AMASS clips from a licensed generic train bank.

No new selection or labels. Original source IDs/segments stay fixed. The final
bank must match the published full SHA; failure preserves the candidate file.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def logical_source(value):
    marker = '/processed/smpl/'
    if marker not in str(value):
        raise ValueError('AMASS source identity needs the processed/smpl namespace')
    return str(value).split(marker, 1)[1].split('#frames=', 1)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-bank', type=Path, required=True)
    parser.add_argument('--split', choices=('train','validation','test'), default='train')
    parser.add_argument('--manifest', type=Path, default=ROOT/'data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'data/datasets/amass')
    args = parser.parse_args()
    import torch
    from CrowdSim.humanoid.training.amass_curation import (
        FIELDS, _slice_clip, resample_clip, joint_angular_velocity_xyzw,
        canonical_fingerprint, _refresh_segment_velocity,
    )
    from protomotions.robot_configs.smpl import SmplRobotConfig
    torch.set_num_threads(4)
    manifest = json.loads(args.manifest.read_text())
    entry = manifest['splits'][args.split]
    if args.split != 'train':
        from social_navigation.serialized_bank import write_bank
        bank = torch.load(args.source_bank, weights_only=True, map_location='cpu', mmap=True)
        result = write_bank(ROOT/f'configs/reproduction/amass_containers/{args.split}.json',
            bank, args.output_dir/f'amass_smpl_{args.split}.pt')
        print(json.dumps(result,indent=2))
        return 0
    if entry['clip_count'] != 2848 or entry['dof_velocity_frame'] != 'child_joint':
        raise ValueError('Expected the frozen v7 real-AMASS child-joint corpus')
    bank = torch.load(args.source_bank, weights_only=True, map_location='cpu', mmap=True)
    names = [logical_source(value) for value in bank['motion_files']]
    if len(set(names)) != len(names):
        raise ValueError('Generic bank has ambiguous logical AMASS source identities')
    lookup = dict(zip(names, range(len(names))))
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    # Torch ZIP stores the output basename: keep the original archive root.
    target = destination/'manifest_labeled_v7_train.pt'
    if target.exists():
        raise FileExistsError('Preserve existing fixed bank: '+str(target))
    information = SmplRobotConfig().kinematic_info
    tensors = {key: [] for key in (*FIELDS, 'contacts')}
    cache = {}
    remaining = {}
    for row in entry['clips']:
        key = logical_source(row['parent_source_path'])
        remaining[key] = remaining.get(key, 0) + 1
    for index, row in enumerate(entry['clips']):
        if row['motion_id'] != index:
            raise ValueError('Fixed membership/order changed')
        key = logical_source(row['parent_source_path'])
        if key not in cache:
            source_id = lookup[key]
            clip = _slice_clip(bank, source_id)
            dt = float(row['resampling']['source_dt'])
            if abs(float(bank['motion_dt'][source_id]) - dt) > 1e-7:
                raise ValueError('Source sample clock differs from fixed metadata')
            cache[key], _ = resample_clip(clip, dt, 30, information, dof_velocity_frame='parent_joint')
        parent = cache[key]
        first, last = row['parent_control_frame_start'], row['parent_control_frame_stop']
        selected = _refresh_segment_velocity({field: value[first:last].clone() for field, value in parent.items()},30,dof_velocity_frame='parent_joint')
        if len(selected['gts']) != row['frames'] or canonical_fingerprint(selected, row['dt']) != row['canonical_sha256']:
            raise ValueError('Fixed pose/source segment fingerprint differs at '+str(index))
        selected['dvs'] = joint_angular_velocity_xyzw(selected['lrs'][:,1:], row['dt'], frame='child_joint').reshape(row['frames'],69)
        for field in tensors:
            tensors[field].append(selected[field])
        remaining[key] -= 1
        if not remaining[key]:
            del cache[key]
        if (index+1)%100==0:
            print(json.dumps({'rebuilt_clips':index+1,'total':2848}),flush=True)
    counts = torch.tensor([row['frames'] for row in entry['clips']], dtype=torch.long)
    result = {field: torch.cat(values).contiguous() for field, values in tensors.items()}
    result.update(length_starts=counts.cumsum(0)-counts, motion_num_frames=counts,
        motion_dt=torch.full((2848,),1/30),motion_lengths=(counts-1)/30,
        motion_weights=torch.full((2848,),1/2848),
        motion_files=tuple(row['source_path'] for row in entry['clips']),dof_velocity_frame='child_joint')
    from social_navigation.serialized_bank import write_bank
    write_bank(ROOT/'configs/reproduction/amass_containers/train.json',result,target)
    actual = digest(target)
    passed = actual==entry['sha256'] and target.stat().st_size==entry['bytes']
    receipt = {'schema':'social_navigation.fixed_amass_rebuild.v1','status':'passed' if passed else 'failed_preserved',
        'source_bank':str(args.source_bank.resolve()),'source_bank_sha256':digest(args.source_bank),
        'manifest_sha256':digest(args.manifest),'train_clips':2848,'new_selection':False,
        'output':str(target),'bytes':target.stat().st_size,'sha256':actual,'expected_sha256':entry['sha256']}
    (destination/'fixed_amass_rebuild.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
    if not passed:
        raise ValueError('Numeric fingerprints passed but full bank identity differs; candidate retained and not admitted')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
