"""Training-only temporal gait statistics fitted to real AMASS locomotion.

Rewards actual motion windows, never prescribes poses or a sinusoid.
Reset history when an actor resets; no interval crosses an episode boundary.
"""
import torch
from .kinematic_gait_prior import features


def gait_angles(state, names):
    _, targets = features(state, names)
    # Each row pairs a leg with the contralateral arm. Statistics share sides.
    return targets[..., [2, 1]]  # radians: arm sagittal angle, knee flexion


class TemporalGaitPrior:
    def __init__(self, data, actors, device):
        if (data['schema'] != 'crowdsim.real_amass_temporal_gait.v1'
                or data['fit_split'] != 'train' or data['train_clip_count'] != 2848):
            raise ValueError('Fixed real train AMASS statistics required')
        self.data = data
        self.device = device
        self.steps = int(data['window_controls'])
        if self.steps != 60 or data['control_hz'] != 30:
            raise ValueError('Physical clock must match the real 2s/30Hz windows')
        self.edges = torch.tensor(data['speed_edges_m_s'], device=device)
        self.std_target = torch.tensor(data['angle_std_target_rad'], device=device)
        self.mean_target = torch.tensor(data['angle_mean_median_rad'],device=device)
        self.rate_target = torch.tensor(data['angular_rate_rms_target_rad_s'], device=device)
        if (self.std_target.shape != (len(self.edges)-1, 2)
                or self.rate_target.shape != self.std_target.shape
                or self.mean_target.shape != self.std_target.shape
                or not torch.isfinite(self.mean_target).all()
                or not torch.isfinite(self.std_target).all()
                or not torch.isfinite(self.rate_target).all()
                or (self.std_target <= 0).any() or (self.rate_target <= 0).any()
                or (self.edges[1:] <= self.edges[:-1]).any()):
            raise ValueError('Positive finite side-shared fitted gait statistics required')
        self.angles = torch.zeros(actors, self.steps, 2, 2, device=device)
        self.speed = torch.zeros(actors, self.steps, device=device)
        self.count = torch.zeros(actors, dtype=torch.long, device=device)
        self.cursor = 0

    def reset(self, ids):
        self.count[ids] = 0

    @torch.no_grad()
    def reward(self, state, names):
        self.angles[:, self.cursor] = gait_angles(state, names)
        self.speed[:, self.cursor] = state.root_vel[:, :2].norm(dim=-1)
        self.cursor = (self.cursor + 1) % self.steps
        self.count = (self.count + 1).clamp_max(self.steps)
        order = (torch.arange(self.steps, device=self.device) + self.cursor) % self.steps
        angle = self.angles[:, order]
        speed = self.speed[:, order]
        valid = torch.arange(self.steps, device=self.device)[None] >= self.steps-self.count[:, None]
        denominator = valid.sum(-1).clamp_min(1)
        mask = valid[..., None, None]
        mean = (angle*mask).sum(1)/denominator[:, None, None]
        std = (((angle-mean[:, None]).square()*mask).sum(1)/denominator[:, None, None]).sqrt()
        intervals = valid[:, 1:] & valid[:, :-1]
        delta = torch.atan2((angle[:, 1:]-angle[:, :-1]).sin(), (angle[:, 1:]-angle[:, :-1]).cos())*30
        rms = ((delta.square()*intervals[..., None, None]).sum(1)/intervals.sum(-1).clamp_min(1)[:, None, None]).sqrt()
        mean_speed = (speed*valid).sum(-1)/denominator
        moving_fraction = (((speed >= self.edges[0]) & (speed <= self.edges[-1]))*valid).sum(-1)/denominator
        bins = torch.bucketize(mean_speed, self.edges[1:-1]).clamp(0, len(self.edges)-2)
        std_score = torch.exp(-(torch.log(std.clamp_min(.001)/self.std_target[bins, None])/.45).square())
        rate_score = torch.exp(-(torch.log(rms.clamp_min(.001)/self.rate_target[bins, None])/.5).square())
        bilateral = torch.exp(-(torch.log(std[:, 0].clamp_min(.001)/std[:, 1].clamp_min(.001))/.35).square())
        mean_score=torch.exp(-((mean-self.mean_target[bins,None])/self.std_target[bins,None]).square())
        ready = (self.count >= 30) & (moving_fraction >= .8) & (mean_speed >= self.edges[0]) & (mean_speed <= self.edges[-1])
        reward = (std_score.mean(1)*std_score.new_tensor([.8,.5]) + rate_score.mean(1)*std_score.new_tensor([.25,.15]) + bilateral*.25 + mean_score.mean(1)*mean_score.new_tensor([.15,.05])).sum(-1)*ready
        metrics = {'temporal_gait_reward':reward,'temporal_gait_ready':ready.float(),'actual_arm_angle_std_rad':std[...,0].mean(-1),'actual_knee_angle_std_rad':std[...,1].mean(-1),'temporal_arm_target_std_rad':self.std_target[bins,0],'temporal_knee_target_std_rad':self.std_target[bins,1]}
        return reward, metrics
