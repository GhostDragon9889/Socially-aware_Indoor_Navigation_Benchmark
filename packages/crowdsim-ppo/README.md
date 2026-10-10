# Simulator-free CrowdSim PPO

This package provides the PointGoal PPO example from the NavIsaacLab reference.
It uses NumPy and PyTorch and runs separately from the IsaacLab pedestrian simulator.

From the main repository:

```bash
python -m pip install -e ./packages/crowdsim-ppo
crowdsim-ppo-train --device cpu --total-steps 4096 --output output/point_goal
crowdsim-ppo-eval --device cpu --checkpoint output/point_goal/latest.pt --episodes 20
```

The commands locate the source checkout's `configs/point_goal.json`, or use the
packaged default when installed from a wheel. Pass `--config PATH` to supply
another configuration. The package includes no simulator assets or pretrained
weights. For real SMPL pedestrians see the main repository's migration guide.
