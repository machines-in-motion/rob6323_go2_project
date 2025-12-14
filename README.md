# Project 2

## Major addition
1. New rewards added:
   1. rew_action_rate: $||action||^2$
   2. raibert_heuristic: gait footstep
   3. orient: base flatness (orientation expressed via projected gravity)
   4. lin_vel_z: base linear velocity tracking
   5. dof_vel: joint velocity regularization
   6. ang_vel_xy: base angular velocity regularization
   7. feet_clearance: foot height regularization (during swing phase)
   8. tracking_contacts_shaped_force: foot contact force penalty (druing swing phase) $\exp(-F)-1$
   9. torque_penalty: small regularization on $||\tau||^2$
2. New observation added: clock inputs (sine signal parameterizing the gait phases)
3. New termination condition: base height < 0.5m
4. Actuator friction model with randomization
   1. friction torque added to `_apply_action`
   2. random friction coefficients in `_reset_idx`
5. Others: according to tutorial - `_reward_raibert_heuristic` for computing raibert footstep and `_step_contact_targets` to compute foot phases (desired foot contact states).

## Log comments:

2025-12-14_16-46-47: baseline model
2025-12-14_17-38-49: without friction model
2025-12-14_17-38-49: with friction model


## Command line
```
# assume in folder {}/rob6323_go2_project/source/rob6323_go2
# train policy
python ../../../rob6323_go2_project/scripts/rsl_rl/train.py --task=Template-Rob6323-Go2-Direct-v0 --headless
# play policy
python ../../../rob6323_go2_project/scripts/rsl_rl/play.py  --task=Template-Rob6323-Go2-Direct-v0
```

## Local installation of IsaacLab

```
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab
pip install --upgrade pip
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```