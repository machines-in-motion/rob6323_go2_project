# Go2 Locomotion Environment Implementation Guide

This document summarizes all modifications made to the baseline Isaac Lab environment to create a research-grade locomotion controller for the Unitree Go2 robot. Each section corresponds to a part of the tutorial and explains **what** was changed, **why** it improves learning, and **how** to tune the relevant parameters.

---

## Table of Contents

1. [Overview](#overview)
2. [Part 1: Action Rate Penalties (State History)](#part-1-action-rate-penalties-state-history)
3. [Part 2: Custom PD Controller](#part-2-custom-pd-controller)
4. [Part 3: Early Stopping (Min Base Height)](#part-3-early-stopping-min-base-height)
5. [Part 4: Raibert Heuristic (Gait Shaping)](#part-4-raibert-heuristic-gait-shaping)
6. [Part 5: Refined Reward Function](#part-5-refined-reward-function)
7. [Part 6: Advanced Foot Interaction](#part-6-advanced-foot-interaction)
8. [Tunable Parameters Reference](#tunable-parameters-reference)
9. [Expected Training Behavior](#expected-training-behavior)

---

## Overview

The baseline environment provides basic velocity tracking for the Go2 robot. The modifications progressively add:

- **Smooth motion** via action rate penalties
- **Precise control** via custom PD torque control
- **Faster training** via early termination
- **Natural gaits** via Raibert heuristic and phase-based rewards
- **Stable posture** via orientation and velocity penalties
- **Proper foot interaction** via clearance and contact rewards

### Files Modified

| File | Purpose |
|------|---------|
| `rob6323_go2_env_cfg.py` | Configuration class with all tunable parameters |
| `rob6323_go2_env.py` | Environment implementation with reward logic |

---

## Part 1: Action Rate Penalties (State History)

### What Was Changed

1. **Action History Buffer**: Added a tensor `self.last_actions` of shape `(num_envs, 12, 3)` to store the current and two previous actions.

2. **Reward Computation**: Added penalties for:
   - **First derivative** (action rate): `||a_t - a_{t-1}||^2`
   - **Second derivative** (action acceleration): `||a_t - 2*a_{t-1} + a_{t-2}||^2`

3. **Reset Logic**: Clear action history on environment reset.

### Why This Improves Learning

- **Reduces jittery motion**: Without this penalty, policies often learn high-frequency oscillations that are unrealistic and would damage real hardware.
- **Promotes energy efficiency**: Smooth actions require less torque change, mimicking natural animal locomotion.
- **Aids sim-to-real transfer**: Actuators have bandwidth limits; smooth commands transfer better to hardware.

### Key Code Locations

```python
# In __init__:
self.last_actions = torch.zeros(self.num_envs, 12, 3, ...)

# In _get_rewards:
rew_action_rate = torch.sum(torch.square(self._actions - self.last_actions[:, :, 0]), dim=1)
rew_action_rate += torch.sum(torch.square(self._actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1)
```

---

## Part 2: Custom PD Controller

### What Was Changed

1. **Disabled Implicit PD**: Set `stiffness=0.0` and `damping=0.0` in the actuator config.

2. **Added Custom Gains**: Defined `Kp=20.0`, `Kd=0.5`, and `torque_limits=100.0`.

3. **Implemented Torque Control**: In `_apply_action()`, compute:
   ```
   τ = Kp * (q_desired - q_current) - Kd * q_dot
   ```
   Then clip to torque limits.

### Why This Improves Learning

- **Full control over dynamics**: You can tune gains for different behaviors (stiffer for fast response, softer for compliance).
- **Matches real hardware**: Real robots use explicit PD or impedance controllers.
- **Enables advanced techniques**: Future work can add feedforward terms, variable impedance, etc.

### Key Code Locations

```python
# In _apply_action:
torques = torch.clip(
    self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos)
    - self.Kd * self.robot.data.joint_vel,
    -self.torque_limits, self.torque_limits
)
self.robot.set_joint_effort_target(torques)
```

---

## Part 3: Early Stopping (Min Base Height)

### What Was Changed

1. **Added Termination Threshold**: `base_height_min = 0.20` (20 cm).

2. **Updated `_get_dones`**: Check if `root_pos_w[:, 2] < threshold` and terminate if true.

### Why This Improves Learning

- **Speeds up training**: Episodes that would waste time with a collapsed robot are terminated early.
- **Shapes exploration**: The policy learns that falling is bad, encouraging upright postures.
- **Reduces wasted samples**: Fallen robots produce uninformative gradients.

### Key Code Locations

```python
# In _get_dones:
base_height = self.robot.data.root_pos_w[:, 2]
cstr_base_height_min = base_height < self.cfg.base_height_min
died = cstr_termination_contacts | cstr_upsidedown | cstr_base_height_min
```

---

## Part 4: Raibert Heuristic (Gait Shaping)

### What Was Changed

1. **Gait Phase Variables**: Added `gait_indices`, `clock_inputs`, `foot_indices`, and `desired_contact_states`.

2. **Foot Body Indices**: Found feet in both robot (`_feet_ids`) and sensor (`_feet_ids_sensor`) indexing.

3. **Clock Inputs**: 4-dimensional sinusoidal signals added to observations (observation space: 48 → 52).

4. **Raibert Reward**: Penalizes deviation between actual foot position and the position predicted by the Raibert Heuristic.

### Why This Improves Learning

- **Teaches proper stepping**: The Raibert Heuristic is a classical control formula that places feet to stabilize velocity.
- **Enforces gait timing**: Phase variables create a natural trot/walk rhythm.
- **Provides dense reward signal**: Foot placement errors give continuous feedback, not just success/failure.

### Gait Phase Explanation

The gait uses a trotting pattern with:
- **Frequency**: 3 Hz (3 steps per second per leg)
- **Phase offset**: Diagonal pairs (FL+RR and FR+RL) alternate
- **Duty factor**: 50% stance, 50% swing

### Key Code Locations

```python
# In _step_contact_targets:
self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * frequencies, 1.0)
self.clock_inputs[:, i] = torch.sin(2 * np.pi * foot_indices[i])

# In _reward_raibert_heuristic:
err_raibert_heuristic = torch.abs(desired_footsteps - actual_footsteps)
reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))
```

---

## Part 5: Refined Reward Function

### What Was Changed

Added four penalty terms:

| Reward | Penalizes | Encourages |
|--------|-----------|------------|
| `orient` | Tilted body (projected gravity on XY) | Staying upright |
| `lin_vel_z` | Vertical bouncing | Smooth horizontal motion |
| `dof_vel` | High joint velocities | Energy efficiency |
| `ang_vel_xy` | Roll/pitch angular velocity | Stable torso orientation |

### Why This Improves Learning

- **Stable posture**: Penalizing tilt prevents the robot from leaning excessively.
- **Smooth motion**: Vertical velocity penalty reduces bouncing gaits.
- **Joint protection**: High joint velocities can damage motors.
- **Natural appearance**: Low roll/pitch rates look more natural.

### Key Code Locations

```python
# In _get_rewards:
rew_orient = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=1)
rew_lin_vel_z = torch.square(self.robot.data.root_lin_vel_b[:, 2])
rew_dof_vel = torch.sum(torch.square(self.robot.data.joint_vel), dim=1)
rew_ang_vel_xy = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1)
```

---

## Part 6: Advanced Foot Interaction

### What Was Changed

Added two phase-dependent rewards:

1. **Feet Clearance Reward**: Penalizes low feet during swing phase.
   - Target height: 8 cm at mid-swing + 2 cm foot radius
   - Only active during swing (when `desired_contact_states` is low)

2. **Contact Force Reward**: Penalizes contact forces during swing phase.
   - Uses exponential penalty: `1 - exp(-f²/100)`
   - Only active during swing

### Why This Improves Learning

- **Proper foot lifting**: Without clearance reward, the robot may drag feet, causing trips.
- **Clean swing phase**: Contact during swing indicates stumbling or poor timing.
- **Phase synchronization**: Rewards reinforce the gait timing from Part 4.

### Critical Implementation Note

The contact sensor has its own indexing separate from the robot body indices:
```python
# Robot body indices (for positions):
self._feet_ids = [robot.find_bodies("FL_foot")[0], ...]

# Sensor indices (for forces):
self._feet_ids_sensor = [contact_sensor.find_bodies("FL_foot")[0], ...]
```

---

## Tunable Parameters Reference

### Reward Scales

| Parameter | Current Value | Effect of ↑ | Effect of ↓ | Tuning Guidance |
|-----------|---------------|-------------|-------------|-----------------|
| `lin_vel_reward_scale` | 2.0 | Stronger velocity tracking | More exploration | Primary objective; keep high |
| `yaw_rate_reward_scale` | 1.0 | Better turning | Ignores heading | Balance with lin_vel |
| `action_rate_reward_scale` | -0.01 | Smoother motion | Allows jitter | Increase if motion is jerky |
| `raibert_heuristic_reward_scale` | -1.0 | Stricter foot placement | More freedom | Increase for precise gaits |
| `orient_reward_scale` | -1.0 | More upright | Allows tilt | Increase if robot leans |
| `lin_vel_z_reward_scale` | -0.02 | Less bouncing | Allows hopping | Increase if bouncy |
| `dof_vel_reward_scale` | -0.0001 | Slower joints | Faster joints | Keep small; high values freeze robot |
| `ang_vel_xy_reward_scale` | -0.001 | Stable torso | Allows wobble | Increase if unstable |
| `feet_clearance_reward_scale` | -1.0 | Higher foot lift | Low feet OK | Increase if dragging feet |
| `tracking_contacts_shaped_force_reward_scale` | 4.0 | Cleaner swing | Allows ground contact | Positive reward; tune with clearance |

### Control Parameters

| Parameter | Current Value | Effect of ↑ | Effect of ↓ | Tuning Guidance |
|-----------|---------------|-------------|-------------|-----------------|
| `Kp` | 20.0 | Stiffer joints | Softer joints | Higher for fast tracking |
| `Kd` | 0.5 | More damping | Less damping | Higher for stability |
| `torque_limits` | 100.0 | More power | Safer limits | Match real hardware |
| `action_scale` | 0.25 | Larger motion range | Smaller range | Affects exploration |

### Termination Parameters

| Parameter | Current Value | Effect of ↑ | Effect of ↓ | Tuning Guidance |
|-----------|---------------|-------------|-------------|-----------------|
| `base_height_min` | 0.20 m | Stricter (terminates higher) | More lenient | Set to ~50% standing height |
| `episode_length_s` | 20.0 s | Longer episodes | Shorter episodes | Longer for complex behaviors |

### Gait Parameters (Hardcoded in `_step_contact_targets`)

| Parameter | Current Value | Meaning |
|-----------|---------------|---------|
| `frequencies` | 3.0 Hz | Steps per second per leg |
| `phases` | 0.5 | Diagonal offset (trot) |
| `durations` | 0.5 | Stance/swing ratio |

---

## Expected Training Behavior

### Early Training (0-100 iterations)

- Robot learns to stand upright
- May fall frequently
- Rewards are low and variable

### Mid Training (100-300 iterations)

- Robot starts moving in commanded direction
- Gait becomes periodic (trotting pattern)
- Foot clearance improves
- Fewer falls

### Late Training (300-500 iterations)

- Smooth, stable trotting gait
- Accurate velocity tracking
- Proper foot placement following Raibert heuristic
- Minimal body oscillation

### TensorBoard Metrics to Monitor

| Metric | Expected Trend | Concern if... |
|--------|----------------|---------------|
| `track_lin_vel_xy_exp` | Increases | Stays near 0 |
| `track_ang_vel_z_exp` | Increases | Stays near 0 |
| `rew_action_rate` | Stays near 0 (small negative) | Large negative values |
| `raibert_heuristic` | Becomes less negative | Stays very negative |
| `orient` | Near 0 | Large negative |
| `Episode_Termination/base_contact` | Decreases | Stays high |

---

## Summary of All Rewards

| Reward Key | Type | Scale | Purpose |
|------------|------|-------|---------|
| `track_lin_vel_xy_exp` | Positive | 2.0 | Follow velocity commands |
| `track_ang_vel_z_exp` | Positive | 1.0 | Follow yaw commands |
| `rew_action_rate` | Penalty | -0.01 | Smooth actions |
| `raibert_heuristic` | Penalty | -1.0 | Proper foot placement |
| `orient` | Penalty | -1.0 | Stay upright |
| `lin_vel_z` | Penalty | -0.02 | No bouncing |
| `dof_vel` | Penalty | -0.0001 | Energy efficiency |
| `ang_vel_xy` | Penalty | -0.001 | Stable torso |
| `feet_clearance` | Penalty | -1.0 | Lift feet during swing |
| `tracking_contacts_shaped_force` | Reward | 4.0 | No contact during swing |

---

## References

- [Isaac Lab ANYmal C Environment](https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c)
- [DMO IsaacGymEnvs Go2 Implementation](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/go2_terrain.py)
- [Raibert, M. H. "Legged Robots That Balance" (1986)](https://mitpress.mit.edu/9780262681193/legged-robots-that-balance/)
- [Isaac Lab ArticulationData API](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.ArticulationData)
- [Isaac Lab ContactSensorData API](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html#isaaclab.sensors.ContactSensorData)
