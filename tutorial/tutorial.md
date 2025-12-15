# Isaac Lab Advanced Locomotion Tutorial: From Basic to Expert

In this tutorial, you will learn how to extend a basic `DirectRLEnv` implementation for the Unitree Go2 robot into a sophisticated locomotion controller. We will start with a minimal environment and progressively add features used in modern reinforcement learning research.

**What you will learn:**
1.  **Adding State Variables**: How to track history and internal states (like previous actions and gait phases).
2.  **Custom Controllers**: Implementing a low-level PD controller with manual torque calculation.
3.  **Termination Criteria**: Adding early stops based on robot state (e.g., base height).
4.  **Advanced Rewards**: Implementing the Raibert Heuristic for precise foot placement.
5.  **Observation Expansion**: Adding new signals to the policy input.

---

## Part 1: Adding Action Rate Penalties (State History)

Smooth motion requires penalizing jerky actions. To do this, we need to track the history of actions taken by the policy.

### 1.1 Update Configuration
First, define the reward scale in your configuration file.

```python
# In Rob6323Go2EnvCfg (source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env_cfg.py)

# reward scales
action_rate_reward_scale = -0.1
```

### 1.2 Update `__init__`
We need a buffer to store the last few actions. We'll store a history of length 3 (current + 2 previous). Also, update the logging keys to track this new reward.

```python
# In Rob6323Go2Env.__init__ (source/rob6323_go2/rob6323_go2/tasks/direct/rob6323_go2/rob6323_go2_env.py)

# Update Logging
self._episode_sums = {
    key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    for key in [
        "track_lin_vel_xy_exp",
        "track_ang_vel_z_exp",
        "rew_action_rate",     # <--- Added
        "raibert_heuristic"    # <--- Added
    ]
}

# variables needed for action rate penalization
# Shape: (num_envs, action_dim, history_length)
self.last_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), 3, dtype=torch.float, device=self.device, requires_grad=False)
```

### 1.3 Update `_reset_idx`
When an environment resets, we must clear this history so the new episode starts fresh.

```python
# In Rob6323Go2Env._reset_idx

# Reset last actions hist
self.last_actions[env_ids] = 0.
```

### 1.4 Update `_get_rewards`
We calculate the "rate" (first derivative) and "acceleration" (second derivative) of the actions to penalize high-frequency oscillations. Note that we removed `self.step_dt` from the original tracking rewards to align with standard implementations.

```python
# In Rob6323Go2Env._get_rewards

# action rate penalization
# First derivative (Current - Last)
rew_action_rate = torch.sum(torch.square(self._actions - self.last_actions[:, :, 0]), dim=1) * (self.cfg.action_scale ** 2)
# Second derivative (Current - 2*Last + 2ndLast)
rew_action_rate += torch.sum(torch.square(self._actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1) * (self.cfg.action_scale ** 2)

# Update the prev action hist (roll buffer and insert new action)
self.last_actions = torch.roll(self.last_actions, 1, 2)
self.last_actions[:, :, 0] = self._actions[:]

# Add to rewards dict
rewards = {
    "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale, # Removed step_dt
    "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale, # Removed step_dt
    "rew_action_rate": rew_action_rate * self.cfg.action_rate_reward_scale,
}
```

---

## Part 2: Implementing a Low-Level PD Controller

Instead of relying on the physics engine's implicit PD controller, we will implement our own torque-level control. This gives us full control over the gains and limits.

### 2.1 Update Configuration
First, disable the built-in PD controller in the config and define our custom gains.

```python
# In Rob6323Go2EnvCfg
# add this import:
from isaaclab.actuators import ImplicitActuatorCfg

# PD control gains
Kp = 20.0  # Proportional gain
Kd = 0.5   # Derivative gain
torque_limits = 100.0  # Max torque

# Update robot_cfg
robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
# "base_legs" is an arbitrary key we use to group these actuators
robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    effort_limit=23.5,
    velocity_limit=30.0,
    stiffness=0.0,  # CRITICAL: Set to 0 to disable implicit P-gain
    damping=0.0,    # CRITICAL: Set to 0 to disable implicit D-gain
)
```

### 2.2 Initialize Controller Parameters
In the environment class, we load these gains into tensors for efficient computation.

```python
# In Rob6323Go2Env.__init__

# PD control parameters
self.Kp = torch.tensor([cfg.Kp] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
self.Kd = torch.tensor([cfg.Kd] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
self.motor_offsets = torch.zeros(self.num_envs, 12, device=self.device)
self.torque_limits = cfg.torque_limits
```

### 2.3 Implement Control Logic
We calculate the torques manually using the standard PD formula: $\tau = K_p (q_{des} - q) - K_d \dot{q}$.

```python
# In Rob6323Go2Env

def _pre_physics_step(self, actions: torch.Tensor) -> None:
    self._actions = actions.clone()
    # Compute desired joint positions from policy actions
    self.desired_joint_pos = (
        self.cfg.action_scale * self._actions 
        + self.robot.data.default_joint_pos
    )

def _apply_action(self) -> None:
    # Compute PD torques
    torques = torch.clip(
        (
            self.Kp * (
                self.desired_joint_pos 
                - self.robot.data.joint_pos 
            )
            - self.Kd * self.robot.data.joint_vel
        ),
        -self.torque_limits,
        self.torque_limits,
    )

    # Apply torques to the robot
    self.robot.set_joint_effort_target(torques)
```

---

## Part 3: Early Stopping (Min Base Height)

To speed up training, we should terminate episodes early if the robot falls down or collapses. It will also help learning that the base should stay elevated.

### 3.1 Update Configuration
Define the threshold for termination.

```python
# In Rob6323Go2EnvCfg
base_height_min = 0.20  # Terminate if base is lower than 20cm
```

### 3.2 Update `_get_dones`
Check the robot's base height (z-coordinate) against the threshold.

```python
# In Rob6323Go2Env._get_dones

# terminate if base is too low
base_height = self.robot.data.root_pos_w[:, 2]
cstr_base_height_min = base_height < self.cfg.base_height_min

# apply all terminations
died = cstr_termination_contacts | cstr_upsidedown | cstr_base_height_min
return died, time_out
```

---

## Part 4: Raibert Heuristic (Gait Shaping)

The Raibert Heuristic is a classic control strategy that places feet to stabilize velocity. We will use it as a "teacher" reward to encourage the policy to learn proper stepping. For reference logic, see [IsaacGymEnvs implementation](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/tasks/go2_terrain.py#L670).

### 4.1 Update Configuration
Define the reward scales and increase observation space to include clock inputs (4 phases).

```python
# In Rob6323Go2EnvCfg

observation_space = 48 + 4  # Added 4 for clock inputs

raibert_heuristic_reward_scale = -10.0
feet_clearance_reward_scale = -30.0
tracking_contacts_shaped_force_reward_scale = 4.0
```

### 4.2 Setup State Variables
We need to track the "phase" of the gait and identify feet bodies.

```python
# In Rob6323Go2Env.__init__

# Get specific body indices
self._feet_ids = []
foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
for name in foot_names:
    id_list, _ = self.robot.find_bodies(name)
    self._feet_ids.append(id_list[0])

# Variables needed for the raibert heuristic
self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
```

### 4.3 Define Foot Indices Helper
We need to know which body indices correspond to the feet to get their positions.

```python
# In Rob6323Go2Env (add new property)

@property
def foot_positions_w(self) -> torch.Tensor:
    """Returns the feet positions in the world frame.
    Shape: (num_envs, num_feet, 3)
    """
    return self.robot.data.body_pos_w[:, self._feet_ids]
```

### 4.4 Implement Gait Logic
We implement a function that advances the gait clock and calculates where the feet *should* be based on the command velocity. We also need to reset the gait index on episode reset.

```python
# In Rob6323Go2Env._reset_idx
# Reset raibert quantity
self.gait_indices[env_ids] = 0

# In Rob6323Go2Env (add new method)
# Defines contact plan
def _step_contact_targets(self):
    frequencies = 3.
    phases = 0.5
    offsets = 0.
    bounds = 0.
    durations = 0.5 * torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
    self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * frequencies, 1.0)

    foot_indices = [self.gait_indices + phases + offsets + bounds,
                    self.gait_indices + offsets,
                    self.gait_indices + bounds,
                    self.gait_indices + phases]

    self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

    for idxs in foot_indices:
        stance_idxs = torch.remainder(idxs, 1) < durations
        swing_idxs = torch.remainder(idxs, 1) > durations

        idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
        idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                    0.5 / (1 - durations[swing_idxs]))

    self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
    self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
    self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
    self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

    # von mises distribution
    kappa = 0.07
    smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

    smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                        1 - smoothing_cdf_start(
                                    torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
    smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                        1 - smoothing_cdf_start(
                                    torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
    smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                        1 - smoothing_cdf_start(
                                    torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
    smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                        1 - smoothing_cdf_start(
                                    torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

    self.desired_contact_states[:, 0] = smoothing_multiplier_FL
    self.desired_contact_states[:, 1] = smoothing_multiplier_FR
    self.desired_contact_states[:, 2] = smoothing_multiplier_RL
    self.desired_contact_states[:, 3] = smoothing_multiplier_RR
```

### 4.5 Implement Raibert Reward
We calculate the error between where the foot IS and where the Raibert Heuristic says it SHOULD be.

```python
# In Rob6323Go2Env (add new method)

def _reward_raibert_heuristic(self):
    cur_footsteps_translated = self.foot_positions_w - self.robot.data.root_pos_w.unsqueeze(1)
    footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply_yaw(math_utils.quat_conjugate(self.robot.data.root_quat_w),
                                                          cur_footsteps_translated[:, i, :])

    # nominal positions: [FR, FL, RR, RL]
    desired_stance_width = 0.25
    desired_ys_nom = torch.tensor([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.device).unsqueeze(0)

    desired_stance_length = 0.45
    desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.device).unsqueeze(0)

    # raibert offsets
    phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
    frequencies = torch.tensor([3.0], device=self.device)
    x_vel_des = self._commands[:, 0:1]
    yaw_vel_des = self._commands[:, 2:3]
    y_vel_des = yaw_vel_des * desired_stance_length / 2
    desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
    desired_ys_offset[:, 2:4] *= -1
    desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

    desired_ys_nom = desired_ys_nom + desired_ys_offset
    desired_xs_nom = desired_xs_nom + desired_xs_offset

    desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

    err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

    reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

    return reward
```

### 4.6 Integrate into Observations and Rewards
Finally, expose the clock inputs to the policy and add the reward term.

```python
# In Rob6323Go2Env._get_observations
obs = torch.cat([
    # ... existing obs ...
    self.clock_inputs  # Add gait phase info
], dim=-1)

# In Rob6323Go2Env._get_rewards
self._step_contact_targets() # Update gait state
rew_raibert_heuristic = self._reward_raibert_heuristic()

rewards = {
    # ...
    # Note: This reward is negative (penalty) in the config
    "raibert_heuristic": rew_raibert_heuristic * self.cfg.raibert_heuristic_reward_scale,
}
```

---

## Part 5: Refining the Reward Function

To achieve stable and natural-looking locomotion, we need to shape the robot's behavior further. We will add penalties for: 
1.  **Non-flat body orientation** (projected gravity).
2.  **Vertical body movement** (bouncing).
3.  **Excessive joint velocities**.
4.  **Body rolling and pitching** (angular velocity).

### 5.1 Update Configuration

Add the following reward scales to your configuration class.

```python
# In Rob6323Go2EnvCfg

# Additional reward scales
orient_reward_scale = -5.0
lin_vel_z_reward_scale = -0.02
dof_vel_reward_scale = -0.0001
ang_vel_xy_reward_scale = -0.001
```

### 5.2 Implement Reward Terms

Now, implement the logic to calculate these rewards inside `_get_rewards`. 

**Hint:** You can access the robot's state directly from `self.robot.data`. 
-   For orientation, look for `projected_gravity_b`.
-   For base velocities, check `root_lin_vel_b` and `root_ang_vel_b`.
-   For joint velocities, look at `joint_vel`.

Try to implement the following logic using the available data:

```python
# In Rob6323Go2Env._get_rewards

# 1. Penalize non-vertical orientation (projected gravity on XY plane)
# Hint: We want the robot to stay upright, so gravity should only project onto Z.
# Calculate the sum of squares of the X and Y components of projected_gravity_b.
rew_orient = ... 

# 2. Penalize vertical velocity (z-component of base linear velocity)
# Hint: Square the Z component of the base linear velocity.
rew_lin_vel_z = ... 

# 3. Penalize high joint velocities
# Hint: Sum the squares of all joint velocities.
rew_dof_vel = ... 

# 4. Penalize angular velocity in XY plane (roll/pitch)
# Hint: Sum the squares of the X and Y components of the base angular velocity.
rew_ang_vel_xy = ... 

# Add these to your rewards dictionary with their respective scales:
rewards = {
    ...
    "orient": rew_orient * self.cfg.orient_reward_scale,
    "lin_vel_z": rew_lin_vel_z * self.cfg.lin_vel_z_reward_scale,
    "dof_vel": rew_dof_vel * self.cfg.dof_vel_reward_scale,
    "ang_vel_xy": rew_ang_vel_xy * self.cfg.ang_vel_xy_reward_scale,
}
```

**Important:** Don't forget to add these new keys to your logging dictionary in `__init__` so you can track them in Tensorboard!

---

## Part 6: Advanced Foot Interaction

Next, we will add two critical rewards for legged locomotion: **Foot Clearance** (lifting feet during swing) and **Contact Forces** (grounding feet during stance).

We will adapt the implementation from [IsaacGymEnvs](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/tasks/go2_terrain.py#L1142C1-L1154C133).

### 6.1 Update Configuration

Add the scales for these rewards.

```python
# In Rob6323Go2EnvCfg
feet_clearance_reward_scale = -30.0
tracking_contacts_shaped_force_reward_scale = 4.0
```

### 6.2 Sensor Indices (Critical Step)

Before implementing the rewards, you must be careful with indices. The `_feet_ids` you found earlier using `self.robot.find_bodies(...)` work for accessing robot state (positions), but **cannot** be used for sensor data.

The contact sensor has its own internal indexing. You need to find the feet indices *within the sensor* separately.

```python
# In Rob6323Go2Env.__init__

# Define foot names
foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

# 1. Find indices in the ROBOT (for positions/kinematics)
self._feet_ids = []
# ... iterate foot_names and use self.robot.find_bodies ...
# you already have this

# Find indices in the CONTACT SENSOR (for forces)
self._feet_ids_sensor = []
# ... iterate foot_names and use self._contact_sensor.find_bodies ...
# Be sure to store these! You will need them for the force reward.
```

### 6.3 Implement Rewards

Now, reimplement the logic from the reference code using Isaac Lab's API.

**Key Correspondences:**
*   `self.foot_indices` (Gait Phase) -> You already computed this in `_step_contact_targets`.
*   `foot_height` -> Use `self.foot_positions_w` (which uses `_feet_ids`).
*   `self.contact_forces` -> Use `self._contact_sensor.data.net_forces_w`. **Important:** Index this using `self._feet_ids_sensor`.

By following these steps, you have transformed a simple environment into a research-grade locomotion setup capable of learning robust walking gaits!

---

## Resources

- [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/ecosystem.html) — Everything you need to know about IsaacLab, and more!
- [Isaac Lab ANYmal C environment](https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c) — This targets ANYmal C (not Unitree Go2), so use it as a reference and adapt robot config, assets, and reward to Go2.
- [DMO (IsaacGym) Go2 walking project page](https://machines-in-motion.github.io/DMO/) • [Go2 walking environment used by the authors](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/tasks/go2_terrain.py) • [Config file used by the authors](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/cfg/task/Go2Terrain.yaml) — Look at the function `compute_reward_CaT` (beware that some reward terms have a weight of 0 and thus are deactivated, check weights in the config file); this implementation includes strong reward shaping, domain randomization, and training disturbances for robust sim‑to‑real, but it is written for legacy IsaacGym and the challenge is to re-implement it in Isaac Lab.
- **API References**:
    - [ArticulationData (`robot.data`)](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.ArticulationData) — Contains `root_pos_w`, `joint_pos`, `projected_gravity_b`, etc.
    - [ContactSensorData (`_contact_sensor.data`)](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html#isaaclab.sensors.ContactSensorData) — Contains `net_forces_w` (contact forces).
