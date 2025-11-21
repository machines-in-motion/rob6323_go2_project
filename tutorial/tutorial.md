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
# In Rob6323Go2EnvCfg

# reward scales
action_rate_reward_scale = -0.1
```

### 1.2 Update `__init__`
We need a buffer to store the last few actions. We'll store a history of length 3 (current + 2 previous). Also, update the logging keys to track this new reward.

```python
# In Rob6323Go2Env.__init__

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
self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")

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
def _step_contact_targets(self):
    frequencies = 3.0
    # Advance gait phase
    self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * frequencies, 1.0)
    
    # Calculate clock inputs (sin/cos) for observation
    # ... (See full implementation in reference code for von mises logic)
    
    # Store clock inputs for observation
    self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
    # ... (repeat for other feet)
```

### 4.5 Implement Raibert Reward
We calculate the error between where the foot IS and where the Raibert Heuristic says it SHOULD be.

```python
# In Rob6323Go2Env (add new method)

def _reward_raibert_heuristic(self):
    # Get current foot positions relative to base
    cur_footsteps_translated = self.foot_positions_w - self.robot.data.root_pos_w.unsqueeze(1)
    
    # Transform to body frame
    footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply_yaw(
            math_utils.quat_conjugate(self.robot.data.root_quat_w),
            cur_footsteps_translated[:, i, :]
        )

    # Calculate Desired Footsteps (Nominal + Offset based on velocity)
    # offset = phase * velocity * (0.5 / frequency)
    # ... (See full calculation in reference code)
    
    # Calculate Error
    err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

    # Return squared error (to be minimized)
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

By following these steps, you have transformed a simple environment into a research-grade locomotion setup capable of learning robust walking gaits!
