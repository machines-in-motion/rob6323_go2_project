# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations


import gymnasium as gym
import math
import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
import isaaclab.utils.math as math_utils

from .rob6323_go2_env_cfg import Rob6323Go2EnvCfg


class Rob6323Go2Env(DirectRLEnv):
    cfg: Rob6323Go2EnvCfg

    def __init__(self, cfg: Rob6323Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        action_dim = gym.spaces.flatdim(self.single_action_space)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, action_dim, device=self.device)

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging: reward terms + diagnostics
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "rew_action_rate",                  # Part 1
                "raibert_heuristic",                # Part 4
                "orient",
                "lin_vel_z",
                "dof_vel",
                "ang_vel_xy",
                "feet_clearance",
                "tracking_contacts_shaped_force",
                # Diagnostics for plotting shortcomings (NOT part of reward)
                "diag_slip",
                "diag_toe_drag",
                "diag_foot_timing",
                "diag_jerk",
                "diag_posture",
                "diag_knee_collision",
            ]
        }

        # Part 1: variables needed for action rate penalization
        # Shape: (num_envs, action_dim, history_length)
        self.last_actions = torch.zeros(
            self.num_envs, action_dim, 3, dtype=torch.float, device=self.device, requires_grad=False
        )

        # Part 2: PD control parameters
        self.Kp = torch.full((self.num_envs, action_dim), cfg.Kp, device=self.device)
        self.Kd = torch.full((self.num_envs, action_dim), cfg.Kd, device=self.device)
        self.torque_limits = cfg.torque_limits
        self.desired_joint_pos = torch.zeros(self.num_envs, action_dim, device=self.device)

        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")

        # Thigh links used as 'knee' collision diagnostics
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*thigh")

        # Feet indices (robot state) and contact-sensor indices (forces) – Parts 4 & 6
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self._feet_ids: list[int] = []
        for name in foot_names:
            ids, _ = self.robot.find_bodies(name)
            if len(ids) == 0:
                raise RuntimeError(f"Could not find robot body named {name}")
            self._feet_ids.append(ids[0])

        self._feet_ids_sensor: list[int] = []
        for name in foot_names:
            ids, _ = self._contact_sensor.find_bodies(name)
            if len(ids) == 0:
                raise RuntimeError(f"Could not find contact sensor body named {name}")
            self._feet_ids_sensor.append(ids[0])

        # Part 4: Raibert / gait state
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.desired_contact_states = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.foot_indices = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        # add ground / terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        # Part 2: compute desired joint positions from policy actions
        self.desired_joint_pos = self.cfg.action_scale * self._actions + self.robot.data.default_joint_pos

    def _apply_action(self) -> None:
        # Part 2: manual PD control in torque space (matches tutorial: torch.clip)
        torques = torch.clip(
            self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos)
            - self.Kd * self.robot.data.joint_vel,
            -self.torque_limits,
            self.torque_limits,
        )
        # Apply torques to the robot
        self.robot.set_joint_effort_target(torques)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        # Part 4: update gait state for clock inputs / desired contacts
        self._step_contact_targets()

        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self.robot.data.root_lin_vel_b,
                    self.robot.data.root_ang_vel_b,
                    self.robot.data.projected_gravity_b,
                    self._commands,
                    self.robot.data.joint_pos - self.robot.data.default_joint_pos,
                    self.robot.data.joint_vel,
                    self._actions,
                    self.clock_inputs,  # +4 clock inputs
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

        # Part 1: action rate penalization (1st + 2nd derivative)
        rew_action_rate = torch.sum(torch.square(self._actions - self.last_actions[:, :, 0]), dim=1) * (
            self.cfg.action_scale ** 2
        )
        rew_action_rate += torch.sum(
            torch.square(self._actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]),
            dim=1,
        ) * (self.cfg.action_scale ** 2)

        # Update the prev action hist (roll buffer and insert new action)
        self.last_actions = torch.roll(self.last_actions, 1, 2)
        self.last_actions[:, :, 0] = self._actions

        # Part 4: Raibert heuristic
        self._step_contact_targets()
        rew_raibert_heuristic = self._reward_raibert_heuristic()

        # Part 5: orientation, vertical velocity, joint velocity, ang vel XY
        proj_g = self.robot.data.projected_gravity_b
        rew_orient = torch.sum(proj_g[:, :2] ** 2, dim=1)

        root_lin_vel_b = self.robot.data.root_lin_vel_b
        rew_lin_vel_z = root_lin_vel_b[:, 2] ** 2

        joint_vel = self.robot.data.joint_vel
        rew_dof_vel = torch.sum(joint_vel ** 2, dim=1)

        root_ang_vel_b = self.robot.data.root_ang_vel_b
        rew_ang_vel_xy = torch.sum(root_ang_vel_b[:, :2] ** 2, dim=1)

        # Part 6: foot clearance and contact shaping
        rew_feet_clearance, rew_contacts_shaped = self._reward_feet_and_contacts()

        # ---------- Diagnostics (NOT part of reward) ----------
        diag_slip, diag_toe_drag = self._metric_slip_and_toedrag()
        diag_foot_timing = self._metric_foot_timing()
        diag_jerk = self._metric_jerk()
        diag_posture = self._metric_posture()
        diag_knee_collision = self._metric_knee_collision()

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale,
            "rew_action_rate": rew_action_rate * self.cfg.action_rate_reward_scale,
            "raibert_heuristic": rew_raibert_heuristic * self.cfg.raibert_heuristic_reward_scale,
            "orient": rew_orient * self.cfg.orient_reward_scale,
            "lin_vel_z": rew_lin_vel_z * self.cfg.lin_vel_z_reward_scale,
            "dof_vel": rew_dof_vel * self.cfg.dof_vel_reward_scale,
            "ang_vel_xy": rew_ang_vel_xy * self.cfg.ang_vel_xy_reward_scale,
            "feet_clearance": rew_feet_clearance * self.cfg.feet_clearance_reward_scale,
            "tracking_contacts_shaped_force": (
                rew_contacts_shaped * self.cfg.tracking_contacts_shaped_force_reward_scale
            ),
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging – reward terms
        for key, value in rewards.items():
            self._episode_sums[key] += value

        # Logging – diagnostic metrics (unscaled)
        self._episode_sums["diag_slip"] += diag_slip
        self._episode_sums["diag_toe_drag"] += diag_toe_drag
        self._episode_sums["diag_foot_timing"] += diag_foot_timing
        self._episode_sums["diag_jerk"] += diag_jerk
        self._episode_sums["diag_posture"] += diag_posture
        self._episode_sums["diag_knee_collision"] += diag_knee_collision

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        cstr_termination_contacts = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0,
            dim=1,
        )
        cstr_upsidedown = self.robot.data.projected_gravity_b[:, 2] > 0

        # Part 3: base height minimum
        base_height = self.robot.data.root_pos_w[:, 2]
        cstr_base_height_min = base_height < self.cfg.base_height_min

        died = cstr_termination_contacts | cstr_upsidedown | cstr_base_height_min
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0

        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)

        # Reset Raibert quantities
        self.gait_indices[env_ids] = 0.0
        self.clock_inputs[env_ids] = 0.0
        self.desired_contact_states[env_ids] = 0.0
        self.foot_indices[env_ids] = 0.0

        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            # diagnostics go under Episode_Diag/, rewards under Episode_Reward/
            prefix = "Episode_Reward/"
            if key.startswith("diag_"):
                prefix = "Episode_Diag/"
            extras[prefix + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._commands[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

    # ------------------------------------------------------------------
    # Helpers for Raibert / foot rewards
    # ------------------------------------------------------------------
    @property
    def foot_positions_w(self) -> torch.Tensor:
        """Returns the feet positions in the world frame. Shape: (num_envs, 4, 3)."""
        return self.robot.data.body_pos_w[:, self._feet_ids]

    def _step_contact_targets(self) -> None:
        """Update gait phase, clock inputs and desired contact states."""
        frequencies = 3.0
        phases = 0.5
        offsets = 0.0
        bounds = 0.0

        durations = 0.5 * torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)

        # advance gait phase
        self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * frequencies, 1.0)

        foot_indices_list = [
            self.gait_indices + phases + offsets + bounds,
            self.gait_indices + offsets,
            self.gait_indices + bounds,
            self.gait_indices + phases,
        ]

        # store per-foot indices in [0, 1)
        self.foot_indices = torch.remainder(torch.stack(foot_indices_list, dim=1), 1.0)

        # stance / swing mapping
        for fi in foot_indices_list:
            stance_idxs = torch.remainder(fi, 1.0) < durations
            swing_idxs = torch.remainder(fi, 1.0) > durations

            fi[stance_idxs] = torch.remainder(fi[stance_idxs], 1.0) * (0.5 / durations[stance_idxs])
            fi[swing_idxs] = 0.5 + (
                torch.remainder(fi[swing_idxs], 1.0) - durations[swing_idxs]
            ) * (0.5 / (1.0 - durations[swing_idxs]))

        # clock inputs
        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices_list[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices_list[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices_list[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices_list[3])

        # smooth desired contact states using a normal CDF
        kappa = 0.07
        normal_dist = torch.distributions.normal.Normal(0.0, kappa)

        def smooth(fi: torch.Tensor) -> torch.Tensor:
            phase = torch.remainder(fi, 1.0)
            return normal_dist.cdf(phase) * (1.0 - normal_dist.cdf(phase - 0.5)) + normal_dist.cdf(phase - 1.0) * (
                1.0 - normal_dist.cdf(phase - 0.5 - 1.0)
            )

        self.desired_contact_states[:, 0] = smooth(foot_indices_list[0])
        self.desired_contact_states[:, 1] = smooth(foot_indices_list[1])
        self.desired_contact_states[:, 2] = smooth(foot_indices_list[2])
        self.desired_contact_states[:, 3] = smooth(foot_indices_list[3])

    def _reward_raibert_heuristic(self) -> torch.Tensor:
        """Raibert foothold error (penalty)."""
        cur_footsteps_translated = self.foot_positions_w - self.robot.data.root_pos_w.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros_like(cur_footsteps_translated)

        base_quat_w = self.robot.data.root_quat_w
        base_quat_conj = math_utils.quat_conjugate(base_quat_w)

        for i in range(4):
            footsteps_in_body_frame[:, i, :] = math_utils.quat_apply_yaw(
                base_quat_conj, cur_footsteps_translated[:, i, :]
            )

        # nominal positions: [FR, FL, RR, RL]
        desired_stance_width = 0.25
        desired_ys_nom = torch.tensor(
            [desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2],
            device=self.device,
        ).unsqueeze(0)

        desired_stance_length = 0.45
        desired_xs_nom = torch.tensor(
            [desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2],
            device=self.device,
        ).unsqueeze(0)

        # raibert offsets
        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = torch.tensor([3.0], device=self.device)
        x_vel_des = self._commands[:, 0:1]
        yaw_vel_des = self._commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2.0

        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1.0
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat(
            (desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)),
            dim=2,
        )

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])
        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))
        return reward

    def _reward_feet_and_contacts(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Feet clearance and contact-shaping penalties."""
        # foot heights relative to base
        foot_pos = self.foot_positions_w
        base_height = self.robot.data.root_pos_w[:, 2].unsqueeze(1)
        rel_foot_height = foot_pos[:, :, 2] - (base_height - 0.05)

        stance_height = 0.03
        swing_height = 0.15
        desired_height = stance_height + (swing_height - stance_height) * (1.0 - self.desired_contact_states)

        feet_clearance_error = (rel_foot_height - desired_height) ** 2
        rew_feet_clearance = torch.sum(feet_clearance_error, dim=1)

        # contact shaping
        contact_forces = self._contact_sensor.data.net_forces_w
        feet_forces = contact_forces[:, self._feet_ids_sensor, :]
        feet_force_mag = torch.norm(feet_forces, dim=-1)

        contact_binary = (feet_force_mag > 1.0).float()
        contacts_error = (contact_binary - self.desired_contact_states) ** 2
        rew_contacts_shaped = torch.sum(contacts_error, dim=1)

        return rew_feet_clearance, rew_contacts_shaped

    # ------------------------------------------------------------------
    # Diagnostic helpers (shortcomings for plotting only)
    # ------------------------------------------------------------------
    def _metric_slip_and_toedrag(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Diagnostics: slip in stance & toe-drag in swing (not used in reward)."""
        # Foot positions and velocities
        foot_pos = self.foot_positions_w
        foot_vel = self.robot.data.body_lin_vel_w[:, self._feet_ids]

        base_height = self.robot.data.root_pos_w[:, 2].unsqueeze(1)

        # swing / stance masks from Raibert desired contacts
        stance_mask = self.desired_contact_states           # ~1 when we want contact
        swing_mask = 1.0 - stance_mask

        # --- toe drag: low height or contact during swing
        rel_height = foot_pos[:, :, 2] - (base_height - 0.02)  # height above nominal ground
        low_height = swing_mask * torch.relu(0.03 - rel_height)  # < 3 cm during swing

        contact_forces = self._contact_sensor.data.net_forces_w
        feet_forces = contact_forces[:, self._feet_ids_sensor, :]
        feet_force_mag = torch.norm(feet_forces, dim=-1)
        swing_contact = swing_mask * (feet_force_mag > 1.0).float()

        toe_drag = torch.sum(low_height + swing_contact, dim=1)

        # --- slip: horizontal velocity during stance
        horiz_vel = torch.norm(foot_vel[:, :, :2], dim=-1)
        slip = torch.sum(stance_mask * torch.relu(horiz_vel - 0.3), dim=1)

        return slip, toe_drag

    def _metric_foot_timing(self) -> torch.Tensor:
        """Diagnostics: mismatch between desired and actual contact pattern (sloppy timing)."""
        contact_forces = self._contact_sensor.data.net_forces_w
        feet_forces = contact_forces[:, self._feet_ids_sensor, :]
        feet_force_mag = torch.norm(feet_forces, dim=-1)
        actual_contact = (feet_force_mag > 1.0).float()
        timing_error = (actual_contact - self.desired_contact_states) ** 2
        return torch.sum(timing_error, dim=1)

    def _metric_jerk(self) -> torch.Tensor:
        """Diagnostics: jerkiness based on first and second action differences.

        Uses the action history buffer after it has been updated in `_get_rewards`.
        At this point, for each env and joint:
            last_actions[:, :, 0] = a_t
            last_actions[:, :, 1] = a_{t-1}
            last_actions[:, :, 2] = a_{t-2}
        """
        a_t = self._actions
        a_tm1 = self.last_actions[:, :, 1]
        a_tm2 = self.last_actions[:, :, 2]

        first_diff = a_t - a_tm1
        second_diff = a_t - 2.0 * a_tm1 + a_tm2

        return torch.sum(first_diff ** 2 + second_diff ** 2, dim=1)

    def _metric_posture(self) -> torch.Tensor:
        """Diagnostics: instability from tilt and angular velocity."""
        proj_g = self.robot.data.projected_gravity_b
        root_ang_vel_b = self.robot.data.root_ang_vel_b
        tilt = torch.sum(proj_g[:, :2] ** 2, dim=1)          # roll/pitch tilt
        ang_xy = torch.sum(root_ang_vel_b[:, :2] ** 2, dim=1)
        return tilt + ang_xy

    def _metric_knee_collision(self) -> torch.Tensor:
        """Diagnostics: continuous knee collision measure based on thigh contact forces."""
        # If we didn't find any thigh bodies, just return zeros
        if len(self._undesired_contact_body_ids) == 0:
            return torch.zeros(self.num_envs, device=self.device)

        contact_forces = self._contact_sensor.data.net_forces_w  # (num_envs, num_bodies, 3)

        # Select thigh / knee bodies
        knees_forces = contact_forces[:, self._undesired_contact_body_ids, :]
        knees_force_mag = torch.norm(knees_forces, dim=-1)

        # Continuous collision score: sum of squared force magnitudes per env
        knee_collision_score = torch.sum(knees_force_mag ** 2, dim=1)

        return knee_collision_score