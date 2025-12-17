# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import (
    BLUE_ARROW_X_MARKER_CFG,
    FRAME_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
)
from isaaclab.actuators import ImplicitActuatorCfg


@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # ------------------------------------------------------------------
    # Environment & spaces
    # ------------------------------------------------------------------
    decimation = 4
    episode_length_s = 20.0

    action_scale = 0.25
    action_space = 12
    # 48 original obs + 4 clock inputs for gait phase signals
    observation_space = 48 + 4
    state_space = 0
    debug_vis = True

    # PD controller gains (used in manual torque control)
    Kp = 20.0
    Kd = 0.5
    torque_limits = 100.0

    # Early termination based on base height
    base_height_min = 0.20

    # ------------------------------------------------------------------
    # Simulation & terrain
    # ------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # ------------------------------------------------------------------
    # Robot
    # ------------------------------------------------------------------
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # Disable built‑in PD and group leg joints under a custom actuator
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_joint",
            ".*_thigh_joint",
            ".*_calf_joint",
        ],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0,  # disable implicit P‑gain
        damping=0.0,    # disable implicit D‑gain
    )

    # ------------------------------------------------------------------
    # Scene & sensors
    # ------------------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = (
        GREEN_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/Command/velocity_goal"
        )
    )
    """Configuration for the goal velocity visualization marker."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = (
        BLUE_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/Command/velocity_current"
        )
    )
    """Configuration for the current velocity visualization marker."""

    # shrink marker size a bit
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # ------------------------------------------------------------------
    # Reward scales
    # ------------------------------------------------------------------
    # Command tracking
    lin_vel_reward_scale = 1.0            # vx, vy tracking
    yaw_rate_reward_scale = 0.5           # yaw‑rate tracking

    # Action regularization / smoothness
    action_rate_reward_scale = -0.01      # penalize jerk in actions
    torque_l2_reward_scale = -1e-4        # tiny penalty on ||tau||^2

    # Gait quality & feet
    raibert_heuristic_reward_scale = -10.0
    feet_clearance_reward_scale = -30.0
    tracking_contacts_shaped_force_reward_scale = 4.0

    # Base stability, attitude and height shaping
    orient_reward_scale = -5.0
    lin_vel_z_reward_scale = -0.02
    dof_vel_reward_scale = -0.0001
    ang_vel_xy_reward_scale = -0.001

    # Foot slip penalty (horizontal slip while in contact)
    feet_slip_reward_scale = -0.5

    # Explicit knee/hip collision penalty
    knee_collision_reward_scale = -0.02
