# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

# Terrain changes
#from https://github.com/isaac-sim/IsaacLab/blob/2ed331acfcbb1b96c47b190564476511836c3754/source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c/anymal_c_env_cfg.py#L114
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG 
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.terrains.height_field import hf_terrains_cfg as terrain_gen

#PART 4
from isaaclab.actuators import ImplicitActuatorCfg

@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0
    # - spaces definition
    action_scale = 0.5
    action_space = 12
    observation_space = 235 + 4
    state_space = 0
    debug_vis = True
    
    # PD control gains
    Kp = 20.0  # Proportional gain
    Kd = 0.5   # Derivative gain
    torque_limits = 100.0  # Max torque
    
    #base height 
    base_height_min = 0.15  # the bot is about 0.30m tall. From tutorials, giving a min of 0.15m . 
    
    #tunable reward scale
    #part1
    lin_vel_reward_scale = 3.0
    yaw_rate_reward_scale = 1.5
    action_rate_reward_scale = -0.01
    raibert_heuristic_reward_scale = -1.0
    orient_reward_scale = -5.0
    lin_vel_z_reward_scale = -0.02
    dof_vel_reward_scale = -0.0001
    ang_vel_xy_reward_scale = -0.001
    feet_clearance_reward_scale = -10.0
    tracking_contacts_shaped_force_reward_scale = 4.0
    

    # simulation
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
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=5.0,  # Flat spawn area
            num_rows=16,
            num_cols=16,
            horizontal_scale=0.1,
            vertical_scale=0.002,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=False,  
            sub_terrains={
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=1.0,  
                    noise_range=(0.00, 0.01),
                    noise_step=0.005,
                    border_width=0.25,
                ),
            },
        ),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # robot(s)
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # scene
    #reducing num_envs to reduce processing time. 
    
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=4.0, replicate_physics=True)
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # reward scales

    
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
