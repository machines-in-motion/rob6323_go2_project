# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # Official terrain config
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


@configclass
class Rob6323Go2RoughEnvCfg(DirectRLEnvCfg):
    #env # - spaces definition
    decimation = 4
    episode_length_s = 20.0
    action_scale = 0.25
    action_space = 12
    


    num_height_scan_points = 187
    observation_space = 48 + 4 + num_height_scan_points  
    state_space = 0
    debug_vis = False
    
     # PD control gains
    Kp = 20.0
    Kd = 0.5
    torque_limits = 100.0
    
    #actuator friction params
    friction_fs_min = 0.0
    friction_fs_max = 2.5
    friction_mu_min = 0.0
    friction_mu_max = 0.3
    

    base_height_min = 0.10  # Increased from 0.05 to 0.10 to avoid immidiate death
    
    #reward scales
    lin_vel_reward_scale = 1.0  
    yaw_rate_reward_scale = 0.5
    action_rate_reward_scale = -0.01
    
    orient_reward_scale = -8.0  
    lin_vel_z_reward_scale = -0.05  
    dof_vel_reward_scale = -0.0001
    ang_vel_xy_reward_scale = -0.005  
    torque_reward_scale = -0.0001
    

    raibert_heuristic_reward_scale = -5.0  
    feet_clearance_reward_scale = -15.0  
    tracking_contacts_shaped_force_reward_scale = 2.0  
    
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
        terrain_generator=ROUGH_TERRAINS_CFG,  
        max_init_terrain_level=5,  
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    
    # robot(s)
    # Update robot_cfg
    
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # "base_legs" is an arbitrary key we use to group these actuators

    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0,  
        damping=0.0,
    )
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,   # 2048,4096,1024
        env_spacing=4.0,
        replicate_physics=True
    )
    
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True
    )
    
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),  
        attach_yaw_only=True,  
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1,  
            size=[1.6, 1.0],  
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    
    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    
    def __post_init__(self):

        if "boxes" in self.terrain.terrain_generator.sub_terrains:
            self.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        
        if "random_rough" in self.terrain.terrain_generator.sub_terrains:
            self.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
            self.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01
        
        self.terrain.terrain_generator.curriculum = False

