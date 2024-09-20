from omni.isaac.lab.assets import ArticulationCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg

from omni.isaac.groundcontrol_assets import GROUNDCONTROL_ASSETS_DATA_DIR

JOINT_NAMES = [
    'chassis_to_back_left_wheel',
    'chassis_to_back_right_wheel',
    'chassis_to_front_left_hinge',
    'chassis_to_front_right_hinge',
    'front_left_hinge_to_wheel',
    'front_right_hinge_to_wheel',
]

MITCAR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{GROUNDCONTROL_ASSETS_DATA_DIR}/Robots/UWRLL/mitcar.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True
        )
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),
        joint_pos={
            'front_left_hinge_to_wheel': 0.0,
            'front_right_hinge_to_wheel': 0.0,
            'chassis_to_back_left_wheel': 0.0,
            'chassis_to_back_right_wheel': 0.0,
            'chassis_to_front_left_hinge': 0.0,
            'chassis_to_front_right_hinge': 0.0,
        },
    ),
    actuators={
        f"{k}_actuator": ImplicitActuatorCfg(
            joint_names_expr=[k],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=10.0,
        ) for k in JOINT_NAMES
    },
)