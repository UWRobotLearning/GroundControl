from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils

# ===== NOTE:IsaacLab imports === ^^^ 
# ===== GroundControl imports === VVV
from groundcontrol_assets import GROUNDCONTROL_ASSETS_DATA_DIR

@configclass
class BarrelTestScenarioCfg:
    scene: InteractiveSceneCfg = InteractiveSceneCfg() 


    def __post_init__(self) -> None:
        self.scene.barrel = AssetBaseCfg(
            prim_path="/World/Barrel",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(8,-8,3)
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{GROUNDCONTROL_ASSETS_DATA_DIR}/Props/o3dyn_pallet.usd",
                scale=(1.0,1.0,1.0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.,
                    max_linear_velocity=1000.,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg()
            ),
        )
