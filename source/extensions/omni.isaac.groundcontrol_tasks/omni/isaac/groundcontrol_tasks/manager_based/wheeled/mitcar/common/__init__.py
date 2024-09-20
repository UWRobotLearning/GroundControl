# This is the observation ordering (DO NOT CHANGE)
# TODO: allow arbitrary re-ordering using this list
JOINT_NAMES = [
    'chassis_to_back_left_wheel',
    'chassis_to_back_right_wheel',
    'chassis_to_front_left_hinge',
    'chassis_to_front_right_hinge',
    'front_left_hinge_to_wheel',
    'front_right_hinge_to_wheel',
]

from .actions import ActionsCfg
from .observations import ObservationsCfg
from .events import EventCfg
from .commands import NoCommandsCfg
