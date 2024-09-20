import os

# DEFAULT_RUN_DIRNAME = "graceful-frost-97_2024-08-12_19-29-14"

WHEELED_LAB_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
WHEELED_LAB_RESOURCES_DIR = os.path.join(WHEELED_LAB_ROOT_DIR, 'resources')
WHEELED_LAB_LOGS_DIR = os.path.join(WHEELED_LAB_ROOT_DIR, 'logs')
WHEELED_LAB_CORE_DATA_DIR = os.path.join(WHEELED_LAB_ROOT_DIR, 'core_data')

# WHEELED_LAB_DEFAULT_RUN_DIR = os.path.join(WHEELED_LAB_CORE_DATA_DIR, DEFAULT_RUN_DIRNAME)