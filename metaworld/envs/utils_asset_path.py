import os

ENV_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets')


def full_visual_path_for(file_name):
    return os.path.join(ENV_ASSET_DIR, file_name)
