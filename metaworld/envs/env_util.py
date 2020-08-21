import os

ENV_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets')
ENV_ASSET_DIR_UPDATE = os.path.join(os.path.dirname(__file__), 'assets_updated')

def get_asset_full_path(file_name, updated=False):
    asset_dir = ENV_ASSET_DIR_UPDATE if updated else ENV_ASSET_DIR
    return os.path.join(asset_dir, file_name)
