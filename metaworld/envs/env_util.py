import os

ENV_ASSET_DIR_V1 = os.path.join(os.path.dirname(__file__), 'assets_v1')
ENV_ASSET_DIR_V2 = os.path.join(os.path.dirname(__file__), 'assets_v2')


def get_asset_full_path(file_name, v2=False):
    asset_dir = ENV_ASSET_DIR_V2 if v2 else ENV_ASSET_DIR_V1
    return os.path.join(asset_dir, file_name)
