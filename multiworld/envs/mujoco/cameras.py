

def create_sawyer_camera_init(
        lookat=(0, 0.85, 0.3),
        distance=0.3,
        elevation=-35,
        azimuth=270,
        trackbodyid=-1,
):
    def init(camera):
        camera.lookat[0] = lookat[0]
        camera.lookat[1] = lookat[1]
        camera.lookat[2] = lookat[2]
        camera.distance = distance
        camera.elevation = elevation
        camera.azimuth = azimuth
        camera.trackbodyid = trackbodyid

    return init


def init_sawyer_camera_v1(camera):
    """
    Do not get so close that the arm crossed the camera plane
    """
    camera.lookat[0] = 0
    camera.lookat[1] = 1
    camera.lookat[2] = 0.3
    camera.distance = 0.35
    camera.elevation = -35
    camera.azimuth = 270
    camera.trackbodyid = -1


def init_sawyer_camera_v2(camera):
    """
    Top down basically. Sees through the arm.
    """
    camera.lookat[0] = 0
    camera.lookat[1] = 0.8
    camera.lookat[2] = 0.3
    camera.distance = 0.3
    camera.elevation = -65
    camera.azimuth = 270
    camera.trackbodyid = -1


def init_sawyer_camera_v3(camera):
    """
    Top down basically. Sees through the arm.
    """
    camera.lookat[0] = 0
    camera.lookat[1] = 0.85
    camera.lookat[2] = 0.3
    camera.distance = 0.3
    camera.elevation = -35
    camera.azimuth = 270
    camera.trackbodyid = -1

def sawyer_pick_and_place_camera(camera):
    camera.lookat[0] = 0.0
    camera.lookat[1] = .67
    camera.lookat[2] = .1
    camera.distance = .7
    camera.elevation = 0
    camera.azimuth = 180
    camera.trackbodyid = 0

def init_sawyer_camera_v4(camera):
    """
    This is the same camera used in old experiments (circa 6/7/2018)
    """
    camera.lookat[0] = 0
    camera.lookat[1] = 0.85
    camera.lookat[2] = 0.3
    camera.distance = 0.3
    camera.elevation = -35
    camera.azimuth = 270
    camera.trackbodyid = -1

