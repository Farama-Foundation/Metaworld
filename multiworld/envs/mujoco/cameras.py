

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
