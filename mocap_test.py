import mujoco
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import numpy as np
from gymnasium_robotics.utils.mujoco_utils import reset_mocap_welds, reset_mocap2body_xpos



def reset_mocap_welds(model, data):
    """Resets the mocap welds that we use for actuation."""
    if model.nmocap > 0 and model.eq_data is not None:
        for i in range(model.eq_data.shape[0]):
            if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                model.eq_data[i, :7] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    # mujoco.mj_forward(model, data)


def reset_mocap2body_xpos(model, data):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """

    if model.eq_type is None or model.eq_obj1id is None or model.eq_obj2id is None:
        return
    for eq_type, obj1_id, obj2_id in zip(
        model.eq_type, model.eq_obj1id, model.eq_obj2id
    ):
        if eq_type != mujoco.mjtEq.mjEQ_WELD:
            continue

        mocap_id = model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert mocap_id != -1
        data.mocap_pos[mocap_id][:] = data.xpos[body_idx]
        data.mocap_quat[mocap_id][:] = data.xquat[body_idx]
        

model = mujoco.MjModel.from_xml_path('metaworld/envs/assets_v2/sawyer_xyz/sawyer_assembly_peg.xml')
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

renderer = MujocoRenderer(model, data)
print(model.nmocap)

body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'mocap')
mocap_id = model.body_mocapid[body_id]


data.qpos[:7] = np.array([2.24, -0.5, -1.32,  1.48, 1.32,  1.13, -2.13], dtype=np.float32)
mujoco.mj_forward(model, data)

reset_mocap_welds(model, data)
reset_mocap2body_xpos(model,data)

mujoco.mj_forward(model, data)

# pos = data.mocap_pos[0].copy()
# quat = data.mocap_quat[0]
while True:
    
    # pos[0] -= 0.0001
    # pos[1] += 0.0001
    # data.mocap_pos[mocap_id] = pos
    
    mujoco.mj_step(model, data, nstep=50)
    if renderer.viewer is not None:
        renderer.viewer.add_marker(pos=data.mocap_pos.copy(), #position of the arrow\
                            size=np.array([0.01,0.01,0.01]), #size of the arrow
                            # mat=render_goal_orn, # orientation as a matrix
                            rgba=np.array([0.,230.,64.,1.]),#color of the arrow
                            type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            label=str('GOAL'))
    renderer.render(render_mode='human')
    
