
from metaworld.envs.mujoco.dynamic_mjc.model_builder import MJCModel
import numpy as np
import os

np.random.seed(0)

COLORS = ["0.5 0.5 0.5 1", 
          "0.0 0.0 0.0 1",
          "0.0 0.5 0.0 1",
          "0.5 0.0 0.5 1",
          "0.0 0.0 0.5 1"]

def rope(num_beads = 5, 
    init_pos=[0.0, 0.0, 0.0],
    texture=False,
    ):
    mjcmodel = MJCModel('sawyer', include_config=True)
    visual = mjcmodel.root.visual()
    visual.headlight(ambient="0.5 0.5 0.5")
    mjcmodel.root.compiler(inertiafromgeom="auto",
                             angle="radian",
                             coordinate="local",
                             eulerseq="XYZ",
                             meshdir=os.path.dirname(os.path.realpath(__file__)) + "/../../assets/meshes",
                             texturedir=os.path.dirname(os.path.realpath(__file__)) + "/../../assets/textures")
    mjcmodel.root.size(njmax="6000", nconmax="6000")
    mjcmodel.root.option(timestep='0.0025', iterations="50", tolerance="1e-10", solver="Newton", jacobian="dense", cone="elliptic")
    default = mjcmodel.root.default()
    default.joint(limited="true",
                   damping="1",
                   stiffness="0",
                   armature=".1",
                   user="0")
    default.geom(solref="0.02 1",
                solimp="1 1 0")
    default.motor(ctrllimited="true", ctrlrange="-1 1")
    default.position(ctrllimited="true")
    equality = mjcmodel.root.equality()
    equality.weld(body1="mocap", body2="hand", solref="0.02 1", end_with_name=True)

    # mjcmodel.root.include(file=os.path.dirname(os.path.realpath(__file__)) + "/../../assets/sawyer_xyz/shared_config.xml", end_with_name=True)

    # Make base
    worldbody = mjcmodel.root.worldbody()
    # include file only allows relative path
    worldbody.include(file=".." + os.path.dirname(os.path.realpath(__file__)) + "/../../assets/sawyer_xyz/sawyer_xyz_base.xml", end_with_name=True)
    worldbody.camera(name="robotview_zoomed", pos="0.0 1.3 0.5", euler="-0.78 0.0 3.14159")
    worldbody.camera(name="leftcam", euler="0 1.57 0", pos="1.0 0.75 0.2")
    
    # displacement = [0.0, 0.07, 0.0]
    displacement = [0.0, 0.05, 0.0]
    
    #bead_pos = [0.0, 0.0, 0.015] #for cuboidal beads
    site_pos = [0.0, 0.0, 0.0] #for spherical beads
    # tendon_range = [0.0, 0.07]
    tendon_range = [0.0, 0.05]

    # color = np.random.choice(COLORS)
    color = "0.5 0.5 0.5 1"

    beads = []
    for i in range(num_beads):
        new_pos = list(np.asarray(init_pos) + i*(np.asarray(displacement)))
        beads.append(worldbody.body(name="bead_{}".format(i), pos=new_pos))
        beads[i].joint(type="free",limited='false')
        if texture:
            # beads[i].geom(type="sphere", size="0.03", rgba=color, 
            #       mass="0.03", contype="7", conaffinity="7", friction="1.0 0.10 0.002",
            #       condim="6", solimp="0.99 0.99 0.01", solref="0.01 1", material="bead_material")
            beads[i].geom(type="sphere", size="0.02", rgba=color, 
                  mass="0.03", contype="7", conaffinity="7", friction="1.0 0.10 0.002",
                  condim="6", solimp="0.99 0.99 0.01", solref="0.01 1", material="bead_material")
            # beads[i].geom(type="sphere", size="0.03", rgba="0.5 0.5 0.5 1", 
            #       mass="0.03", contype="7", conaffinity="7", friction="1.0 0.10 0.002",
            #       condim="6", solimp="0.99 0.99 0.01", solref="0.01 1", material="bead_material")
        else:
            # beads[i].geom(type="sphere", size="0.03", rgba="0.8 0.2 0.2 1", 
            #       mass="0.03", contype="7", conaffinity="7", friction="1.0 0.10 0.002",
            #       condim="6", solimp="0.99 0.99 0.01", solref="0.01 1")
            beads[i].geom(type="sphere", size="0.02", rgba="0.8 0.2 0.2 1", 
                  mass="0.03", contype="7", conaffinity="7", friction="1.0 0.10 0.002",
                  condim="6", solimp="0.99 0.99 0.01", solref="0.01 1")

        beads[i].site(name="site_{}".format(i), pos=site_pos, type="sphere", size="0.01")

        # beads[i].geom(type="box", size="0.015 0.03 0.015", rgba="0.8 0.2 0.2 1", 
        #           mass="0.03", contype="7", conaffinity="7", friction="1.0 0.10 0.002",
        #           condim="6", solimp="0.99 0x.99 0.01", solref="0.01 1")
        # beads[i].site(name="site_{}".format(i), pos=site_pos, type="sphere", size="0.01")

    # container = worldbody.body(name="container", pos=[0,0,-0.05])
    # border_front = container.body(name="border_front", pos="0 -.5  0")
    # border_front.geom(type="box", size=".5 .01 .1", rgba="0 .1 .9 .3")
    # border_rear = container.body(name="border_rear", pos="0 .5  0")
    # border_rear.geom(type="box", size=".5 .01 .1", rgba="0 .1 .9 .3")
    # border_right = container.body(name="border_right", pos=".5 0. 0")
    # border_right.geom(type="box", size=".01  .5 .1", rgba="0 .1 .9 .3")
    # border_left = container.body(name="border_left", pos="-.5 0. 0")
    # border_left.geom(type="box", size=".01  .5 .1", rgba="0 .1 .9 .3")
    # table = container.body(name="table", pos="0 0 -.01")
    # if texture:
    #     table.geom(type="box", size=".5  .5 .01", rgba=".5 .5 .5 1", contype="7", conaffinity="7", material="table_material")
    # else:
    #     table.geom(type="box", size=".5  .5 .01", rgba="0 .9 0 1", contype="7", conaffinity="7")
        
    # light = worldbody.body(name="light", pos=[0,0,1])
    # light.light(name="light0", mode="fixed", directional="false", active="true", castshadow="true")

    tendons = mjcmodel.root.tendon()
    tendon_list = []
    for i in range(num_beads-1):
        tendon_list.append(tendons.spatial(limited="true", range=tendon_range, width="0.005"))
        tendon_list[i].site(site="site_{}".format(i))
        tendon_list[i].site(site="site_{}".format(i+1))

    actuator = mjcmodel.root.actuator()
    actuator.position(ctrllimited="true", ctrlrange="-1 1", joint="r_close", kp="400",  user="1")
    actuator.position(ctrllimited="true", ctrlrange="-1 1", joint="l_close", kp="400",  user="1")

    asset = mjcmodel.root.asset()
    asset.texture(file='wood.png', name='table_texture')
    asset.material(name='table_material', rgba='1 1 1 1', shininess='0.3', specular='1', texture='table_texture')
    asset.texture(file='marble.png', name='bead_texture')
    asset.material(name='bead_material', rgba='1 1 1 1', shininess='0.3', specular='1', texture='bead_texture')
    asset.mesh(name="pedestal", file="sawyer/pedestal.stl")
    asset.mesh(name="base", file="sawyer/base.stl")
    asset.mesh(name="l0", file="sawyer/l0.stl")
    asset.mesh(name="head", file="sawyer/head.stl")
    asset.mesh(name="l1", file="sawyer/l1.stl")
    asset.mesh(name="l2", file="sawyer/l2.stl")
    asset.mesh(name="l3", file="sawyer/l3.stl")
    asset.mesh(name="l4", file="sawyer/l4.stl")
    asset.mesh(name="l5", file="sawyer/l5.stl")
    asset.mesh(name="l6", file="sawyer/l6.stl")
    asset.mesh(name="eGripperBase", file="sawyer/eGripperBase.stl")

    return mjcmodel

# <asset>
# <texture file="describable/dtd/images/waffled/waffled_0169.png" name="vase_texture" />
# <material name="vase_material" rgba="1 1 1 1" shininess="0.3" specular="1" texture="vase_texture" />
# </asset>