import numpy as np
import xml.etree.cElementTree as ET
import xml.dom.minidom as minidom
import imp
import glob
import os
import random

import numpy as np
import stl
from stl import mesh
import multiworld

def find_mins_maxs(obj):
    minx = maxx = miny = maxy = minz = maxz = None
    for p in obj.points:
        # p contains (x, y, z)
        if minx is None:
            minx = p[stl.Dimension.X]
            maxx = p[stl.Dimension.X]
            miny = p[stl.Dimension.Y]
            maxy = p[stl.Dimension.Y]
            minz = p[stl.Dimension.Z]
            maxz = p[stl.Dimension.Z]
        else:
            maxx = max(p[stl.Dimension.X], maxx)
            minx = min(p[stl.Dimension.X], minx)
            maxy = max(p[stl.Dimension.Y], maxy)
            miny = min(p[stl.Dimension.Y], miny)
            maxz = max(p[stl.Dimension.Z], maxz)
            minz = min(p[stl.Dimension.Z], minz)
    return minx, maxx, miny, maxy, minz, maxz


def file_len(fname):
    i = 0
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


ASSET_BASE_DIR = '/'.join(os.path.abspath(multiworld.__file__).split('/')[:-2]) +  '/multiworld/envs/assets/multiobject_models/'


def create_object_xml(filename, num_objects, object_mass, friction_params, object_meshes,
                      finger_sensors, maxlen, minlen, load_dict_list, obj_classname = None,
                      block_height = 0.03, block_width = 0.03, cylinder_radius = 0.04):
    """
    :param hyperparams:
    :param load_dict_list: if not none load configuration, instead of sampling
    :return: if not loading, save dictionary with static object properties
    """
    xmldir = '/'.join(str.split(filename, '/')[:-1])
    root = ET.Element("top")

    save_dict_list = []

    if finger_sensors:
        sensor_frame = ET.SubElement(root, "sensor")
        ET.SubElement(sensor_frame, "touch", name = "finger1_sensor", site = "finger1_surf")
        ET.SubElement(sensor_frame, "touch", name = "finger2_sensor", site = "finger2_surf")
    else:
        sensor_frame = None
    f_sliding, f_torsion, f_rolling = friction_params
    world_body = ET.SubElement(root, "worldbody")

    loaded_meshes = {}

    for i in range(num_objects):
        if load_dict_list == None:
            dict = {}

            color1 = dict['color1'] = np.random.uniform(0.3, 1., 3)
            color2 = dict['color2'] = np.random.uniform(0.3, 1., 3)


            l1 = dict['l1'] =np.random.uniform(minlen, maxlen)
            l2 = dict['l2'] =np.random.uniform(minlen, maxlen)

            pos2 = dict['pos2']= np.random.uniform(0.01, l1)
        else:
            dict = load_dict_list[i]
            color1 = dict.get('color1', (1, 0, 0))
            color2 = dict.get('color2', (0, 1, 0))
            l1 = dict.get('l1', 0)
            l2 = dict.get('l2', 0)
            pos2 = dict.get('pos2', (0, 0, 0))


        save_dict_list.append(dict)

        obj_string = "object{}".format(i)
        print('using friction=({}, {}, {}), object mass{}'.format(f_sliding, f_torsion, f_rolling, object_mass))


        if object_meshes is not None:
            assets = ET.SubElement(root, "asset")

            chosen_mesh = random.choice(object_meshes)
            if chosen_mesh not in loaded_meshes:
                o_mesh = ASSET_BASE_DIR + '{}/'.format(chosen_mesh)
                print('import mesh dir', o_mesh)
                stl_files = glob.glob(o_mesh + '*.stl')
                convex_hull_files = [x for x in stl_files if 'Shape_IndexedFaceSet' in x]
                object_file = [x for x in stl_files
                               if x not in convex_hull_files and 'Lamp' not in x and 'Camera' not in x and 'GM' not in x][0]



                mesh_object = mesh.Mesh.from_file(object_file)
                minx, maxx, miny, maxy, minz, maxz = find_mins_maxs(mesh_object)
                min_length = min((maxx - minx), (maxy - miny))
                scale = [0.12 / min_length for _ in range(3)]

                if chosen_mesh in ['Knife', 'Fork', 'Spoon']:      #should add a more extensible way to handle different rescale rules
                    max_length = max((maxx - minx), (maxy - miny))
                    scale = [0.24 / max_length for _ in range(3)]

                    if chosen_mesh == 'Knife':
                        scale[2] *= 10

                object_pos = [0., 0., 0.]
                object_pos[0] -= scale[0] * (minx + maxx) / 2.0
                object_pos[1] -= scale[1] * (miny + maxy) / 2.0
                object_pos[2] -= 0.08 + scale[2] * (minz + maxz) / 2.0

                mass_per_elem, n_cvx_files = object_mass / (1 + len(convex_hull_files)), len(convex_hull_files)
                loaded_meshes[chosen_mesh] = (object_pos, mass_per_elem, n_cvx_files)

                ET.SubElement(assets, "mesh", name=chosen_mesh + "_mesh", file=object_file,
                              scale="{} {} {}".format(scale[0], scale[1], scale[2]))
                for n, c_file in enumerate(convex_hull_files):
                    ET.SubElement(assets, "mesh", name=chosen_mesh + "_convex_mesh{}".format(n), file=c_file,
                                  scale="{} {} {}".format(scale[0], scale[1], scale[2]))

            else: object_pos, mass_per_elem, n_cvx_files = loaded_meshes[chosen_mesh]

            pos_str = "{} {} {}".format(object_pos[0], object_pos[1], object_pos[2])

            if obj_classname is not None:
                obj = ET.SubElement(world_body, "body",name=obj_string, pos=pos_str,
                                    childclass=obj_classname)
            else: obj = ET.SubElement(world_body, "body",name=obj_string, pos=pos_str)

            ET.SubElement(obj, "joint", type="free")

            #visual mesh
            ET.SubElement(obj, "geom", type="mesh", mesh = chosen_mesh + "_mesh",
                          rgba="{} {} {} 1".format(color1[0], color1[1], color1[2]), mass="{}".format(mass_per_elem),
                          contype="0", conaffinity="0")
            #contact meshes
            for n in range(n_cvx_files):
                ET.SubElement(obj, "geom", type="mesh", mesh=chosen_mesh + "_convex_mesh{}".format(n),
                              rgba="0 1 0 0", mass="{}".format(mass_per_elem),
                              contype="7", conaffinity="7", friction="{} {} {}".format(f_sliding, f_torsion, f_rolling)
                              )



        else:
            obj = None
            if obj_classname is not None:
                obj = ET.SubElement(world_body, "body", name=obj_string, pos="0 0 0",
                                    childclass=obj_classname)
            else: obj = ET.SubElement(world_body, "body", name=obj_string, pos="0 0 0")


            ET.SubElement(obj, "joint", type="free", limited='false', damping="0", armature="0")

            # ET.SubElement(obj, "geom", type="box", size="{} {} {}".format(block_width, l1, block_height),
            #                rgba="{} {} {} 1".format(color1[0], color1[1], color1[2]), mass="{}".format(object_mass),
            #                contype="7", conaffinity="7", friction="{} {} {}".format(f_sliding, f_torsion, f_rolling)
            #                )
            # ET.SubElement(obj, "geom", pos="{} {} 0.0".format(l2, pos2),
            #                type="box", size="{} {} {}".format(l2, block_width, block_height),
            #                rgba="{} {} {} 1".format(color2[0], color2[1], color2[2]), mass="{}".format(object_mass),
            #                contype="7", conaffinity="7", friction="{} {} {}".format(f_sliding, f_torsion, f_rolling)
            #                )
            ET.SubElement(obj, "inertial", mass="0.1", pos="0 0 0", diaginertia="100000 100000 100000")

            # ET.SubElement(obj, "geom", pos="{} {} 0.0".format(l2, pos2), type="cylinder", size=str(cylinder_radius) + " 0.02",
            #                             rgba="{} {} {} 1".format(color2[0], color2[1], color2[2]), mass="{}".format(object_mass),
            #                             contype="7", conaffinity="7", friction="{} {} {}".format(f_sliding, f_torsion, f_rolling))
            ET.SubElement(obj, "geom", pos="0 0 0", type="cylinder", size=str(cylinder_radius) + " 0.015",
                                        rgba="{} {} {} 1".format(color2[0], color2[1], color2[2]),
                                        contype="18", conaffinity="20")

            # ET.SubElement(obj, "site", name=obj_string, pos="{} {} 0.0".format(l2, pos2), size="0.01")

        if sensor_frame is None:
            sensor_frame = ET.SubElement(root, "sensor")
        ET.SubElement(sensor_frame, "framepos", name=obj_string + '_sensor', objtype="body", objname=obj_string)

    tree = ET.ElementTree(root)

    xml_str = minidom.parseString(ET.tostring(
        tree.getroot(),
        'utf-8')).toprettyxml(indent="    ")

    xml_str = xml_str.splitlines()[1:]
    xml_str = "\n".join(xml_str)
    with open(xmldir + "/auto_gen_objects{}.xml".format(os.getpid()), "w") as f:
        f.write(xml_str)

    return save_dict_list

def clean_xml(filename):
    """
    After sim is loaded no need for XML files. Function deletes them before they clutter everything.
    :param filename:
    :return: None
    """
    xml_number = int(filename.split('auto_gen')[-1][:-4])
    obj_file = '/'.join(filename.split('/')[:-1]) + '/auto_gen_objects{}.xml'.format(xml_number)
    print('deleting main file: {} and obj_file: {}'.format(filename, obj_file))
    os.remove(filename)
    os.remove(obj_file)
def create_root_xml(filename):
    """
    copy root xml but replace three lines so that they referecne the object xml
    :param hyperparams:
    :return:
    """

    newlines = []
    autoreplace = False
    replace_done =False
    with open(filename) as f:
        for i, l in enumerate(f):
            if 'begin_auto_replace' in l:
                autoreplace = True
                if not replace_done:
                    newlines.append('        <!--begin auto replaced section -->\n' )
                    newlines.append('        <include file="auto_gen_objects{}.xml"/>\n'.format(os.getpid()))
                    newlines.append('        <!--end auto replaced section -->\n')
                    replace_done = True

            if not autoreplace:
                newlines.append(l)

            if 'end_auto_replace' in l:
                autoreplace = False

    outfile = '/'.join(str.split(filename, '/')[:-1]) + '/auto_gen{}.xml'.format(os.getpid())
    with open(outfile, 'w') as f:
        for l in newlines:
            f.write(l)

    return outfile

if __name__ == '__main__':
    params = imp.load_source('hyper', "/home/frederik/Documents/catkin_ws/src/visual_mpc/pushing_data/cartgripper_genobj/hyperparams.py")
    agentparams = params.config['agent']
    # create_xml(agentparams)
    create_root_xml(agentparams)
