from glob import glob
import re
import os

import argparse

# xyz_base_dependencies.xml => jaco_dependencies.xml
# xyz_base.xml => jaco.xml
# xyz_motor.xml => jaco_motor.xml


def main(assets_dir, arm="ur5e"):
    xmls = glob(os.path.join(assets_dir, "sawyer_xyz", "*.xml"))
    for xml in xmls:
        with open(xml, "r") as f:
            xml_str = f.read()
        xml_str = re.sub(
            r"xyz_base_dependencies.xml", f"{arm}_dependencies.xml", xml_str
        )
        xml_str = re.sub(r"xyz_base.xml", f"{arm}.xml", xml_str)
        xml_str = re.sub(r"xyz_motor.xml", f"{arm}_motor.xml", xml_str)

        new_file_name = os.path.basename(xml).replace("sawyer", f"{arm}")
        new_xml = os.path.join(assets_dir, f"{arm}", new_file_name)
        with open(new_xml, "w") as f:
            f.write(xml_str)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets_dir", type=str, default="./")
    parser.add_argument("--arm", type=str, default="jaco")
    parser.add_argument("--arm_upper", type=str, default="Jaco")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(**vars(args))
