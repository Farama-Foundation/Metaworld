import xml.etree.cElementTree as ET
import xml.dom.minidom as minidom
import numpy as np

num_sensors = 20
angle_shift = np.pi*2/num_sensors

root = ET.Element("to_be_inserted_into_particle_body")

ball_radius = 0.05

circum = 2*np.pi*ball_radius
srad = circum/ num_sensors/2
srad = round(srad, 3)

for i in range(num_sensors):
    angle = i*angle_shift
    x = round(np.cos(angle)*ball_radius, 4)
    y = round(np.sin(angle)*ball_radius, 4)
    # body = ET.SubElement(root, "body", name="sensbody" + str(i), pos="0 0 0", euler="0 0 " + str(angle))
    ET.SubElement(root, "site", name="sensorsurf"+str(i), pos=str(x)+" "+str(y)+ " 0", size=str(srad)+" "+str(srad)+" "+str(srad), type="ellipsoid", rgba="0.3 0.2 0.1 1")


for i in range(num_sensors):
    body = ET.SubElement(root, "touch", name="touchsensor"+ str(i), site="sensorsurf"+str(i))

tree = ET.ElementTree(root)




xml_str =  minidom.parseString(
            ET.tostring(
              tree.getroot(),
              'utf-8')).toprettyxml(indent="    ")


f =  open("touchsensor.xml", "wb")
f.write(xml_str)
f.close()





# tree.write("filename.xml")