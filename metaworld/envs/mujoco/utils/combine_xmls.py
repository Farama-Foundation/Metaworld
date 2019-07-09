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
import metaworld

def combine_xmls(xmls):
    trees = [ET.parse(xml) for xml in xmls]
    roots = [tree.getroot() for tree in trees]
    for i in range(len(roots)):
        for worldbody in roots[i].findall('worldbody'):
            for body in worldbody.findall('body'):
                if 'name' in body.attrib:
                    body.set('name', 'task%d_' % i + body.get('name'))
                    for geom in body.findall('geom'):
                        if 'name' in geom.attrib:
                            geom.set('name', 'task%d_' % i + geom.get('name'))
                    for joint in body.findall('joint'):
                        if 'name' in joint.attrib:
                            joint.set('name', 'task%d_' % i + joint.get('name'))
                    for site in body.findall('site'):
                        if 'name' in site.attrib:
                            site.set('name', 'task%d_' % i + site.get('name'))
                if i != 0:
                    roots[0].find('worldbody').append(body)
            for site in worldbody.findall('site'):
                if 'name' in site.attrib:
                    site.set('name', 'task%d_' % i + site.get('name'))
                if i != 0:
                    roots[0].find('worldbody').append(site)
        for tendon in roots[i].findall('tendon'):
            for site in tendon.iter('site'):
                if 'site' in site.attrib:
                    site.set('site', 'task%d_' % i + site.get('site'))
            if i != 0:
                roots[0].append(tendon)
    trees[0].write(os.path.expanduser('~/combined.xml'))
    return trees[0]

if __name__ == '__main__':
    xmls = glob.glob(os.path.expanduser('~/metaworld/metaworld/envs/assets/sawyer_multitask/sawyer*.xml'))
    xmls = [xml for xml in xmls if 'base' not in xml]
    random_xmls = np.random.choice(xmls, size=3, replace=False)
    print(random_xmls)
    combine_xmls(random_xmls)
