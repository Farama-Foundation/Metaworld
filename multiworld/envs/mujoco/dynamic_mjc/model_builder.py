"""
model_builder.py
A small library for programatically building MuJoCo XML files
"""
from contextlib import contextmanager
import tempfile
import numpy as np


def default_model(name):
    """
    Get a model with basic settings such as gravity and RK4 integration enabled
    """
    model = MJCModel(name)
    root = model.root

    # Setup
    root.compiler(angle="radian", inertiafromgeom="true")
    default = root.default()
    default.joint(armature=1, damping=1, limited="true")
    default.geom(contype=0, friction='1 0.1 0.1', rgba='0.7 0.7 0 1')
    root.option(gravity="0 0 -9.81", integrator="RK4", timestep=0.01)
    return model

def pointmass_model(name):
    """
    Get a model with basic settings such as gravity and Euler integration enabled
    """
    model = MJCModel(name)
    root = model.root

    # Setup
    root.compiler(angle="radian", inertiafromgeom="true", coordinate="local")
    default = root.default()
    default.joint(limited="false", damping=1)
    default.geom(contype=2, conaffinity="1", condim="1", friction=".5 .1 .1", density="1000", margin="0.002")
    root.option(timestep=0.01, gravity="0 0 0", iterations="20", integrator="Euler")
    return model


class MJCModel(object):
    def __init__(self, name, include_config=False):
        self.name = name
        if not include_config:
            self.root = MJCTreeNode("mujoco").add_attr('model', name)
        else:
            self.root = MJCTreeNode("mujoco")
            
    @contextmanager
    def asfile(self):
        """
        Usage:

        model = MJCModel('reacher')
        with model.asfile() as f:
            print f.read()  # prints a dump of the model

        """
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.xml', delete=True) as f:
            self.root.write(f)
            f.seek(0)
            yield f

    def open(self):
        self.file = tempfile.NamedTemporaryFile(mode='w+', suffix='.xml', delete=True)
        self.root.write(self.file)
        self.file.seek(0)
        return self.file

    def save(self, path):
        with open(path, 'w') as f:
            self.root.write(f)

    def close(self):
        self.file.close()


class MJCModelRegen(MJCModel):
    def __init__(self, name, regen_fn):
        super(MJCModelRegen, self).__init__(name)
        self.regen_fn = regen_fn

    def regenerate(self):
        self.root = self.regen_fn().root



class MJCTreeNode(object):
    def __init__(self, name, end_with_name=False):
        self.name = name
        self.attrs = {}
        self.children = []
        self.end_with_name = end_with_name

    def add_attr(self, key, value):
        if isinstance(value, str):  # should be basestring in python2
            pass
        elif isinstance(value, list) or isinstance(value, np.ndarray):
            value = ' '.join([str(val) for val in value])

        self.attrs[key] = value
        return self

    def __getattr__(self, name):
        def wrapper(**kwargs):
            if 'end_with_name' in kwargs.keys():
                end_with_name = kwargs.pop('end_with_name')
            else:
                end_with_name = False
            newnode =  MJCTreeNode(name, end_with_name=end_with_name)
            for (k, v) in kwargs.items(): # iteritems in python2
                newnode.add_attr(k, v)
            self.children.append(newnode)
            return newnode
        return wrapper

    def dfs(self):
        yield self
        if self.children:
            for child in self.children:
                for node in child.dfs():
                    yield node

    def write(self, ostream, tabs=0):
        contents = ' '.join(['%s="%s"'%(k,v) for (k,v) in self.attrs.items()])
        if self.children:

            ostream.write('\t'*tabs)
            ostream.write('<%s %s>\n' % (self.name, contents))
            for child in self.children:
                child.write(ostream, tabs=tabs+1)
            ostream.write('\t'*tabs)
            ostream.write('</%s>\n' % self.name)
        else:
            ostream.write('\t'*tabs)
            if self.end_with_name:
                ostream.write('<%s %s></%s>\n' % (self.name, contents, self.name))
            else:
                ostream.write('<%s %s/>\n' % (self.name, contents))

    def __str__(self):
        s = "<"+self.name
        s += ' '.join(['%s="%s"'%(k,v) for (k,v) in self.attrs.items()])
        return s+">"