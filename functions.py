import numpy as np
from matplotlib import  pyplot as plt
import math
import cmath
import yaml
from argparse import Namespace
import math
import numpy as np
import bisect

def load_config(path, fname):
    full_path = path + '/config/' + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf

def add_angles(a1, a2):
    angle = (a1+a2)%(2*np.pi)

    return angle

def add_angles_complex(a1, a2):
    real = math.cos(a1) * math.cos(a2) - math.sin(a1) * math.sin(a2)
    im = math.cos(a1) * math.sin(a2) + math.sin(a1) * math.cos(a2)

    cpx = complex(real, im)
    phase = cmath.phase(cpx)

    return phase

def distance_between_points(x1, x2, y1, y2):
    distance = math.hypot(x2-x1, y2-y1)
    
    return distance


