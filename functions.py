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

def plot_history(state_history, action_history, local_path_history):
    
    for sh, ah, lph in zip(state_history, action_history, local_path_history):
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        #plt.plot([x[0] for x in state_history], [x[1] for x in state_history])
        #plt.plot([x[0] for x in action_history], [x[1] for x in action_history], 'x')
        plt.arrow(sh[0], sh[1], 0.1*math.cos(sh[2]), 0.1*math.sin(sh[2]), head_length=0.04,head_width=0.02, ec='None', fc='blue')
        plt.arrow(sh[0], sh[1], 0.1*math.cos(sh[2]+sh[3]), 0.1*math.sin(sh[2]+sh[3]), head_length=0.04,head_width=0.02, ec='None', fc='red')
        plt.plot(sh[0], sh[1], 'o')
        plt.plot(ah[0], ah[1], 'x')
        plt.plot(lph[0], lph[1], 'g')
        plt.plot([1.5, 2.5, 2.5, 1.5, 1.5], [1.5, 1.5, 2.5, 2.5, 1.5], 'r')
        plt.legend(["vehicle trajectory", "predicted waypoints", "goal area"])
        plt.xlabel('x coordinate')
        plt.ylabel('y coordinate')
        plt.xlim([-0.5,3])
        plt.ylim([-0.5,3])
        plt.grid(True)
        plt.title('Vehicle trajectory')
        plt.pause(0.01)


