import numpy as np
from matplotlib import  pyplot as plt
from matplotlib import image
import math
import cmath
import yaml
from argparse import Namespace
import math
import numpy as np
import bisect
import sys
from PIL import Image, ImageOps


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


def map_generator(map_name):
    map_config_path = sys.path[0] + '/maps/' + map_name + '.yaml'
    image_path = sys.path[0] + '/maps/' + map_name + '.png'
    with open(map_config_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    map_conf = Namespace(**conf_dict)
    
    res=map_conf.resolution

    with Image.open(image_path) as im:
        gray_im = ImageOps.grayscale(im)
        map_array = np.asarray(gray_im)
        map_height = gray_im.height*res
        map_width = gray_im.width*res
        occupancy_grid = map_array<1
    
        #im = image.imread(image_path)
        #plt.imshow(im, extent=(0,30,0,30))
        #plt.plot(16, 2, 'x')
        #plt.show()
    
    return occupancy_grid, map_height, map_width, res 


def detect_collision(occupancy_grid, x, y, res):
    cell = (np.array([30-x, 30-y])/res).astype(int)
    #plt.imshow(occupancy_grid)
    #plt.show()
    #if occupancy_grid[cell[0], cell[1]] == True:
    #    return True
    #else:
    #    return False

    for y in range(0,100):
        print('row ', y)
        for x in range(0,600):
            if occupancy_grid[x, y] == False:
                plt.plot(x, y, 'x')
    plt.show()
        

    


        
occupancy_grid, map_height, map_width, res = map_generator(map_name='berlin')
print(detect_collision(occupancy_grid, 16, 2, res))

#im = image.imread(image_path)
#plt.imshow(im, extent=(0,30,0,30))
#plt.show()