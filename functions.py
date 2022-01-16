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
        '''
        im = image.imread(image_path)
        plt.imshow(im, extent=(0,30,0,30))
        plt.plot(16, 3, 'x')
        plt.plot(18,4, 'x')
        plt.plot(18, 7, 'x')
        plt.plot(18, 10, 'x')
        plt.plot(18.5, 13, 'x')
        plt.plot(19.5, 16, 'x')
        plt.plot(20.5, 19, 'x')   
        plt.plot(19.5, 22, 'x')
        plt.plot(17.5, 24.5, 'x')
        plt.plot(15.5, 26, 'x')
        plt.plot(13, 26.5, 'x')
        plt.plot(10, 26, 'x')
        plt.plot(7.5, 25, 'x')
        plt.plot(6, 23, 'x')
        plt.plot(7, 21.5, 'x')
        plt.plot(9.5, 21.5, 'x')
        plt.plot(11, 21.2, 'x')
        plt.plot(11, 20, 'x')
        plt.plot(10.5, 18, 'x')
        plt.plot(11, 16, 'x')
        plt.plot(12, 14, 'x')
        plt.plot(13, 12, 'x')
        plt.plot(13.5, 10, 'x')
        plt.plot(13.5, 8, 'x')
        plt.plot(14, 6, 'x')
        plt.plot(14.5, 4.5, 'x')
        plt.show()
        '''
         #[[18, 4], [18, 4], [18,7], [18,10], [18.5, 13], [19.5,16], [20.5,19], [19.5,22], [17.5,24.5], [15.5,26], [13,26.5], [10,26], [7.5,25], [6,23], [7,21.5], 
         # [9.5,21.5], [11,20], [10.5,18], [11,16], [12,14], [13,12], [13.5,10], [13.5,8], [14,6], [14.5,4.5]]
    
    return occupancy_grid, map_height, map_width, res 


def detect_collision(occupancy_grid, x, y, res):
    cell = (np.array([30-y, x])/res).astype(int)
    #plt.imshow(occupancy_grid)
    #plt.show()
    if occupancy_grid[cell[0], cell[1]] == True:
        return True
    else:
        return False

        

    


        
#ccupancy_grid, map_height, map_width, res = map_generator(map_name='berlin')
#print(detect_collision(occupancy_grid, 16, 2, res))

#im = image.imread(image_path)
#plt.imshow(im, extent=(0,30,0,30))
#plt.show()