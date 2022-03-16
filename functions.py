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
import cubic_spline_planner
import yaml
from PIL import Image, ImageOps, ImageDraw
import random
from datetime import datetime
import time
from numba import njit
from numba import int32, int64, float32, float64,bool_    
from numba.experimental import jitclass

def load_config(path, fname):
    full_path = path + '/config/' + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf


def add_angles(a1, a2):
    angle = (a1+a2)%(2*np.pi)

    return angle

def sub_angles(a1, a2):
    angle = (a1-a2)%(2*np.pi)

    return angle

def add_angles_complex(a1, a2):
    real = math.cos(a1) * math.cos(a2) - math.sin(a1) * math.sin(a2)
    im = math.cos(a1) * math.sin(a2) + math.sin(a1) * math.cos(a2)

    cpx = complex(real, im)
    phase = cmath.phase(cpx)

    return phase

def sub_angles_complex(a1, a2): 
    real = math.cos(a1) * math.cos(a2) + math.sin(a1) * math.sin(a2)
    im = - math.cos(a1) * math.sin(a2) + math.sin(a1) * math.cos(a2)

    cpx = complex(real, im)
    phase = cmath.phase(cpx)

    return phase

@njit(cache=True)
def distance_between_points(x1, x2, y1, y2):
    distance = math.hypot(x2-x1, y2-y1)
    
    return distance


def generate_circle_image():
    from matplotlib import image
    image = Image.new('RGBA', (600, 600))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 600, 600), fill = 'black', outline ='black')
    draw.ellipse((50, 50, 550, 550), fill = 'white', outline ='white')
    draw.ellipse((150, 150, 450, 450), fill = 'black', outline ='black')
    draw.point((100, 100), 'red')
    image_path = sys.path[0] + '\\maps\\circle' + '.png'
    image.save(image_path, 'png')


def generate_circle_goals():
    from matplotlib import image
    #image_path = sys.path[0] + '\\maps\\circle' + '.png'
    #im = image.imread(image_path)
    #plt.imshow(im, extent=(0,30,0,30))

    R=10
    theta=np.linspace(0, 2*math.pi, 17)
    x = 15+R*np.cos(theta-math.pi/2)
    y = 15+R*np.sin(theta-math.pi/2)
    rx, ry, ryaw, rk, s = cubic_spline_planner.calc_spline_course(x, y)
    #plt.plot(rx, ry, "-r", label="spline")
    #plt.plot(x, y, 'x')
    #plt.show()
    return x, y, rx, ry, ryaw, rk, s


def generate_berlin_goals():
    from matplotlib import image
    #image_path = sys.path[0] + '/maps/berlin' + '.png'
    #im = image.imread(image_path)
    #plt.imshow(im, extent=(0,30,0,30))
    
    goals = [[16,3], [18,4], [18,7], [18,10], [18.5, 13], [19.5,16], [20.5,19], [19.5,22], [17.5,24.5], 
            [15.5,26], [13,26.5], [10,26], [7.5,25], [6,23], [7,21.5], [9.5,21.5], [11, 21.5], 
            [11,20], [10.5,18], [11,16], [12,14], [13,12], [13.5,10], [13.5,8], [14,6], [14.5,4.5], [16,3]]
    
    x = []
    y = []

    for xy in goals:
        x.append(xy[0])
        y.append(xy[1])
    
    rx, ry, ryaw, rk, s = cubic_spline_planner.calc_spline_course(x, y)

    #plt.plot(rx, ry, "-r", label="spline")
    #plt.plot(x, y, 'x')
    #plt.show()

    return x, y, rx, ry, ryaw, rk, s
    

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
    
    return occupancy_grid, map_height, map_width, res 

   
def random_start(x, y, rx, ry, ryaw, rk, s, episode):
    offset=0.5
    random.seed(datetime.now())
    
    if episode < 20000:
        if random.uniform(0,1)<0.1:
            i = int(random.uniform(0, len(x)-2))
        else:
            i = int(random.uniform(10, 14))
    
    elif episode >= 20000 and episode <50000:
        if random.uniform(0,1)<0.5:
            i = int(random.uniform(0, len(x)-2))
        else:
            i = int(random.uniform(10, 14))

    else:
        i = int(random.uniform(0, len(x)-2))
    
    #i = int(random.uniform(0, len(x)-2))
    #i = int(random.uniform(10, 12))
    
    next_i = (i+1)%len(y)
    start_x = x[i] + (random.uniform(-1.5, 1.5))
    start_y = y[i] + (random.uniform(-1.5, 1.5))
    start_theta = math.atan2(y[next_i]-start_y, x[next_i]-start_x) + (random.uniform(-math.pi/4, math.pi/4))
    next_goal = (i+1)%len(x)

    return start_x, start_y, start_theta, next_i


def find_closest_point(rx, ry, x, y):

    dx = [x - irx for irx in rx]
    dy = [y - iry for iry in ry]
    d = np.hypot(dx, dy)    
    ind = np.argmin(d)
    
    return ind



def find_angle_to_line(ryaw, theta):

    angle = np.abs(sub_angles_complex(ryaw, theta))

    return angle


@njit(cache=True)
def occupied_cell(x, y, occupancy_grid, res, map_height):
    
    cell = (np.array([map_height-y, x])/res).astype(np.int64)

    if occupancy_grid[cell[0], cell[1]] == True:
        return True
    else:
        return False


spec = [('lidar_res', float32),
        ('n_beams', int32),
        ('max_range', float32),
        ('fov', float32),
        ('occupancy_grid', bool_[:,:]),
        ('map_res', float32),
        ('map_height', float32),
        ('beam_angles', float64[:])]


@jitclass(spec)
class lidar_scan():
    def __init__(self, lidar_res, n_beams, max_range, fov, occupancy_grid, map_res, map_height):
        
        self.lidar_res = lidar_res
        self.n_beams  = n_beams
        self.max_range = max_range
        self.fov = fov
        
        #self.beam_angles = (self.fov/(self.n_beams-1))*np.arange(self.n_beams)
        
        self.beam_angles = np.zeros(self.n_beams, dtype=np.float64)
        for n in range(self.n_beams):
            self.beam_angles[n] = (self.fov/(self.n_beams-1))*n

        self.occupancy_grid = occupancy_grid
        self.map_res = map_res
        self.map_height = map_height

    def get_scan(self, x, y, theta):
        
        scan = np.zeros((self.n_beams))
        coords = np.zeros((self.n_beams, 2))
        
        for n in range(self.n_beams):
            i=1
            occupied=False

            while i<(self.max_range/self.lidar_res) and occupied==False:
                x_beam = x + np.cos(theta+self.beam_angles[n]-self.fov/2)*i*self.lidar_res
                y_beam = y + np.sin(theta+self.beam_angles[n]-self.fov/2)*i*self.lidar_res
                occupied = occupied_cell(x_beam, y_beam, self.occupancy_grid, self.map_res, self.map_height)
                i+=1
            
            coords[n,:] = [np.round(x_beam,3), np.round(y_beam,3)]
            #dist = np.linalg.norm([x_beam-x, y_beam-y])
            dist = math.sqrt((x_beam-x)**2 + (y_beam-y)**2)
            
            scan[n] = np.round(dist,3)

        return scan, coords

#generate_berlin_goals()
if __name__ == 'main':
    #def velocity_along_line(theta, velocity, ryaw, )

    #generate_berlin_goals()
    #x, y, rx, ry, ryaw, rk, s = generate_circle_goals()
    #start_x, start_y, start_theta, next_goal = random_start(x, y, rx, ry, ryaw, rk, s)

    #image_path = sys.path[0] + '/maps/' + 'circle' + '.png'       
    #occupancy_grid, map_height, map_width, res = map_generator(map_name='circle')
    #a = lidar_scan(res, 3, 10, np.pi, occupancy_grid, res, 30)
    #print(a.get_scan(15,5,0))


    #im = image.imread(image_path)
    #plt.imshow(im, extent=(0,30,0,30))
    #plt.plot(start_x, start_y, 'x')
    #print(start_theta)
    #plt.arrow(start_x, start_y, math.cos(start_theta), math.sin(start_theta))
    #plt.plot(x, y, 's')
    #plt.plot(x[next_goal], y[next_goal], 'o')
    #plt.show()
    pass

