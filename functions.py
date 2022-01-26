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


def detect_collision(occupancy_grid, x, y, res):
    cell = (np.array([30-y, x])/res).astype(int)
    #plt.imshow(occupancy_grid)
    #plt.show()
    if occupancy_grid[cell[0], cell[1]] == True:
        return True
    else:
        return False

      
def random_start(x, y, rx, ry, ryaw, rk, s):
    offset=0.5

    random.seed(datetime.now())
    i = int(random.uniform(0, len(x)-2))
    next_i = (i+1)%len(y)
    start_x = x[i] + (random.uniform(-1.5, 1.5))
    start_y = y[i] + (random.uniform(-1.5, 1.5))
    start_theta = math.atan2(y[next_i]-start_y, x[next_i]-start_x) + (random.uniform(-math.pi/4, math.pi/4))
    next_goal = (i+1)%len(x)

    return start_x, start_y, start_theta, next_i


class measure_progress():
    def __init__(self, rx, ry):
        self.rx = rx
        self.ry = ry
        self.old_nearest_point_index = None

    def search_nearest_index(self, x, y):
    
        if self.old_nearest_point_index is None:
            #Get distances to every point
            dx = [x - icx for icx in self.cx]
            dy = [y - icy for icy in self.cy]
            d = np.hypot(dx, dy)    
            ind = np.argmin(d)      #Get nearest point
            self.old_nearest_point_index = ind  #Set previous nearest point to nearest point
        
        else:   #If there exists a previous nearest point - after the start
            #Search for closest waypoint after ind
            ind = self.old_nearest_point_index  
            #self.ind_history.append(ind)
        
            distance_this_index = distance_between_points(self.cx[ind], x, self.cy[ind], y)   
            
            while True:
                if (ind+1)>=len(self.cx):
                    break
                
                distance_next_index = distance_between_points(self.cx[ind + 1], x, self.cy[ind + 1], y)
                
                if distance_this_index < distance_next_index:
                    break

                ind = ind + 1 if (ind + 1) < len(self.cx) else ind  #Increment index - search for closest waypoint
                
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        return ind
    
    def progress(self, x, y):

        new_ind = self.search_nearest_index(x, y)
        prg = (new_ind-self.old_nearest_point_index)/len(self.rx)
        self.old_nearest_point_index = new_ind
        return prg



#generate_berlin_goals()
#x, y, rx, ry, ryaw, rk, s = generate_circle_goals()
#start_x, start_y, start_theta, next_goal = random_start(x, y, rx, ry, ryaw, rk, s)

#image_path = sys.path[0] + '/maps/' + 'circle' + '.png'       
#occupancy_grid, map_height, map_width, res = map_generator(map_name='circle')
#print(detect_collision(occupancy_grid, 15, 5, res))

#im = image.imread(image_path)
#plt.imshow(im, extent=(0,30,0,30))
#plt.plot(start_x, start_y, 'x')
#print(start_theta)
#plt.arrow(start_x, start_y, math.cos(start_theta), math.sin(start_theta))
#plt.plot(x, y, 's')
#plt.plot(x[next_goal], y[next_goal], 'o')
#plt.show()

