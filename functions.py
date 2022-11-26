import numpy as np
from matplotlib import  pyplot as plt
from matplotlib import image
import math
import cmath
import yaml
from argparse import Namespace
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
import pickle
import mapping
import cubic_spline_planner
from matplotlib import rc
import matplotlib.font_manager as font_manager

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

def add_locations(x1=[0, 0], x2=[0, 0], dx=1):
    # dx is a scaling factor
    ret = [0.0, 0.0]
    for i in range(2):
        ret[i] = x1[i] + x2[i] * dx
    return ret


def get_distance(x1=[0, 0], x2=[0, 0]):
    d = [0.0, 0.0]
    for i in range(2):
        d[i] = x1[i] - x2[i]
    return np.linalg.norm(d)
     
def sub_locations(x1=[0, 0], x2=[0, 0], dx=1):
    # dx is a scaling factor
    ret = [0.0, 0.0]
    for i in range(2):
        ret[i] = x1[i] - x2[i] * dx
    return ret



def get_gradient(x1=[0, 0], x2=[0, 0]):
    t = (x1[1] - x2[1])
    b = (x1[0] - x2[0])
    if b != 0:
        return t / b
    return 1000000 # near infinite gradient. 

def transform_coords(x=[0, 0], theta=np.pi):
    # i want this function to transform coords from one coord system to another
    new_x = x[0] * np.cos(theta) - x[1] * np.sin(theta)
    new_y = x[0] * np.sin(theta) + x[1] * np.cos(theta)

    return np.array([new_x, new_y])

def normalise_coords(x=[0, 0]):
    r = x[0]/x[1]
    y = np.sqrt(1/(1+r**2)) * abs(x[1]) / x[1] # carries the sign
    x = y * r
    return [x, y]

def get_bearing(x1=[0, 0], x2=[0, 0]):
    grad = get_gradient(x1, x2)
    dx = x2[0] - x1[0]
    th_start_end = np.arctan(grad)
    if dx == 0:
        if x2[1] - x1[1] > 0:
            th_start_end = 0
        else:
            th_start_end = np.pi
    elif th_start_end > 0:
        if dx > 0:
            th_start_end = np.pi / 2 - th_start_end
        else:
            th_start_end = -np.pi/2 - th_start_end
    else:
        if dx > 0:
            th_start_end = np.pi / 2 - th_start_end
        else:
            th_start_end = - np.pi/2 - th_start_end

    return th_start_end


#@njit(cache=True)
def distance_between_points(x1, x2, y1, y2):
    distance = math.hypot(x2-x1, y2-y1)
    
    return distance

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

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

   
def random_start(rx, ry, ryaw, distance_offset, angle_offset):
    
    #random.seed(datetime.now())
    
    '''
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
    '''

    # i = int(random.uniform(0, len(x)-2))
    
    # #i = int(random.uniform(0, len(x)-2))
    # #i = int(random.uniform(10, 12))
    
    # next_i = (i+1)%len(y)
    # start_x = x[i] + (random.uniform(-distance_offset, distance_offset))
    # start_y = y[i] + (random.uniform(-distance_offset, distance_offset))
    
    # start_theta = math.atan2(y[next_i]-y[i], x[next_i]-x[i]) + (random.uniform(-angle_offset, angle_offset))
    # next_goal = (i+1)%len(x)

    i = int(random.uniform(0, len(rx)))
    start_x = rx[i] + random.uniform(-distance_offset, distance_offset)
    start_y = ry[i] + random.uniform(-distance_offset, distance_offset)
    start_theta = ryaw[i] + random.uniform(-angle_offset, angle_offset)
    next_i = 0

    return start_x, start_y, start_theta, next_i



def find_closest_point(rx, ry, x, y):

    dx = [x - irx for irx in rx]
    dy = [y - iry for iry in ry]
    d = np.hypot(dx, dy)    
    ind = np.argmin(d)
    
    return ind

def check_closest_point(rx, ry, x, y, occupancy_grid, res, map_height):
    cp_ind = find_closest_point(rx, ry, x, y)   #find closest point 
    
    cp_x = rx[cp_ind]
    cp_y = ry[cp_ind]
    
    los_x = np.linspace(x, cp_x)
    los_y = np.linspace(y, cp_y)

    for x, y in zip(los_x, los_y):
        if occupied_cell(x, y, occupancy_grid, res, map_height):
            return True
    return False


def is_line_of_sight_clear(x1, y1, x2, y2, occupancy_grid, res, map_height):
    
    los_x = np.linspace(x1, x2)
    los_y = np.linspace(y1, y2)

    for x, y in zip(los_x, los_y):
        if occupied_cell(x, y, occupancy_grid, res, map_height):
            return False
    return True

def find_correct_closest_point(rx, ry, x, y, occupancy_grid, res, map_height):
    
    ind = find_closest_point(rx, ry, x, y)
    cpx = rx[ind]
    cpy = ry[ind]
    if is_line_of_sight_clear(x, y, cpx, cpy, occupancy_grid, res, map_height):
        return ind
    else:
        dx = [x - irx for irx in rx]
        dy = [y - iry for iry in ry]
        d = np.hypot(dx, dy)    
        inds = np.argsort(d)

        for i in inds:
            cpx = rx[i]
            cpy = ry[i]
            if is_line_of_sight_clear(x, y, cpx, cpy, occupancy_grid, res, map_height):
                return i
        else:
            print('No line of sight to centerline')
            return ind

def convert_xy_to_sn(rx, ry, ryaw, x, y, ds):
    dx = [x - irx for irx in rx]    
    dy = [y - iry for iry in ry]
    d = np.hypot(dx, dy)    #Get distances from (x,y) to each point on centerline    
    ind = np.argmin(d)      #Index of s coordinate
    s = ind*ds              #Exact position of s
    n = d[ind]              #n distance (unsigned), not interpolated

    #Get sign of n by comparing angle between (x,y) and (s,0), and the angle of the centerline at s
    xy_angle = np.arctan2((y-ry[ind]),(x-rx[ind]))      #angle between (x,y) and (s,0)
    yaw_angle = ryaw[ind]                               #angle at s
    angle = sub_angles_complex(xy_angle, yaw_angle)     
    if angle >=0:   #Vehicle is above s line
        direct=1    #Positive n direction
    else:           #Vehicle is below s line
        direct=-1   #Negative n direction

    n = n*direct   #Include sign 

    return s, ind, n

def find_angle(A, B, C):
    # RETURNS THE ANGLE BÂC
    vec_AB = A - B
    vec_AC = A - C 
    dot = vec_AB.dot(vec_AC)
    #dot = (A[0] - C[0])*(A[0] - B[0]) + (A[1] - C[1])*(A[1] - B[1])
    magnitude_AB = np.linalg.norm(vec_AB)
    magnitude_AC = np.linalg.norm(vec_AC)

    angle = np.arccos(dot/(magnitude_AB*magnitude_AC))
    
    return angle

def get_angle(a,b,c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    print(np.degrees(angle))

    return angle



def convert_sn_to_xy(s, n, csp):
    x = []
    y = []
    yaw = []
    ds = []
    c = []
    for i in range(len(s)):
        ix, iy = csp.calc_position(s[i])
        if ix is None:
            break
        i_yaw = csp.calc_yaw(s[i])
        ni = n[i]
        fx = ix + ni * math.cos(i_yaw + math.pi / 2.0)
        fy = iy + ni * math.sin(i_yaw + math.pi / 2.0)
        x.append(fx)
        y.append(fy)

    # calc yaw and ds
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        yaw.append(math.atan2(dy, dx))
        ds.append(math.hypot(dx, dy))
        yaw.append(yaw[-1])
        ds.append(ds[-1])

    # calc curvature
    #for i in range(len(yaw) - 1):
    #    c.append((yaw[i + 1] - yaw[i]) / ds[i])
    c = 0

    return x, y, yaw, ds, c

def generate_line(x, y):
    csp = cubic_spline_planner.Spline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, s, csp

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

def generate_initial_condition(name, episodes, distance_offset, angle_offset, vel_select):
    file_name = 'test_initial_condition/' + name
   
    initial_conditions = []
   
    track = mapping.map(name)
    track.find_centerline()
    goal_x = track.centerline[:,0]
    goal_y = track.centerline[:,1]
    rx, ry, ryaw, rk, d = cubic_spline_planner.calc_spline_course(goal_x, goal_y)
    
    k = [i for i in range(len(rk)) if abs(rk[i])>1]
    spawn_ind = np.full(len(rx), True)
    for i in k:
        spawn_ind[np.arange(i-10, i+5)] = False
    
    x = [rx[i] for i in range(len(rx)) if spawn_ind[i]==True]
    y = [ry[i] for i in range(len(ry)) if spawn_ind[i]==True]
    yaw = [ryaw[i] for i in range(len(ryaw)) if spawn_ind[i]==True]
    
    for eps in range(episodes):
        x_s, y_s, theta_s, current_goal = random_start(x, y, yaw, distance_offset, angle_offset)
        #x, y, theta = random_start(goal_x, goal_y, rx, ry, ryaw, rk, d, distance_offset, angle_offset)
        v_s = random.random()*(vel_select[1]-vel_select[0])+vel_select[0]
        delta_s = 0
        i = {'x':x_s, 'y':y_s, 'v':v_s, 'delta':delta_s, 'theta':theta_s, 'goal':current_goal}
        initial_conditions.append(i)

    #initial_conditions = [ [] for _ in range(episodes)]

    x = [initial_conditions[i]['x'] for i in range(len(initial_conditions))]
    y = [initial_conditions[i]['y'] for i in range(len(initial_conditions))]  
    
    plt.imshow(track.gray_im, extent=(0,track.map_width,0,track.map_height))
    plt.plot(rx, ry)
    plt.plot(goal_x, goal_y, 'o')
    plt.plot(x, y, 'x')
    plt.show()


    outfile=open(file_name, 'wb')
    pickle.dump(initial_conditions, outfile)
    outfile.close()


if __name__ == '__main__':
    
    ds = 0.1
    x_sparse = np.array([0,10])
    y_sparse = [0,0]
    rx, ry, ryaw, rk, s, csp = generate_line(x_sparse, y_sparse)
    
    x = 1
    y = 1
    #transform_XY_to_NS(rx, ry, x, y)
    convert_xy_to_sn(rx, ry, ryaw, x, y, ds)
    
    s_0 = 0
    s_1 = s_0+1
    s_2 = s_1+0.5
    theta = 0.5
    n_0 = 0.25
    
    ns = [[] for _ in range(3)]


  

    fig, ax = plt.subplots(1, figsize=(5,4.2))
    ax.set_xticks(ticks=[s_0, s_1], labels=['$s_0$', '$s_1$'],)
    ax.set_yticks(ticks=[n_0, 0.7], labels=['$n_0$', '$n_1$'])

    font_legend = font_manager.FontProperties(family='Serif',
                                   style='normal', size=12)

    font_dict = {'family': 'serif',
        'size': 12,
        }


    ax.plot(s_0,n_0, 'x', label='Vehicle position')
    ax.grid(True)

    for i in range(3): 
        # n_1 = 1
        # n_1=0.7    
        n_1s = [0.9,-0.9,0.7]
        #colors = ['orange','orange','red']
        #alphas = [0.8,0.8,1]

        A = np.array([[3*s_1**2, 2*s_1, 1, 0], [3*s_0**2, 2*s_0, 1, 0], [s_0**3, s_0**2, s_0, 1], [s_1**3, s_1**2, s_1, 1]])
        B = np.array([0, theta, n_0, n_1s[i]])
        x = np.linalg.solve(A, B)
        #print(x)

        a = x[0]
        b = x[1]
        c = x[2]
        d = x[3]

        s = np.linspace(s_0, s_1)
        n = a*s**3 + b*s**2 + c*s + d
        s = np.concatenate((s, np.linspace(s_1, s_2)))
        n = np.concatenate((n, np.ones(len(np.linspace(s_1, s_2)))*n_1s[i]))
        ns[i].append(n)

    
    ax.plot(s, ns[2][0], color='dodgerblue', label='Sample path')
    #plt.plot(s, ns[1][0])
    #plt.plot(s, ns[2][0])

    ax.fill_between(x=s, y1=ns[1][0], y2=ns[0][0], alpha=0.2, color='dodgerblue', label='Range of paths')

    #plt.plot(np.linspace(s_1, s_2), np.ones(len(np.linspace(s_1, s_2)))*n_1s[i], color=colors[i],alpha=alphas[i])
    xlims = [s_0-0.2, s_2+0.2]
    ax.hlines(y=1, xmin=xlims[0], xmax=xlims[1], linestyle='solid',color='black',label='Track boundaries')
    ax.hlines(y=-1, xmin=xlims[0], xmax=xlims[1], linestyle='solid',color='black',label='_nolegend_')
    ax.hlines(y=0, xmin=xlims[0], xmax=xlims[1], linestyle='--',color='grey',label='Centerline')
   
    ax.set_xlim(xlims)
    ax.set_xlabel('s [m]', fontdict=font_dict)
    ax.set_ylabel('n [m]', fontdict=font_dict)
    #ax.legend(loc='lower left')
    
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35)
    fig.subplots_adjust(right=0.8)
    fig.subplots_adjust(left=0.2)
    plt.figlegend(loc = 'lower center', ncol=2, prop=font_legend)
    plt.show()
    
    
    #def velocity_along_line(theta, velocity, ryaw, )
    

    #generate_berlin_goals()
    #x, y, rx, ry, ryaw, rk, s = generate_circle_goals()
    #start_x, start_y, start_theta, next_goal = random_start(x, y, rx, ry, ryaw, rk, s)

    #image_path = sys.path[0] + '/maps/' + 'circle' + '.png'       
    #occupancy_grid, map_height, map_width, res = map_generator(map_name='circle')
    #a = lidar_scan(res, 3, 10, np.pi, occupancy_grid, res, 30)
    #print(a.get_scan(15,5,0))
    
    # generate_initial_condition('porto_1', 2000, distance_offset=0.2, angle_offset=np.pi/8, vel_select=[3,5])
    # generate_initial_condition('columbia_1', 2000, distance_offset=0.2, angle_offset=np.pi/8, vel_select=[3,5])
    # generate_initial_condition('circle', 2000, distance_offset=0.2, angle_offset=np.pi/8, vel_select=[3,5])
    # generate_initial_condition('berlin', 2000, distance_offset=0.2, angle_offset=np.pi/8, vel_select=[3,5])
    # generate_initial_condition('torino', 2000, distance_offset=0.2, angle_offset=np.pi/8, vel_select=[3,5])
    # generate_initial_condition('redbull_ring', 2000, distance_offset=0.2, angle_offset=np.pi/8, vel_select=[3,5])

    #im = image.imread(image_path)
    #plt.imshow(im, extent=(0,30,0,30))
    #plt.plot(start_x, start_y, 'x')
    #print(start_theta)
    #plt.arrow(start_x, start_y, math.cos(start_theta), math.sin(start_theta))
    #plt.plot(x, y, 's')
    #plt.plot(x[next_goal], y[next_goal], 'o')
    #plt.show()
    

