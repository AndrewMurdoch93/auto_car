import math
from re import X
import numpy as np
import functions
from numba import njit
from numba import int32, int64, float32, float64,bool_    
from numba.experimental import jitclass

@njit(cache=True)
def pure_pursuit(wheelbase, waypoint, x, y, theta):
    '''
    A simple pure pursuit implementation for single waypoint following
    Based on formulas in the youtube video:
    https://www.youtube.com/watch?v=zMdoLO4kRKg&t=73s

    Only tracks 1 point at a time
    '''
    
    ld = math.sqrt((waypoint[0]-x)**2 + (waypoint[1]-y)**2)
    alpha = math.atan2(waypoint[1] - y, waypoint[0] - x) - theta
    delta_ref = math.atan2(2.0 * wheelbase * math.sin(alpha) / ld, 1.0)

    return delta_ref


#spec = [('cx', float64[:]),
#        ('cy', float64[:]),
#        ('old_nearest_point_index',  int32),
#        ('wheelbase', float64),
#        ('k', float64),
#        ('Lfc', float64)]
#@jitclass(spec)
class pure_pursuit_path():
    '''
    More complex pure pursuit for multiple waypoint/ path following
    '''
    
    def __init__(self, track_dict):

        self.old_nearest_point_index = None
        self.wheelbase = track_dict['wheelbase']
        self.k = track_dict['k']
        self.Lfc = track_dict['Lfc']

    def record_waypoints(self, cx, cy, cyaw):
        #Initialise waypoints for planner
        self.cx=cx
        self.cy=cy
        self.cyaw = cyaw
        self.old_nearest_point_index = None

    def search_target_waypoint(self, x, y, v):
        
        #If there is no previous nearest point - at the start
        if self.old_nearest_point_index == None:
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
        
            distance_this_index = functions.distance_between_points(self.cx[ind], x, self.cy[ind], y)   
            
            while True:
                if (ind+1)>=len(self.cx):
                    break
                
                distance_next_index = functions.distance_between_points(self.cx[ind + 1], x, self.cy[ind + 1], y)
                
                if distance_this_index < distance_next_index:
                    break

                ind = ind + 1 if (ind + 1) < len(self.cx) else ind  #Increment index - search for closest waypoint
                
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind  

        Lf = self.k * v + self.Lfc  # update look ahead distance

        # search look ahead target point index
        while Lf > functions.distance_between_points(self.cx[ind], x, self.cy[ind], y):
            if (ind + 1) >= len(self.cx):
                break  # not exceed goal
            ind += 1

        return ind, Lf

    def pure_pursuit_steer_control(self, x, y, theta, v, pind):
            
            ind, Lf = self.search_target_waypoint(x, y, v)

            if pind >= ind:
                ind = pind

            if ind < len(self.cx):
                tx = self.cx[ind]
                ty = self.cy[ind]
            else:  # toward goal
                tx = self.cx[-1]
                ty = self.cy[-1]
                ind = len(self.cx) - 1

            alpha = math.atan2(ty - y, tx - x) - theta
            delta = math.atan2(2.0 * self.wheelbase * math.sin(alpha) / Lf, 1.0)

            return delta, ind


class stanley():
    def __init__(self, track_dict):

        self.k = track_dict['k']
        self.max_steer = track_dict['max_steer']
        self.l_front = track_dict['l_front'] 
    
    def record_waypoints(self, cx, cy, cyaw):
        #Initialise waypoints for planner
        self.cx=cx
        self.cy=cy
        self.cyaw = cyaw
            
    def stanley_control(self, state, last_target_idx):
        """
        Stanley steering control.

        :param state: (State object)
        :param cx: ([float])
        :param cy: ([float])
        :param cyaw: ([float])
        :param last_target_idx: (int)
        :return: (float, int)
        """

        x=state[0]
        y=state[1]
        v=state[3]
        yaw=state[4]

        current_target_idx, error_front_axle = self.calc_target_index(state)

        if last_target_idx >= current_target_idx:
            current_target_idx = last_target_idx

        # theta_e corrects the heading error
        theta_e = self.normalize_angle(self.cyaw[current_target_idx] - yaw)
        # theta_d corrects the cross track error
        theta_d = np.arctan2(self.k * error_front_axle, v)
        # Steering control
        delta = theta_e + theta_d

        return delta, current_target_idx


    def normalize_angle(self, angle):
        """
        Normalize an angle to [-pi, pi].

        :param angle: (float)
        :return: (float) Angle in radian in [-pi, pi]
        """
        while angle > np.pi:
            angle -= 2.0 * np.pi

        while angle < -np.pi:
            angle += 2.0 * np.pi

        return angle


    def calc_target_index(self, state):
        """
        Compute index in the trajectory list of the target.

        :param state: (State object)
        :param cx: [float]
        :param cy: [float]
        :return: (int, float)
        """
        x=state[0]
        y=state[1]
        yaw=state[4]

        # Calc front axle position
        fx = x + self.l_front * np.cos(yaw)
        fy = y + self.l_front * np.sin(yaw)

        # Search nearest point index
        dx = [fx - icx for icx in self.cx]
        dy = [fy - icy for icy in self.cy]
        d = np.hypot(dx, dy)
        target_idx = np.argmin(d)

        # Project RMS error onto front axle vector
        front_axle_vec = [-np.cos(yaw + np.pi / 2), -np.sin(yaw + np.pi / 2)]
        error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

        return target_idx, error_front_axle
