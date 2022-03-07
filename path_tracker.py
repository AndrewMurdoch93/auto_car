import math
import numpy as np
import functions
from numba import njit

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


class local_path_tracker():
    '''
    More complex pure pursuit for multiple waypoint/ path following
    '''
    
    def __init__(self, track_dict):
        self.cx = []
        self.cy = []
        self.old_nearest_point_index = None
        self.wheelbase=track_dict['wheelbase']
        self.k = track_dict['k']
        self.Lfc = track_dict['Lfc']

    def record_waypoints(self, cx, cy):
        #Initialise waypoints for planner
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_waypoint(self, x, y, v):
        
        #If there is no previous nearest point - at the start
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
