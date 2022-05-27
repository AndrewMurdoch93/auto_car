from tracemalloc import start
import numpy as np
import math
import functions
import sys
import matplotlib.pyplot as plt
from matplotlib import image
from PIL import Image
import path_tracker
import pickle
from itertools import chain
import random
import vehicle_model
import time
from numba import njit
from numba import int32, int64, float32, float64,bool_    
from numba.experimental import jitclass
import numba
import mapping
import cubic_spline_planner

class environment():

    #def __init__(self, sim_conf, save_history, map_name, max_steps, local_path, waypoint_strategy, 
    #            reward_signal, num_actions, control_steps, agent_name, display, start_condition):
    def __init__(self, input_dict): 
        
        self.initial_condition_name = 'initial_conditions/' + input_dict['name']
        self.history_file_name = 'history/' + input_dict['name'] 

        self.input_dict = input_dict

        self.save_history = input_dict['save_history']
        self.sim_conf = input_dict['sim_conf']
        self.map_name = input_dict['map_name']
        self.max_steps = input_dict['max_steps']
        self.control_steps = input_dict['control_steps']
        self.display = input_dict['display']
        
        self.params = input_dict['car_params']
        self.reward_signal = input_dict['reward_signal']
        self.lidar_dict = input_dict['lidar_dict']
        self.action_space_dict = input_dict['action_space_dict']
        self.path_dict = input_dict['path_dict']
        
        self.local_path = self.path_dict['local_path']
        self.waypoint_strategy = self.path_dict['waypoint_strategy']
        self.wpt_arc = self.path_dict['wpt_arc']
        
        
        #Initialise car parameters
        self.mass = self.params['m']                         
        self.max_delta = self.params['s_max']               
        self.max_delta_dot = self.params['sv_max']          
        self.max_v = self.params['v_max']                        
        self.max_a = self.params['a_max']
        self.wheelbase = self.params['lf'] + self.params['lr'] 
        
        
        if self.local_path==True:
            self.path_strategy = self.path_dict['path_strategy']
            self.track_dict = self.path_dict['track_dict']
            
            if self.path_dict['control_strategy'] == 'pure_pursuit':
                self.track_dict['wheelbase'] = self.wheelbase 
                self.path_tracker = path_tracker.pure_pursuit_path(self.track_dict)

            if self.path_dict['control_strategy'] == 'stanley':
                self.path_tracker = path_tracker.stanley(self.track_dict)

        
        
        self.vel_select = self.action_space_dict['vel_select']
        self.R_range = self.action_space_dict['R_range']
        self.action_space = self.action_space_dict['action_space']
        if self.action_space == 'discrete':
            self.num_waypoints = self.action_space_dict['n_waypoints']
            self.num_vel = len(self.action_space_dict['vel_select'])
            self.num_R = len(self.action_space_dict['R_range']) 
            self.num_actions = self.num_waypoints*self.num_vel*self.num_R
        if self.action_space == 'continuous':
            self.num_actions = 1 + int(len(self.action_space_dict['vel_select'])>1) + int(len(self.action_space_dict['R_range'])>1)

        #simulation parameters
        self.dt = self.sim_conf.time_step

        #Initialise car parameters
        self.mass = self.params['m']                         
        self.max_delta = self.params['s_max']               
        self.max_delta_dot = self.params['sv_max']          
        self.max_v = self.params['v_max']                        
        self.max_a = self.params['a_max']
        self.wheelbase = self.params['lf'] + self.params['lr'] 
        
        #Initialise map and goal settings
        #self.occupancy_grid, self.map_height, c, self.map_res = functions.map_generator(map_name = self.map_name)
        
        track = mapping.map(self.map_name)
        self.occupancy_grid = track.occupancy_grid
        self.map_height = track.map_height
        self.map_width = track.map_width
        self.map_res = track.resolution

        self.s=2

        image_path = sys.path[0] + '/maps/' + input_dict['map_name'] + '.png'
        self.im = image.imread(image_path)
        
        track.find_centerline()
        self.goal_x = track.centerline[:,0]
        self.goal_y = track.centerline[:,1]
        self.goals=[]
        self.max_goals_reached=False
        for x,y in zip(self.goal_x, self.goal_y):
            self.goals.append([x, y])

        #self.goal_x, self.goal_y, self.rx, self.ry, self.ryaw, self.rk, self.d = functions.generate_circle_goals()
        #self.goal_x, self.goal_y, self.rx, self.ry, self.ryaw, self.rk, self.d = functions.generate_berlin_goals()
        
        self.rx, self.ry, self.ryaw, self.rk, self.d = cubic_spline_planner.calc_spline_course(self.goal_x, self.goal_y)

        #Car sensors - lidar
        if self.lidar_dict['is_lidar']==True:
            lidar_res=self.lidar_dict['lidar_res']
            n_beams=self.lidar_dict['n_beams']
            max_range=self.lidar_dict['max_range']
            fov=self.lidar_dict['fov']
            
         
            self.lidar = functions.lidar_scan(lidar_res=lidar_res, n_beams=n_beams, max_range=max_range, fov=fov
                        , occupancy_grid=self.occupancy_grid, map_res=self.map_res, map_height=self.map_height)
        
        self.episode = 0
        #self.reset(self.save_history, start_condition=)



    def reset(self, save_history, start_condition, get_lap_time=False):
        self.episode+=1
        self.save_history=save_history
        self.start_condition = start_condition
        self.get_lap_time = get_lap_time
        
        #Inialise state variables
        if self.start_condition:
            self.x = self.start_condition['x']
            self.y = self.start_condition['y']
            self.v = self.start_condition['v']
            self.theta = self.start_condition['theta']
            self.delta = self.start_condition['delta']
            self.current_goal = self.start_condition['goal']
        else:
            self.x, self.y, self.theta, self.current_goal = functions.random_start(self.goal_x, self.goal_y, self.rx, self.ry, self.ryaw, self.rk, self.d)
            self.v = random.random()*self.vel_select[-1]
            #self.v = random.random()*7
            #self.v=0    
            #self.v = 20
            self.delta = 0
        
        self.theta_dot = 0      
        self.delta_dot = 0
        self.slip_angle = 0
        self.v_ref = self.v
        self.state = np.array([self.x, self.y, self.delta, self.v, self.theta, self.theta_dot, self.slip_angle])
        
        self.x_to_goal = self.goals[self.current_goal][0] - self.x
        self.y_to_goal = self.goals[self.current_goal][1] - self.y
        self.old_d_goal = np.linalg.norm(np.array([self.x_to_goal, self.y_to_goal]))
        self.new_d_goal = np.linalg.norm(np.array([self.x_to_goal, self.y_to_goal]))
        
        if self.lidar_dict['is_lidar']==True:
            self.lidar_dists, self.lidar_coords = self.lidar.get_scan(self.x, self.y, self.theta)
        
        self.current_progress = 0
        self.vel_par_line = 0
        self.dist_to_line = 0
        self.angle_to_line = 0

        #Initialise history
        self.pose_history = []
        self.action_history = []
        self.local_path_history = []
        self.goal_history = []
        self.observation_history = []
        self.reward_history = []
        self.progress_history = []
        self.closest_point_history = []
        self.waypoint_history = []
        self.lidar_dist_history = []
        self.lidar_coords_history = []
        self.action_step_history = []

        #Initialise flags
        self.max_goals_reached=False
        self.out_of_bounds = False
        self.max_steps_reached = False
        self.goal_reached = False
        self.collision=False
        self.backwards=False
        self.park=False
        
        #Progress indicators
        self.steps = 0
        self.goals_reached = 0
        self.progress = 0
        self.current_distance = 0
        #self.det_prg.search_index(self.x, self.y)
        self.old_closest_point = functions.find_closest_point(self.rx, self.ry, self.x, self.y)
        #Initialise state and observation vector 
        self.save_pose()

        if self.save_history==True:
            self.append_history_func()

        self.initial_condition_dict = {'x':self.x, 'y':self.y, 'theta':self.theta, 'v':self.v, 'delta':self.delta, 'goal': self.current_goal}

    def take_action(self, act):
        self.action_history.append(act)

        reward = 0
        done=False
        
        waypoint, wpt_angle, R, v_ref = self.convert_action_to_coord(strategy=self.waypoint_strategy, action=act)

        if self.local_path==False:
            for _ in range(self.control_steps):
                if self.display==True:
                    self.visualise(waypoint)
                
                if self.save_history==True:
                    self.waypoint_history.append(waypoint)
                    self.action_step_history.append(act)
                
                #delta_ref = path_tracker.pure_pursuit(self.wheelbase, waypoint, self.x, self.y, self.theta)
                
                if self.action_space=='discrete':
                    delta_ref = math.pi/16-(math.pi/8)*(act/(self.num_actions-1))         
                else:
                    delta_ref = (self.max_delta)*act[0]       

                #delta_dot, a = self.control_system(self.delta, delta_ref, self.v, v_ref)
                
                self.update_pose(delta_ref, v_ref)
                self.update_variables()
                self.steps += 1
                self.set_flags()
                reward += self.getReward()

                #print(reward)

                done = self.isEnd()
                
                if self.lidar_dict['is_lidar']==True:
                    self.lidar_dists, self.lidar_coords = self.lidar.get_scan(self.x, self.y, self.theta)

                self.save_pose()
                
                if self.save_history==True:
                    self.reward_history.append(reward)
                    self.append_history_func()
                
                if done==True:
                    break
            #self.save_state(waypoint, reward)
           
        else:
            
            cx, cy, cyaw = self.define_path(waypoint, wpt_angle, R)
            self.path_tracker.record_waypoints(cx, cy, cyaw)
            
            if self.path_dict['control_strategy'] == 'pure_pursuit':
                target_index, _ = self.path_tracker.search_target_waypoint(self.x, self.y, self.v)
            elif self.path_dict['control_strategy'] == 'stanley':
                target_index, _ = self.path_tracker.calc_target_index(self.state)

            lastIndex = len(cx)-1
            i=0
            while (lastIndex > target_index) and i<self.control_steps:
                if self.display==True:
                    self.visualise(waypoint)
               
                if self.save_history==True:
                    self.waypoint_history.append(waypoint)
                    self.action_step_history.append(act)
                
                if self.path_dict['control_strategy'] == 'pure_pursuit':
                    delta_ref, target_index = self.path_tracker.pure_pursuit_steer_control(self.x, self.y, self.theta, self.v, target_index)
                if self.path_dict['control_strategy'] == 'stanley':
                    delta_ref, target_idx = self.path_tracker.stanley_control(self.state, target_index)
                
                #delta_dot, a = self.control_system(self.delta, delta_ref, self.v, v_ref)
                self.update_pose(delta_ref, v_ref)
                self.update_variables()

                self.steps += 1

                self.local_path_history.append([cx, cy][:])
                self.set_flags()
                reward += self.getReward()
                
                if self.lidar_dict['is_lidar']==True:
                    self.lidar_dists, self.lidar_coords = self.lidar.get_scan(self.x, self.y, self.theta)

                self.save_pose()

                if self.save_history==True:
                    self.reward_history.append(reward)
                    self.append_history_func()
                
                #plt.plot(self.rx, self.ry)
                #plt.plot(self.x, self.y, 'x')
                #plt.plot(self.rx[self.det_prg.old_nearest_point_index], self.ry[self.det_prg.old_nearest_point_index], 'x')
                #plt.show()
                
                i+=1
                done = self.isEnd()
                if done == True:
                    break
            
            if done == True:
                #print(self.steps)
                if self.save_history == True:
                    self.save_history_func()
                
        return self.observation, reward, done
    
    def define_path(self, waypoint,  wpt_angle , R):
        if self.path_strategy == 'linear':
            cx = (((np.arange(0.1, 1, 0.01))*(waypoint[0] - self.x)) + self.x).tolist()
            cy = ((np.arange(0.1, 1, 0.01))*(waypoint[1] - self.y) + self.y)

        if self.path_strategy == 'circle':
            if np.pi/2-0.01 < wpt_angle < np.pi/2+0.01:
                cx = (((np.arange(0.1, 1, 0.01))*(waypoint[0] - self.x)) + self.x).tolist()
                cy = ((np.arange(0.1, 1, 0.01))*(waypoint[1] - self.y) + self.y)
            
            elif 0.01<wpt_angle<np.pi-0.01:
                L=R*np.sin(wpt_angle)/np.sin(np.pi-2*wpt_angle)
                a = self.theta-np.pi/2
                angles = np.linspace(0,np.pi-2*wpt_angle)
                x_arc = L*(1 - np.cos(angles))
                y_arc = L*np.sin(angles)
                d_arc = np.sqrt(np.power(x_arc[1:-1], 2) + np.power(y_arc[1:-1], 2))
                if wpt_angle<np.pi/2:
                    a_arc = np.arctan(np.true_divide(y_arc[1:-1], x_arc[1:-1]))
                if wpt_angle>np.pi/2:
                    a_arc = np.arctan(np.true_divide(y_arc[1:-1], x_arc[1:-1]))+np.pi
                cx = d_arc*np.cos(a_arc+a) + self.x
                cy = d_arc*np.sin(a_arc+a) + self.y

                cx = cx.flatten().tolist()
                cy = cy.flatten()
            
            else:
                L=R/2
                a = self.theta-np.pi/2
                angles = np.linspace(0,np.pi-2*wpt_angle)
                x_arc = R*(1 - np.cos(angles))
                y_arc = R*np.sin(angles)
                d_arc = np.sqrt(np.power(x_arc[1:-1], 2) + np.power(y_arc[1:-1], 2))
                if wpt_angle<np.pi/2:
                    a_arc = np.arctan(np.true_divide(y_arc[1:-1], x_arc[1:-1]))
                if wpt_angle>np.pi/2:
                    a_arc = np.arctan(np.true_divide(y_arc[1:-1], x_arc[1:-1]))+np.pi
                cx = d_arc*np.cos(a_arc+a) + self.x
                cy = d_arc*np.sin(a_arc+a) + self.y

                cx = cx.flatten().tolist()
                cy = cy.flatten()

            

            cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(cx, cy)
            
            #plt.plot(x_arc, y_arc, 'x')
            #plt.plot(cx, cy, 'o')
            #plt.axis('equal')
            #plt.show()

        return np.array(cx), np.array(cy), np.array(cyaw)



    def visualise(self, waypoint):
        
        current_goal = self.goals[self.current_goal]
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        plt.imshow(self.im, extent=(0,self.map_width,0,self.map_height))
        
        if self.local_path==True:
            plt.plot(self.path_tracker.cx, self.path_tracker.cy)
        
        plt.arrow(self.x, self.y, 0.5*math.cos(self.theta), 0.5*math.sin(self.theta), head_length=0.5, head_width=0.5, shape='full', ec='None', fc='blue')
        plt.arrow(self.x, self.y, 0.5*math.cos(self.theta+self.delta), 0.5*math.sin(self.theta+self.delta), head_length=0.5, head_width=0.5, shape='full',ec='None', fc='red')
        
        plt.plot(self.x, self.y, 'o')
        plt.plot(waypoint[0], waypoint[1], 'x')

        plt.plot([current_goal[0]-self.s, current_goal[0]+self.s, current_goal[0]+self.s, current_goal[0]-self.s, 
        current_goal[0]-self.s], [current_goal[1]-self.s, current_goal[1]-self.s, current_goal[1]+self.s, 
        current_goal[1]+self.s, current_goal[1]-self.s], 'r')

        if self.lidar_dict['is_lidar']==True:
            for coord in self.lidar_coords:
                plt.plot(coord[0], coord[1], 'xb')
            
        plt.plot(np.array(self.pose_history)[0:self.steps,0], np.array(self.pose_history)[0:self.steps,1])
        
        plt.plot(self.rx, self.ry)
        plt.plot(self.rx[self.old_closest_point], self.ry[self.old_closest_point], 'x')
        plt.xlabel('x coordinate')
        plt.ylabel('y coordinate')
        plt.xlim([0,self.map_width])
        plt.ylim([0,self.map_height])
        plt.title('Episode history')
        plt.pause(0.001)

    
    def save_initial_condition(self):
        
        outfile = open(self.initial_condition_name, 'wb')
        pickle.dump(self.initial_condition_dict, outfile)
        outfile.close()
    
    def save_pose(self):
        
        self.pose = [self.x, self.y, self.theta, self.delta, self.v]
        
        x_norm = self.x/self.map_width
        y_norm = self.y/self.map_height
        theta_norm = (self.theta)/(2*math.pi)
        v_norm = self.v/self.max_v
        v_ref_norm = self.v_ref/self.max_v
        
        #self.observation = [x_norm, y_norm, theta_norm]
        self.observation = [x_norm, y_norm, theta_norm, v_norm]
        
        #self.observation = []
        if self.lidar_dict['is_lidar']==True:
            #lidar_norm = np.array(self.lidar_dists)<0.5
            lidar_norm = np.array(self.lidar_dists)/self.lidar_dict['max_range']
            for n in lidar_norm:
                self.observation.append(n)
            pass


    
    def save_history_func(self):
        outfile = open(self.history_file_name, 'wb')
        pickle.dump(self.waypoint_history, outfile)
        pickle.dump(self.reward_history, outfile)
        pickle.dump(self.pose_history, outfile)
        pickle.dump(self.goal_history, outfile)
        pickle.dump(self.observation_history, outfile)
        pickle.dump(self.progress_history, outfile)
        pickle.dump(self.closest_point_history, outfile)
        pickle.dump(self.action_step_history, outfile)
        if self.lidar_dict['is_lidar']==True:
            pickle.dump(self.lidar_coords_history, outfile)
        if self.local_path == True:
            pickle.dump(self.local_path_history, outfile)
        outfile.close()

    
    def load_history_func(self):
        infile = open(self.history_file_name, 'rb')
        self.waypoint_history = pickle.load(infile)
        self.reward_history = pickle.load(infile)
        self.pose_history = pickle.load(infile)
        self.goal_history = pickle.load(infile)
        self.observation_history = pickle.load(infile)
        self.progress_history = pickle.load(infile)
        self.closest_point_history = pickle.load(infile)
        self.action_step_history = pickle.load(infile)
        if self.lidar_dict['is_lidar']==True:
            self.lidar_coords_history = pickle.load(infile)
        if self.local_path == True:
            self.local_path_history = pickle.load(infile)
        infile.close()
        
    
    def append_history_func(self):
        self.pose_history.append(self.pose[:])
        self.goal_history.append(self.goals[self.current_goal])
        self.observation_history.append(self.observation)
        self.progress_history.append(self.progress)
        if self.lidar_dict['is_lidar']==True:
            self.lidar_coords_history.append(self.lidar_coords)

          
    def convert_action_to_coord(self, strategy, action):
        #wpt = action%self.num_waypoints
        #print('wpt = ', wpt)
        
        if self.action_space=='discrete':
            wpt = action%self.num_waypoints
            R=self.R_range[-1]
            if strategy=='global':
                waypoint = [int((action+1)%3), int((action+1)/3)]

            if strategy=='local':
                waypoint_relative_angle = self.theta - self.wpt_arc +(2*self.wpt_arc)*(wpt/(self.num_waypoints-1))
                waypoint = [self.x + R*math.cos(waypoint_relative_angle), self.y + R*math.sin(waypoint_relative_angle)]
            
            if strategy == 'waypoint':
                waypoint = action

            i = int(action/self.num_waypoints)
            self.v_ref = self.vel_select[i]

        '''
        if self.v_ref>=1 and self.v_ref<=7:
            self.v_ref += self.vel_select[i]
        elif self.v_ref<1:
            self.v_ref = 1
        else:
            self.v_ref = 7
        '''
        '''
        if self.v_ref>=1 and self.v_ref<=7:
            self.v_ref = self.v + self.vel_select[i]
        elif self.v_ref<1:
            self.v_ref = 1
        else:
            self.v_ref = 7
        '''
        #print('v = ', self.v, 'v_ref = ',  self.v_ref)
        #v_ref = int(action/self.num_waypoints)*self.max_v

        if self.action_space=='continuous':
            #wpt_action = action[0]
            #vel_action = action[1]
            #R_action = action[2]
            
            wpt_action=action
            R = self.R_range[-1]

            waypoint_relative_angle = self.theta + self.wpt_arc*(wpt_action)
            waypoint = [self.x + R*math.cos(waypoint_relative_angle), self.y + R*math.sin(waypoint_relative_angle)]
            self.v_ref=self.vel_select[-1]

        wpt_angle =  waypoint_relative_angle-(self.theta-np.pi/2) #angle in cars reference frame

        return waypoint,  wpt_angle, R, self.v_ref

    def set_flags(self):
        
        if (self.x>self.goals[self.current_goal][0]-self.s and self.x<self.goals[self.current_goal][0]+self.s) and (self.y>self.goals[self.current_goal][1]-self.s and self.y<self.goals[self.current_goal][1]+self.s):
            self.current_goal = (self.current_goal+1)%(len(self.goals)-1)
            self.goal_reached = True
            self.goals_reached+=1
            #self.progress = self.goals_reached/len(self.goals)
            #self.progress = self.det_prg.progress(self.x,self.y)
            
        elif self.goal_reached == True:
            self.goal_reached = False
        
        if self.x>self.map_width or self.x<0 or self.y>self.map_height or self.y<0:        
            self.out_of_bounds=True
            
        if self.steps >= self.max_steps:
            self.max_steps_reached=True

        if self.goals_reached==(len(self.goals)):
            self.max_goals_reached=True
        
        if self.steps>1 and self.v<0.1:
            self.park=True
        else:
            self.park=False

        if functions.occupied_cell(self.x, self.y, self.occupancy_grid, self.map_res, self.map_height)==True:
            self.collision=True
   

            
        
    def getReward(self):

        if self.goal_reached==True:
            return self.reward_signal['goal_reached']

        if  self.out_of_bounds==True:
            return self.reward_signal['out_of_bounds']

        elif self.max_steps_reached==True:
            return self.reward_signal['max_steps']

        elif self.collision==True:
            return self.reward_signal['collision']

        elif self.backwards==True:
            return self.reward_signal['backwards']

        elif self.park==True:
            reward=self.reward_signal['park']

        else:
            reward=0
            reward+=self.reward_signal['time_step']
            reward+=self.current_progress * self.reward_signal['progress']
            reward+=self.current_distance *  self.reward_signal['distance']
            #reward += self.vel_par_line * (1/self.max_v) * self.reward_signal[6]
            #reward += np.abs(self.angle_to_line) * (1/(np.pi)) * self.reward_signal[7]
            #reward += self.dist_to_line * self.reward_signal[8]
            #reward += self.progress * self.reward_signal[5]

        return reward
    
            
    def isEnd(self):
        if 2*self.max_goals_reached==True:
            return True
        elif self.out_of_bounds==True:       
            return True
        elif self.max_steps_reached==True:
            #print('max steps reached')
            return True
        elif self.collision==True:
            #print('collide')
            return True
        elif self.backwards==True:
            return True
        elif self.park==True:
            return True
        elif self.get_lap_time==True and self.progress>=1:
            return True
        else:
            return False
    

    def update_variables(self):
        self.x_to_goal = self.goals[self.current_goal][0] - self.x
        self.y_to_goal = self.goals[self.current_goal][1] - self.y
        
        self.new_d_goal = np.linalg.norm(np.array([self.x_to_goal, self.y_to_goal]))
        self.new_angle = math.atan2(self.y-15, self.x-15)%(2*math.pi)

        new_closest_point = functions.find_closest_point(self.rx, self.ry, self.x, self.y)
        self.closest_point_history.append(new_closest_point)

        #Find angle between vehicle and line (vehicle heading error)
        self.angle_to_line = np.abs(functions.sub_angles_complex(self.ryaw[new_closest_point], self.theta))

        #Find vehicle progress along line
        point_dif = new_closest_point-self.old_closest_point
        
        #if point_dif>=0 or point_dif<-100:
        #    self.current_progress = ((new_closest_point-self.old_closest_point)%len(self.rx))/len(self.rx)
        #else:
        #    self.current_progress = -((self.old_closest_point-new_closest_point)%len(self.rx))/len(self.rx)   
        
        if point_dif==0:
            self.current_progress = 0
        
        elif 0<point_dif<int(len(self.rx)/2):   #forward progress
            self.current_progress = new_closest_point-self.old_closest_point
        
        elif point_dif<-int(len(self.rx)/2):    #Crossing start location going forward
            self.current_progress = (len(self.rx)-np.abs(self.old_closest_point))+new_closest_point

        elif -int(len(self.rx)/2)<point_dif<0:  #Backwards
            self.current_progress = new_closest_point-self.old_closest_point
            #self.backwards=True

        elif point_dif>=int(len(self.rx)/2):    #Crossing start location going backwards
            self.current_progress = -(self.old_closest_point+(np.abs(len(self.rx)-new_closest_point)))
            #self.backwards=True

        self.current_progress/=len(self.rx)

        self.progress += self.current_progress
        
        self.current_distance = self.current_progress*self.d[-1]
        

        '''
        print('old point = ', self.old_closest_point)
        print('new point = ', new_closest_point)
        print('point difference = ', point_dif)
        print('current progress = ', self.current_progress)
        print('Progress = ', self.progress)
        print('------------------')
        '''

        #Velocity component along line
        self.vel_par_line = self.v * np.cos(self.angle_to_line)
        #Distance to nearest point 
        self.dist_to_line = np.hypot(self.x-self.rx[new_closest_point], self.y-self.ry[new_closest_point])
        self.old_closest_point = new_closest_point

    def update_pose(self, steer, vel):
         # steering angle velocity input to steering velocity acceleration input
        accl, sv = vehicle_model.pid(vel, steer, self.state[3], self.state[2], self.params['sv_max'], self.params['a_max'], self.params['v_max'], self.params['v_min'])
        
        # update physics, get RHS of diff'eq
        f = vehicle_model.vehicle_dynamics_st(
            self.state,
            np.array([sv, accl]),
            self.params['mu'],
            self.params['C_Sf'],
            self.params['C_Sr'],
            self.params['lf'],
            self.params['lr'],
            self.params['h'],
            self.params['m'],
            self.params['I'],
            self.params['s_min'],
            self.params['s_max'],
            self.params['sv_min'],
            self.params['sv_max'],
            self.params['v_switch'],
            self.params['a_max'],
            self.params['v_min'],
            self.params['v_max'])

        # update state
        self.state = self.state + f * self.dt

        self.x = self.state[0]
        self.y = self.state[1]
        self.theta = self.state[4]%(2*np.pi)
        self.v = self.state[3]
        self.delta = self.state[2]

        #print(self.theta)
        #print(self.delta)

def test_environment():
    
    agent_name = 'rainbow_distance'
    replay_episode_name = 'replay_episodes/' + agent_name
    
    infile=open(replay_episode_name, 'rb')
    action_history = pickle.load(infile)
    initial_condition = pickle.load(infile)
    infile.close()
    
    infile = open('environments/' + agent_name, 'rb')
    env_dict = pickle.load(infile)
    infile.close()
    env_dict['display']=True
    

    
    #env_dict = {'sim_conf': functions.load_config(sys.path[0], "config"), 'save_history': False, 'map_name': 'circle'
    #        , 'max_steps': 500, 'local_path': True, 'waypoint_strategy': 'local', 'wpt_arc': np.pi/2, 'action_space': 'discrete'
    #        , 'reward_signal': {'goal_reached':0, 'out_of_bounds':-1, 'max_steps':0, 'collision':-1, 'backwards':-1, 'park':-0.5, 'time_step':-0.005, 'progress':0, 'distance':0.3}
    #        , 'n_waypoints': 11, 'vel_select':[7], 'control_steps': 15, 'display': True, 'R':3, 'track_dict':{'k':0.1, 'Lfc':1}
    #        , 'lidar_dict': {'is_lidar':True, 'lidar_res':0.1, 'n_beams':8, 'max_range':20, 'fov':np.pi} } 
    
    car_params =   {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145
                , 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2
                , 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}
    
    reward_signal = {'goal_reached':0, 'out_of_bounds':-1, 'max_steps':0, 'collision':-1, 'backwards':-1, 'park':-0.5, 'time_step':-0.005, 'progress':0, 'distance':0.3}    
    
    action_space_dict = {'action_space': 'continuous', 'vel_select':[7], 'R_range':[3]}
    #action_space_dict = {'action_space': 'discrete', 'n_waypoints': 10, 'vel_select':[7], 'R_range':[3]}
    
    path_dict = {'local_path':True, 'waypoint_strategy':'local', 'wpt_arc':np.pi/2}
     
    if path_dict['local_path'] == True: #True or false
        path_dict['path_strategy'] = 'circle' #circle or linear
        path_dict['control_strategy'] = 'stanley' #pure_pursuit or stanley
        
        if path_dict['control_strategy'] == 'pure_pursuit':
            path_dict['track_dict'] = {'k':0.1, 'Lfc':1}
        if path_dict['control_strategy'] == 'stanley':
            path_dict['track_dict'] = {'l_front': car_params['lf'], 'k':1, 'max_steer':car_params['s_max']}
    
    lidar_dict = {'is_lidar':True, 'lidar_res':0.1, 'n_beams':8, 'max_range':20, 'fov':np.pi}
    
    env_dict = {'sim_conf': functions.load_config(sys.path[0], "config")
        , 'save_history': False
        , 'map_name': 'circle'
        , 'max_steps': 500
        , 'control_steps': 15
        , 'display': True
        , 'car_params':car_params
        , 'reward_signal':reward_signal
        , 'lidar_dict':lidar_dict
        , 'action_space_dict':action_space_dict
        , 'path_dict': path_dict
        } 


    env_dict['name'] = 'test'
    #initial_condition = {'x':8.18, 'y':26.24, 'v':4, 'delta':0, 'theta':np.pi, 'goal':1}
    initial_condition = {'x':15, 'y':5, 'v':7, 'delta':0, 'theta':0, 'goal':1}
    #initial_condition = {'x':8, 'y':3, 'v':7, 'delta':0, 'theta':np.pi, 'goal':1}
    #initial_condition = []



    env = environment(env_dict)
    env.reset(save_history=True, start_condition=initial_condition, get_lap_time=False)
    
    done=False
    
    action_history = np.ones(20)*-1

    score=0
    i=0
    while done==False:
        
        action = action_history[i]
        i+=1
        print('action = ', action)
        #action = env.goals[env.current_goal]
        state, reward, done = env.take_action(action)
        score+=reward

    print('score = ', score)
    #env.save_initial_condition()
    
if __name__=='__main__':
    test_environment()


        
