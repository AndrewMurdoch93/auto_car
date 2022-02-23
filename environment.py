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

class environment():

    #def __init__(self, sim_conf, save_history, map_name, max_steps, local_path, waypoint_strategy, 
    #            reward_signal, num_actions, control_steps, agent_name, display, start_condition):
    def __init__(self, input_dict, start_condition): 
        
        self.initial_condition_name = 'initial_conditions/' + input_dict['name']
        self.history_file_name = 'test_history/' + input_dict['name'] 

        self.save_history = input_dict['save_history']
        self.sim_conf = input_dict['sim_conf']
        self.map_name = input_dict['map_name']
        self.max_steps = input_dict['max_steps']
        self.local_path = input_dict['local_path']
        self.waypoint_strategy = input_dict['waypoint_strategy']
        self.reward_signal = input_dict['reward_signal']
        self.num_actions = input_dict['n_actions']
        self.control_steps = input_dict['control_steps']
        self.display=input_dict['display']
        self.R=input_dict['R']
        self.track_dict = input_dict['track_dict']
        self.lidar_dict = input_dict['lidar_dict']

        self.start_condition = start_condition
        
        #simulation parameters
        self.dt = self.sim_conf.time_step
        
        #Initialise car parameters
        self.mass = self.sim_conf.m                              
        self.max_delta = self.sim_conf.max_delta                 
        self.max_delta_dot = self.sim_conf.max_delta_dot            
        self.max_v = self.sim_conf.max_v                         
        self.max_a = self.sim_conf.max_a
        self.wheelbase = self.sim_conf.l_f + self.sim_conf.l_r 
        
        self.track_dict['wheelbase'] = self.wheelbase 

        if self.local_path == True:
            self.path_tracker = path_tracker.local_path_tracker(self.track_dict)
        
        #Initialise map and goal settings
        self.occupancy_grid, self.map_height, self.map_width, self.map_res = functions.map_generator(map_name = self.map_name)
        self.s=2

        image_path = sys.path[0] + '/maps/' + 'circle' + '.png'
        self.im = image.imread(image_path)
        
        self.goal_x, self.goal_y, self.rx, self.ry, self.ryaw, self.rk, self.d = functions.generate_circle_goals()
        self.goals=[]
        self.max_goals_reached=False

        for x,y in zip(self.goal_x, self.goal_y):
            self.goals.append([x, y])
        
        #Car sensors - lidar
        if self.lidar_dict['is_lidar']==True:
            self.lidar = functions.lidar_scan(self.lidar_dict, occupancy_grid=self.occupancy_grid, map_res=self.map_res, map_height=self.map_height)

        self.reset(self.save_history)



    def reset(self, save_history):
        self.save_history=save_history

        #Inialise state variables
        if self.start_condition:
            self.x = self.start_condition['x']
            self.y = self.start_condition['y']
            self.theta = self.start_condition['theta']
            self.current_goal = self.start_condition['goal']
        else:
            self.x, self.y, self.theta, self.current_goal = functions.random_start(self.goal_x, self.goal_y, self.rx, self.ry, self.ryaw, self.rk, self.d)
            
        self.v = 7    
        self.delta = 0
        self.theta_dot = 0      
        self.delta_dot = 0
        
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
        self.state_history = []
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

        #Initialise flags
        self.max_goals_reached=False
        self.out_of_bounds = False
        self.max_steps_reached = False
        self.goal_reached = False
        self.collision=False
        self.backwards=False
        
        #Progress indicators
        self.steps = 0
        self.goals_reached = 0
        self.progress = 0
        #self.det_prg.search_index(self.x, self.y)
        self.old_closest_point = functions.find_closest_point(self.rx, self.ry, self.x, self.y)

        #Initialise state and observation vector 
        self.save_state()

        if self.save_history==True:
            self.append_history_func()

        self.initial_condition_dict = {'x':self.x, 'y':self.y, 'theta':self.theta, 'v':self.v, 'delta':self.delta, 'goal': self.current_goal}

    def take_action(self, act):
        self.action_history.append(act)

        reward = 0
        done=False
        
        waypoint = self.convert_action_to_coord(strategy=self.waypoint_strategy, action=act)
        v_ref = 7


        if self.local_path==False:
            for _ in range(self.control_steps):
                if self.display==True:
                    self.visualise(waypoint)
                
                if self.save_history==True:
                    self.waypoint_history.append(waypoint)
                
                #delta_ref = path_tracker.pure_pursuit(self.wheelbase, waypoint, self.x, self.y, self.theta)
        
                delta_ref = math.pi/4-(math.pi/2)*(act/(self.num_actions-1))         
                
                delta_dot, a = self.control_system(self.delta, delta_ref, self.v, v_ref)
                self.update_kinematic_state(a, delta_dot)
                self.update_variables()
                self.steps += 1
                self.set_flags()
                reward += self.getReward() 
                done = self.isEnd()
                self.save_state()
                
                if self.save_history==True:
                    self.reward_history.append(reward)
                    self.append_history_func()
                
                if done==True:
                    break
            #self.save_state(waypoint, reward)
           
        else:
            cx = (((np.arange(0.1, 1, 0.01))*(waypoint[0] - self.x)) + self.x).tolist()
            cy = ((np.arange(0.1, 1, 0.01))*(waypoint[1] - self.y) + self.y)
            self.path_tracker.record_waypoints(cx, cy)
            target_index, _ = self.path_tracker.search_target_waypoint(self.x, self.y, self.v)

            lastIndex = len(cx)-1
            i=0
            while (lastIndex > target_index) and i<self.control_steps:
                if self.display==True:
                    self.visualise(waypoint)
               
                if self.save_history==True:
                    self.waypoint_history.append(waypoint)
                
                delta_ref, target_index = self.path_tracker.pure_pursuit_steer_control(self.x, self.y, self.theta, self.v, target_index)
                delta_dot, a = self.control_system(self.delta, delta_ref, self.v, v_ref)
                self.update_kinematic_state(a, delta_dot)
                self.update_variables()

                self.steps += 1

                self.local_path_history.append([cx, cy][:])
                self.set_flags()
                reward += self.getReward()

                self.save_state()

                if self.save_history==True:
                    self.reward_history.append(reward)
                    self.append_history_func()
                
                #plt.plot(self.rx, self.ry)
                #plt.plot(self.x, self.y, 'x')
                #plt.plot(self.rx[self.det_prg.old_nearest_point_index], self.ry[self.det_prg.old_nearest_point_index], 'x')
                #plt.show()

                done = self.isEnd()
                
                if done == True:
                    if self.save_history == True:
                        self.save_history_func()
                    break
   
                i+=1
        #print(reward)
        
        #reward = self.getReward()
        
        if self.lidar_dict['is_lidar']==True:
            self.lidar_dists, self.lidar_coords = self.lidar.get_scan(self.x, self.y, self.theta)
        
        return self.observation, reward, done
    

    def visualise(self, waypoint):
        
        current_goal = self.goals[self.current_goal]
        plt.cla()
        plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
        plt.imshow(self.im, extent=(0,30,0,30))
        
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
            
        plt.plot(np.array(self.state_history)[0:self.steps,0], np.array(self.state_history)[0:self.steps,1])
        
        plt.plot(self.rx, self.ry)
        plt.plot(self.rx[self.old_closest_point], self.ry[self.old_closest_point], 'x')
        plt.xlabel('x coordinate')
        plt.ylabel('y coordinate')
        plt.xlim([0,30])
        plt.ylim([0,30])
        plt.title('Episode history')
        plt.pause(0.001)

    
    def save_initial_condition(self):
        
        outfile = open(self.initial_condition_name, 'wb')
        pickle.dump(self.initial_condition_dict, outfile)
        outfile.close()
    
    def save_state(self):
        
        self.state = [self.x, self.y, self.theta, self.delta, self.v]
        
        #self.observation = [self.x/self.map_width, self.y/self.map_height, (self.delta+self.max_delta)/(2*self.max_delta), self.v/self.max_v, (self.theta+math.pi)/(2*math.pi), (self.x_to_goal+0.5*self.map_width)/self.map_width, (self.y_to_goal+0.5*self.map_height)/self.map_height]
        #self.observation = [self.x/self.map_width, self.y/self.map_height,(self.theta+math.pi)/(2*math.pi)]
        #self.observation = [self.x/self.map_width, self.y/self.map_height, (self.theta+math.pi)/(2*math.pi), (self.x_to_goal+0.5*self.map_width)/self.map_width, (self.y_to_goal+0.5*self.map_height)/self.map_height]
        x_norm = self.x/self.map_width
        y_norm = self.y/self.map_height
        theta_norm = (self.theta+math.pi)/(2*math.pi)
        #lidar_norm = np.array(self.lidar_dists)/self.lidar_dict['max_range']
        #lidar_norm = np.array(self.lidar_dists)<0.5
        self.observation = [x_norm, y_norm, theta_norm]
        
        #self.observation = []
        #for n in lidar_norm:
        #    self.observation.append(n)
        #pass


    
    def save_history_func(self):
        outfile = open(self.history_file_name, 'wb')
        pickle.dump(self.waypoint_history, outfile)
        pickle.dump(self.reward_history, outfile)
        pickle.dump(self.state_history, outfile)
        pickle.dump(self.goal_history, outfile)
        pickle.dump(self.observation_history, outfile)
        pickle.dump(self.progress_history, outfile)
        pickle.dump(self.closest_point_history, outfile)
        if self.lidar_dict['is_lidar']==True:
            pickle.dump(self.lidar_coords_history, outfile)
        if self.local_path == True:
            pickle.dump(self.local_path_history, outfile)
        outfile.close()

    
    def load_history_func(self):
        infile = open(self.history_file_name, 'rb')
        self.waypoint_history = pickle.load(infile)
        self.reward_history = pickle.load(infile)
        self.state_history = pickle.load(infile)
        self.goal_history = pickle.load(infile)
        self.observation_history = pickle.load(infile)
        self.progress_history = pickle.load(infile)
        self.closest_point_history = pickle.load(infile)
        if self.lidar_dict['is_lidar']==True:
            self.lidar_coords_history = pickle.load(infile)
        if self.local_path == True:
            self.local_path_history = pickle.load(infile)
        infile.close()
        
    
    def append_history_func(self):
        self.state_history.append(self.state[:])
        self.goal_history.append(self.goals[self.current_goal])
        self.observation_history.append(self.observation)
        self.progress_history.append(self.progress)
        if self.lidar_dict['is_lidar']==True:
            self.lidar_coords_history.append(self.lidar_coords)

          
    def convert_action_to_coord(self, strategy, action):
        if strategy=='global':
            waypoint = [int((action+1)%3), int((action+1)/3)]

        if strategy=='local':
            waypoint_relative_angle = self.theta+math.pi/4-(math.pi/2)*(action/(self.num_actions-1))
            waypoint = [self.x + self.R*math.cos(waypoint_relative_angle), self.y + self.R*math.sin(waypoint_relative_angle)]
        
        if strategy == 'waypoint':
            waypoint = action

        return waypoint

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
   
        if functions.occupied_cell(self.x, self.y, self.occupancy_grid, self.map_res, self.map_height)==True:
            self.collision=True
            
                
        
        
    def getReward(self):

        if self.goal_reached==True:
            return self.reward_signal[0]

        if  self.out_of_bounds==True:
            return self.reward_signal[1]

        elif self.max_steps_reached==True:
            return self.reward_signal[2]

        elif self.collision==True:
            return self.reward_signal[3]
        
        elif self.backwards==True:
            return -1
        else:
            reward=0

            #Time penalty
            reward+=self.reward_signal[4]
            #reward+=self.progress
            #return reward
            reward += self.current_progress * self.reward_signal[5]
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
            return True
        elif self.collision==True:
            return True
        elif self.backwards==True:
            return True
        else:
            return False
    
    def control_system(self, delta, delta_ref, v, v_ref):
        '''
        Generates acceleration and steering velocity commands to follow a reference velocity and steering angle
        Note: the controller gains are hand tuned in the fcn

        Args:
            d_ref: reference steering to be followed
            v_ref: the reference velocity to be followed
    

        Returns:
            a: acceleration
            delta_dot: the change in delta = steering velocity
        '''
        
        kp_delta = 40
        delta_dot = (delta_ref-delta)*kp_delta
        delta_dot = np.clip(delta_dot, -self.max_delta_dot, self.max_delta_dot)

        kp_a = 10
        a = (v_ref-v)*kp_a
        a = np.clip(a, -self.max_a, self.max_a)

        return delta_dot, a

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
            self.backwards=True

        elif point_dif>=int(len(self.rx)/2):    #Crossing start location going backwards
            self.current_progress = -(self.old_closest_point+(np.abs(len(self.rx)-new_closest_point)))
            self.backwards=True

        self.current_progress/=len(self.rx)

        self.progress += self.current_progress
        
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



        

    def update_kinematic_state(self, a, delta_dot):
        '''
        Updates the internal state of the vehicle according to the kinematic equations for a bicycle model
        see this link for a derivation of the equations:
        https://dingyan89.medium.com/simple-understanding-of-kinematic-bicycle-model-81cac6420357

        Car reference frame is the center of the rear axle (affects calculations significatntly)

        Args:
            a: scalar acceleration
            delta_dot: rate of change of steering angle
        '''

        #Update (rear axle) position
        self.x = self.x + self.v * np.cos(self.theta) * self.dt
        self.y = self.y + self.v * np.sin(self.theta) * self.dt

        #Update car heading angle
        self.theta_dot = (self.v / self.wheelbase) * np.tan(self.delta)  #rate of change of heading
        dtheta = self.theta_dot * self.dt   #change in heading angle
        self.theta = functions.add_angles_complex(self.theta, dtheta)   #new heading angle

        a = np.clip(a, -self.max_a, self.max_a) #Truncate maximum acceleration
        delta_dot = np.clip(delta_dot, -self.max_delta_dot, self.max_delta_dot) #truncate maximum steering angle rate of change  

        self.delta = self.delta + delta_dot * self.dt   #new steering angle
        self.v = self.v + a * self.dt     #new velocity

        self.delta = np.clip(self.delta, -self.max_delta, self.max_delta)    #truncate steering angle
        self.v = np.clip(self.v, -self.max_v, self.max_v)         #truncate velocity


def test_environment():
    
    agent_name = 'end_to_end'
    replay_episode_name = 'replay_episodes/' + agent_name
    
    infile=open(replay_episode_name, 'rb')
    action_history = pickle.load(infile)
    initial_condition = pickle.load(infile)
    infile.close()

    infile = open('environments/' + agent_name, 'rb')
    env_dict = pickle.load(infile)
    infile.close()
    env_dict['display']=True
    


    #env_dict = {'name':'test_agent', 'sim_conf': functions.load_config(sys.path[0], "config"), 'save_history': False, 'map_name': 'circle'
    #        , 'max_steps': 1000, 'local_path': False, 'waypoint_strategy': 'local'
    #        , 'reward_signal': [0, -1, 0, -1, -0.01, 10, 0, 0, 0], 'n_actions': 11, 'control_steps': 20
    #        , 'display': True, 'R':6, 'track_dict':{'k':0.1, 'Lfc':0.2}
    #        , 'lidar_dict': {'is_lidar':False, 'lidar_res':0.1, 'n_beams':10, 'max_range':20, 'fov':np.pi} } 
    #initial_condition={'x':15, 'y':5, 'theta':0, 'goal':0}
    
    env_dict['R']=6
    env_dict['track_dict'] = {'k':0.1, 'Lfc':0.2}
    env_dict['lidar_dict'] = {'is_lidar':False, 'lidar_res':0.1, 'n_beams':10, 'max_range':20, 'fov':np.pi}

    env = environment(env_dict, initial_condition)

    env.reset(save_history=True)
    done=False
    
    #action_history = [5,5,5,5,5,5]
    score=0
    i=0
    while done==False:
        
        action = action_history[i]
        i+=1
        print('action')
        #action = env.goals[env.current_goal]
        state, reward, done = env.take_action(action)
        score+=reward

    print('score = ', score)
    #env.save_initial_condition()
    
if __name__=='__main__':
    test_environment()


        
