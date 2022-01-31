import numpy as np
import math
import functions
import sys
import matplotlib.pyplot as plt
from matplotlib import image
from PIL import Image
import path_tracker

class environment():

    def __init__(self, sim_conf, save_history, map_name, max_steps, local_path, waypoint_strategy, reward_signal, num_actions, control_steps):
        
        self.save_history=save_history
        self.sim_conf = sim_conf
        self.map_name = map_name
        self.max_steps = max_steps
        self.local_path = local_path
        self.waypoint_strategy = waypoint_strategy
        self.reward_signal = reward_signal
        self.num_actions = num_actions
        self.control_steps = control_steps
        
        #simulation parameters
        self.dt = self.sim_conf.time_step
        
        #Initialise car parameters
        self.mass = self.sim_conf.m                              
        self.max_delta = self.sim_conf.max_delta                 
        self.max_delta_dot = self.sim_conf.max_delta_dot            
        self.max_v = self.sim_conf.max_v                         
        self.max_a = self.sim_conf.max_a
        self.wheelbase = self.sim_conf.l_f + self.sim_conf.l_r   

        if self.local_path == True:
            self.path_tracker = path_tracker.local_path_tracker(self.wheelbase)
        
        #Initialise map and goal settings
        self.occupancy_grid, self.map_height, self.map_width, self.res = functions.map_generator(map_name = self.map_name)
        self.s=2
        
        self.goal_x, self.goal_y, self.rx, self.ry, self.ryaw, self.rk, self.d = functions.generate_circle_goals()
        self.goals=[]
        self.max_goals_reached=False
        
        self.det_prg = functions.measure_progress(self.rx, self.ry)

        for x,y in zip(self.goal_x, self.goal_y):
            self.goals.append([x, y])

        self.reset(self.save_history)

    def reset(self, save_history):
        self.save_history=save_history

        #Inialise state variables
        self.x, self.y, self.theta, self.current_goal = functions.random_start(self.goal_x, self.goal_y, self.rx, self.ry, self.ryaw, self.rk, self.d)
        self.v = 0    
        self.delta = 0
        self.theta_dot = 0      
        self.delta_dot = 0
        
        self.x_to_goal = self.goals[self.current_goal][0] - self.x
        self.y_to_goal = self.goals[self.current_goal][1] - self.y
        self.old_d_goal = np.linalg.norm(np.array([self.x_to_goal, self.y_to_goal]))
        self.new_d_goal = np.linalg.norm(np.array([self.x_to_goal, self.y_to_goal]))

        #Initialise history
        self.state_history = []
        self.action_history = []
        self.local_path_history = []
        self.goal_history = []
        self.observation_history = []
        self.reward_history = []
        self.progress_history = []
        self.closest_point_history = []
        
        #Initialise flags
        self.max_goals_reached=False
        self.out_of_bounds = False
        self.max_steps_reached = False
        self.goal_reached = False
        self.collision=False
        
        #Progress indicators
        self.steps = 0
        self.goals_reached = 0
        self.progress = 0
        #self.det_prg.search_index(self.x, self.y)
        self.old_closest_point = functions.find_closest_point(self.rx, self.ry, self.x, self.y)

        #Initialise state and observation vector 
        self.save_state()

        if self.save_history==True:
            self.state_history.append(self.state[:])
            self.goal_history.append(self.goals[self.current_goal])
            self.observation_history.append(self.observation)
            self.progress_history.append(self.progress)
            self.closest_point_history.append(self.old_closest_point)


    def take_action(self, act):
        reward = 0
        done=False
        
        waypoint = self.convert_action_to_coord(strategy=self.waypoint_strategy, action=act)
        v_ref = 7

        if self.local_path==False:
            for _ in range(self.control_steps):
                delta_ref = path_tracker.pure_pursuit(self.wheelbase, waypoint, self.x, self.y, self.theta)
                delta_dot, a = self.control_system(self.delta, delta_ref, self.v, v_ref)
                self.update_kinematic_state(a, delta_dot)
                self.steps += 1
                self.set_flags()
                reward += self.getReward() 
                done = self.isEnd()
                self.save_state()
                self.save_history_func(waypoint, reward)
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
            while (lastIndex > target_index) and i<10:

                delta_ref, target_index = self.path_tracker.pure_pursuit_steer_control(self.x, self.y, self.theta, self.v, target_index)
                delta_dot, a = self.control_system(self.delta, delta_ref, self.v, v_ref)
                self.update_kinematic_state(a, delta_dot)
                
                self.steps += 1

                self.local_path_history.append([cx, cy][:])
                self.set_flags()
                reward += self.getReward()

                self.save_state()
                self.save_history_func(waypoint, reward)
                
                #plt.plot(self.rx, self.ry)
                #plt.plot(self.x, self.y, 'x')
                #plt.plot(self.rx[self.det_prg.old_nearest_point_index], self.ry[self.det_prg.old_nearest_point_index], 'x')
                #plt.show()

                done = self.isEnd()
                if done == True:
                    break
   
                i+=1
        #print(reward)
        return self.observation, reward, done
    
    def save_state(self):
        
        self.state = [self.x, self.y, self.theta, self.delta, self.v]
        #self.observation = [self.x/self.map_width, self.y/self.map_height, (self.delta+self.max_delta)/(2*self.max_delta), self.v/self.max_v, (self.theta+math.pi)/(2*math.pi), (self.x_to_goal+0.5*self.map_width)/self.map_width, (self.y_to_goal+0.5*self.map_height)/self.map_height]
        #self.observation = [self.x/self.map_width, self.y/self.map_height,(self.theta+math.pi)/(2*math.pi)]
        self.observation = [self.x/self.map_width, self.y/self.map_height, (self.theta+math.pi)/(2*math.pi), (self.x_to_goal+0.5*self.map_width)/self.map_width, (self.y_to_goal+0.5*self.map_height)/self.map_height]
    
    def save_history_func(self, waypoint, reward):

        if self.save_history==True:
            self.state_history.append(self.state[:])
            self.action_history.append(waypoint)
            self.goal_history.append(self.goals[self.current_goal])
            self.observation_history.append(self.observation)
            self.reward_history.append(reward)
            self.progress_history.append(self.progress)
            self.closest_point_history.append(self.old_closest_point)
    
    
    def convert_action_to_coord(self, strategy, action):
        if strategy=='global':
            waypoint = [int((action+1)%3), int((action+1)/3)]

        if strategy=='local':
            waypoint_relative_angle = self.theta+math.pi/2-math.pi*(action/8)
            waypoint = [self.x + 4*math.cos(waypoint_relative_angle), self.y + 4*math.sin(waypoint_relative_angle)]
        
        if strategy == 'waypoint':
            waypoint = action

        return waypoint

    def set_flags(self):
        if (self.x>self.goals[self.current_goal][0]-self.s and self.x<self.goals[self.current_goal][0]+self.s) and (self.y>self.goals[self.current_goal][1]-self.s and self.y<self.goals[self.current_goal][1]+self.s):
            self.current_goal = (self.current_goal+1)%(len(self.goals)-1)
            self.goal_reached = True
            self.goals_reached+=1
            #self.progress = self.goals_reached/len(self.goals)
            self.progress = self.det_prg.progress(self.x,self.y)
            
        elif self.goal_reached == True:
            self.goal_reached = False
        
        if self.x>self.map_width or self.x<0 or self.y>self.map_height or self.y<0:        
            self.out_of_bounds=True
            
        if self.steps >= self.max_steps:
            self.max_steps_reached=True

        if self.goals_reached==(len(self.goals)):
            self.max_goals_reached=True

        if functions.detect_collision(self.occupancy_grid, self.x, self.y, self.res):
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

        else:
            reward=0
            reward+=self.reward_signal[4]
            reward+=self.progress
            return reward
    
            
    def isEnd(self):
        if self.max_goals_reached==True:
            return True
        elif self.out_of_bounds==True:       
            return True
        elif self.max_steps_reached==True:
            return True
        elif self.collision==True:
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
        
        #Update 
        self.x_to_goal = self.goals[self.current_goal][0] - self.x
        self.y_to_goal = self.goals[self.current_goal][1] - self.y
        
        self.new_d_goal = np.linalg.norm(np.array([self.x_to_goal, self.y_to_goal]))
        self.new_angle = math.atan2(self.y-15, self.x-15)%(2*math.pi)

        new_closest_point = functions.find_closest_point(self.rx, self.ry, self.x, self.y)
        #self.progress = (new_closest_point-self.closest_point_history[0])/len(self.rx)
        self.current_progress = (new_closest_point - self.old_closest_point)/len(self.rx)
        self.old_closest_point = new_closest_point

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

    env = environment(sim_conf=functions.load_config(sys.path[0], "config"), save_history=True, map_name='circle', max_steps=1500, 
    local_path=True, waypoint_strategy='waypoint', reward_signal=[1,-1,-1,-1,-0.001], num_actions=8, control_steps=20)
    
    env.reset(save_history=True)
    done=False
    
    while done==False:

        action = env.goals[env.current_goal]
        state, reward, done = env.take_action(action)

        np.sum(np.array(env.reward_history))
        if done==True:
            
            image_path = sys.path[0] + '/maps/' + 'circle' + '.png'
            im = image.imread(image_path)
            plt.imshow(im, extent=(0,30,0,30))

            if env.local_path==False:

                for sh, ah, gh, rh, ph, cph in zip(env.state_history, env.action_history, env.goal_history, env.reward_history, env.progress_history, env.closest_point_history):
                    plt.cla()
                    # Stop the simulation with the esc key.
                    plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
                    #plt.image
                    plt.imshow(im, extent=(0,30,0,30))
                    plt.arrow(sh[0], sh[1], 0.5*math.cos(sh[2]), 0.5*math.sin(sh[2]), head_length=0.5, head_width=0.5, shape='full', ec='None', fc='blue')
                    plt.arrow(sh[0], sh[1], 0.5*math.cos(sh[2]+sh[3]), 0.5*math.sin(sh[2]+sh[3]), head_length=0.5, head_width=0.5, shape='full',ec='None', fc='red')
                    plt.plot(sh[0], sh[1], 'o')
                    plt.plot(ah[0], ah[1], 'x')
                    plt.plot([gh[0]-env.s, gh[0]+env.s, gh[0]+env.s, gh[0]-env.s, gh[0]-env.s], [gh[1]-env.s, gh[1]-env.s, gh[1]+env.s, gh[1]+env.s, gh[1]-env.s], 'r')
                    plt.plot(env.rx, env.ry)
                    plt.plot(env.rx[cph], env.ry[cph], 'x')
                    #plt.legend(["position", "waypoint", "goal area", "heading", "steering angle"])
                    plt.xlabel('x coordinate')
                    plt.ylabel('y coordinate')
                    plt.xlim([0,30])
                    plt.ylim([0,30])
                    #plt.grid(True)
                    plt.title('Episode history')

                    print('Progress = ', ph)
                    plt.pause(0.001)

            else:

                for sh, ah, gh, rh, lph, ph, cph in zip(env.state_history, env.action_history, env.goal_history, env.reward_history, env.local_path_history, env.progress_history, env.closest_point_history):
                    plt.cla()
                    # Stop the simulation with the esc key.
                    plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
                    #plt.image
                    plt.imshow(im, extent=(0,30,0,30))
                    plt.arrow(sh[0], sh[1], 0.5*math.cos(sh[2]), 0.5*math.sin(sh[2]), head_length=0.5, head_width=0.5, shape='full', ec='None', fc='blue')
                    plt.arrow(sh[0], sh[1], 0.5*math.cos(sh[2]+sh[3]), 0.5*math.sin(sh[2]+sh[3]), head_length=0.5, head_width=0.5, shape='full',ec='None', fc='red')
                    plt.plot(sh[0], sh[1], 'o')
                    plt.plot(ah[0], ah[1], 'x')
                    plt.plot([gh[0]-env.s, gh[0]+env.s, gh[0]+env.s, gh[0]-env.s, gh[0]-env.s], [gh[1]-env.s, gh[1]-env.s, gh[1]+env.s, gh[1]+env.s, gh[1]-env.s], 'r')
                    plt.plot(lph[0], lph[1])
                    plt.plot(env.rx, env.ry)
                    plt.plot(env.rx[cph], env.ry[cph], 'x')
                    #plt.legend(["position", "waypoint", "goal area", "heading", "steering angle"])
                    plt.xlabel('x coordinate')
                    plt.ylabel('y coordinate')
                    plt.xlim([0,30])
                    plt.ylim([0,30])
                    #plt.grid(True)
                    plt.title('Episode history')
                    #print('Progress = ', ph)
                    plt.pause(0.001)

test_environment()

        
