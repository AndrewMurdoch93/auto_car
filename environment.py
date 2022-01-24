import numpy as np
import math
import functions
import sys
import matplotlib.pyplot as plt
from matplotlib import image
from PIL import Image

class environment():

    def __init__(self, sim_conf):
        self.sim_conf = sim_conf
        self.local_path=False
        self.reset()

    def reset(self, save_history=False):
        
        self.occupancy_grid, self.map_height, self.map_width, self.res = functions.map_generator(map_name='circle')
        goal_x, goal_y, rx, ry, ryaw, rk, s = functions.generate_circle_goals()
        self.goals=[]
        
        for x,y in zip(goal_x,goal_y):
            self.goals.append([x, y])

        start_x, start_y, start_theta, next_goal = functions.random_start(goal_x, goal_y, rx, ry, ryaw, rk, s)
        
        self.x = start_x
        self.y = start_y
        self.theta = start_theta
        self.current_goal = next_goal

        self.save_history = save_history   
        self.num_actions = 8

        self.s=2
        
        self.v = 0    
        self.delta = 0
        self.x_to_goal = self.goals[self.current_goal][0] - self.x
        self.y_to_goal = self.goals[self.current_goal][1] - self.y
        self.old_d_goal = np.linalg.norm(np.array([self.x_to_goal, self.y_to_goal]))
        self.new_d_goal = np.linalg.norm(np.array([self.x_to_goal, self.y_to_goal]))
        
        self.angle = math.atan2(self.y-15, self.x-15)
        self.new_angle = math.atan2(self.y-15, self.x-15)

        self.theta_dot = 0      #car rate of change of heading
        self.delta_dot = 0
        
        #self.prev_loc = 0
        self.dt = self.sim_conf.time_step
        self.control_steps = self.sim_conf.control_steps
        #Initialise car parameters from 
        self.mass = self.sim_conf.m                              #car mass
        self.max_delta = self.sim_conf.max_delta                 #car maximum steering angle
        self.max_delta_dot = self.sim_conf.max_delta_dot             #car maximum maximum rate of steering angle change
        self.max_v = self.sim_conf.max_v                         
        self.max_a = self.sim_conf.max_a
        self.wheelbase = self.sim_conf.l_f + self.sim_conf.l_r        #Distance between rear and front wheels  

        self.state = [self.x, self.y, self.theta, self.delta, self.v]
        self.observation = [self.x/self.map_width, self.y/self.map_height, (self.theta+math.pi)/(2*math.pi)]#, (self.x_to_goal+0.5*self.map_width)/self.map_width, (self.y_to_goal+0.5*self.map_height)/self.map_height]
        
        self.state_history = []
        self.action_history = []
        self.local_path_history = []
        self.goal_history = []
        self.ind_history = []
        self.observation_history = []
        self.reward_history = []

        self.state_history.append(self.state)
        self.observation_history.append(self.observation)
        
        self.out_of_bounds = False
        self.max_steps_reached = False
        self.goal_reached = False
        self.collision=False
        
        self.steps = 0
        self.max_steps = 1500

        self.goals_reached = 0
        self.progress = 0

    def take_action(self, act):
        reward = 0
        done=False
        
        waypoint = self.convert_action_to_coord(strategy='local', action=act)
        v_ref = 7

        if self.local_path==False:
            for _ in range(self.control_steps):
                delta_ref = self.pure_pursuit(waypoint)
                delta_dot, a = self.control_system(self.delta, delta_ref, self.v, v_ref)
                self.update_kinematic_state(a, delta_dot)
                self.steps += 1
                reward += self.getReward() 
                done = self.isEnd()
                self.save_state(waypoint, reward)
                if done==True:
                    break
            #self.save_state(waypoint, reward)
           

        else:
            cx = (((np.arange(0.1, 1, 0.01))*(waypoint[0] - self.x)) + self.x).tolist()
            cy = ((np.arange(0.1, 1, 0.01))*(waypoint[1] - self.y) + self.y)
            self.record_waypoints(cx, cy)
            target_index, _ = self.search_target_waypoint(self.x, self.y, self.v)

            lastIndex = len(cx)-1
            i=0
            while (lastIndex > target_index) and i<10:

                delta_ref, target_index = self.pure_pursuit_steer_control(self.x, self.y, self.theta, self.v, target_index)
                delta_dot, a = self.control_system(self.delta, delta_ref, self.v, v_ref)
                self.update_kinematic_state(a, delta_dot)
                
                self.steps += 1

                self.save_state(waypoint, reward)
                self.local_path_history.append([cx, cy][:])
                
                reward += self.getReward()
                done = self.isEnd()
                if done == True:
                    break
   
                i+=1
        #print(reward)
        return self.observation, reward, done
    
    def save_state(self, waypoint, reward):
        
        self.state = [self.x, self.y, self.theta, self.delta, self.v]
        self.observation = [self.x/self.map_width, self.y/self.map_height, (self.theta+math.pi)/(2*math.pi)] #, (self.x_to_goal+0.5*self.map_width)/self.map_width, (self.y_to_goal+0.5*self.map_height)/self.map_height]

        if self.save_history==True:
            self.state_history.append(self.state[:])
            self.action_history.append(waypoint)
            self.goal_history.append(self.goals[self.current_goal])
            self.observation_history.append(self.observation)
            self.reward_history.append(reward)
    
    
    def convert_action_to_coord(self, strategy, action):
        if strategy=='global':
            waypoint = [int((action+1)%3), int((action+1)/3)]

        if strategy=='local':
            waypoint_relative_angle = self.theta+math.pi/2-math.pi*(action/8)
            waypoint = [self.x + 4*math.cos(waypoint_relative_angle), self.y + 4*math.sin(waypoint_relative_angle)]
        
        if strategy == 'waypoint':
            waypoint = action

        return waypoint

    def record_waypoints(self, cx, cy):
        #Initialise waypoints for planner
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_waypoint(self, x, y, v):
        k = 0.1
        Lfc = 0.2
        
        #If there is no previous nearest point - at the start
        if self.old_nearest_point_index is None:
            #Get distances to every point
            dx = [x - icx for icx in self.cx]
            dy = [y - icy for icy in self.cy]
            d = np.hypot(dx, dy)    
            ind = np.argmin(d)      #Get nearest point
            self.ind_history.append(ind)
            self.old_nearest_point_index = ind  #Set previous nearest point to nearest point
        else:   #If there exists a previous nearest point - after the start
            #Search for closest waypoint after ind
            ind = self.old_nearest_point_index  
            self.ind_history.append(ind)
        
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

        Lf = k * v + Lfc  # update look ahead distance

        # search look ahead target point index
        while Lf > functions.distance_between_points(self.cx[ind], x, self.cy[ind], y):
            if (ind + 1) >= len(self.cx):
                break  # not exceed goal
            ind += 1

        return ind, Lf


    def pure_pursuit_steer_control(self, x, y, theta, v, pind):
        '''
        More complex pure pursuit for multiple waypoint/ path following
        '''
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


    def pure_pursuit(self, waypoint):
        '''
        A simple pure pursuit implementation for single waypoint following
        Based on formulas in the youtube video:
        https://www.youtube.com/watch?v=zMdoLO4kRKg&t=73s
        '''
        
        ld = math.sqrt((waypoint[0]-self.x)**2 + (waypoint[1]-self.y)**2)
        alpha = math.atan2(waypoint[1] - self.y, waypoint[0] - self.x) - self.theta
        delta_ref = math.atan2(2.0 * self.wheelbase * math.sin(alpha) / ld, 1.0)

        return delta_ref


    def getReward(self):

        if (self.x>self.goals[self.current_goal][0]-self.s and self.x<self.goals[self.current_goal][0]+self.s) and (self.y>self.goals[self.current_goal][1]-self.s and self.y<self.goals[self.current_goal][1]+self.s):
            self.current_goal = (self.current_goal+1)%(len(self.goals)-1)
            self.goal_reached = True
            self.goals_reached+=1
            self.progress = self.goals_reached/len(self.goals)
            #print(self.progress)
            return 1
            
        if self.x>self.map_width or self.x<0 or self.y>self.map_height or self.y<0:        
            self.out_of_bounds=True
            return -1
        
        elif self.steps >= self.max_steps:
            self.max_steps_reached=True
            return -1
        
        elif self.goal_reached == True:
            self.goal_reached=False
            return 0
        
        elif functions.detect_collision(self.occupancy_grid, self.x, self.y, self.res):
            self.collision=True
            return -1

        else:
           #goal_progress = self.old_d_goal-self.new_d_goal
           #return goal_progress*0.1
           return -0.001
    
            
    def isEnd(self):
        if self.goals_reached==(len(self.goals)):
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

        self.old_d_goal = self.new_d_goal
        self.old_angle = self.new_angle%(2*math.pi-0.05)

        #Update (rear axle) position
        self.x = self.x + self.v * np.cos(self.theta) * self.dt
        self.y = self.y + self.v * np.sin(self.theta) * self.dt
        
        #Update 
        self.x_to_goal = self.goals[self.current_goal][0] - self.x
        self.y_to_goal = self.goals[self.current_goal][1] - self.y
        
        self.new_d_goal = np.linalg.norm(np.array([self.x_to_goal, self.y_to_goal]))
        self.new_angle = math.atan2(self.y-15, self.x-15)%(2*math.pi)

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

    env = environment(functions.load_config(sys.path[0], "config"))
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

            for sh, ah, gh, rh in zip(env.state_history, env.action_history, env.goal_history, env.reward_history):
                plt.cla()
                # Stop the simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
                #plt.image
                plt.imshow(im, extent=(0,30,0,30))
                plt.arrow(sh[0], sh[1], 0.5*math.cos(sh[2]), 0.1*math.sin(sh[2]), head_length=0.5, head_width=0.5, shape='full', ec='None', fc='blue')
                plt.arrow(sh[0], sh[1], 0.5*math.cos(sh[2]+sh[3]), 0.1*math.sin(sh[2]+sh[3]), head_length=0.5, head_width=0.5, shape='full',ec='None', fc='red')
                plt.plot(sh[0], sh[1], 'o')
                plt.plot(ah[0], ah[1], 'x')
                plt.plot([gh[0]-env.s, gh[0]+env.s, gh[0]+env.s, gh[0]-env.s, gh[0]-env.s], [gh[1]-env.s, gh[1]-env.s, gh[1]+env.s, gh[1]+env.s, gh[1]-env.s], 'r')
                #plt.legend(["position", "waypoint", "goal area", "heading", "steering angle"])
                plt.xlabel('x coordinate')
                plt.ylabel('y coordinate')
                plt.xlim([0,30])
                plt.ylim([0,30])
                #plt.grid(True)
                plt.title('Episode history')
                plt.pause(0.001)

#test_environment()

        
