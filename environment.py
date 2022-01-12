import numpy as np
import math
import functions
import sys
import matplotlib.pyplot as plt

class environment():

    def __init__(self, sim_conf):
        self.sim_conf = sim_conf
        self.local_path=False
        self.reset()

    def reset(self):
             
        self.num_actions = 8
        
        self.goals = [[1,0.5], [2,1], [1,2], [0.5,1]]
        self.current_goal = 0
        self.s=0.2
        
        self.x = (np.random.rand(1)*0.2)[0]
        self.y = (np.random.rand(1)*0.2)[0]
        self.theta = (np.random.rand(1)*(math.pi/4))[0] 
        
        self.v = 0    
        self.delta = 0
        self.x_to_goal = self.goals[self.current_goal][0] - self.x
        self.y_to_goal = self.goals[self.current_goal][1] - self.y
        self.state = [self.x, self.y, self.theta, self.delta, self.v, self.x_to_goal, self.y_to_goal]
        self.state_history = []
        self.action_history = []
        self.local_path_history = []
        self.goal_history = []
        self.ind_history = []
        
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

        self.upper_bound = 2.5
        self.lower_bound = -0.5
        self.right_bound = 2.5
        self.left_bound = -0.5
        
        #self.goal1 = [2,0.5]
        #self.goal1_reached = False
        #self.goal2 = [2,2]
        #self.goal2_reached = False
        
        self.out_of_bounds = False
        self.max_steps_reached = False
        
        self.steps = 0
        self.max_steps = 300

    def take_action(self, act):
        reward = 0
        done=False
        
        waypoint = self.convert_action_to_coord(strategy='local', action=act)
        v_ref = 2

        if self.local_path==False:
            for _ in range(self.control_steps):
                delta_ref = self.pure_pursuit(waypoint)
                delta_dot, a = self.control_system(self.delta, delta_ref, self.v, v_ref)
                self.update_kinematic_state(a, delta_dot)
                self.steps += 1
                self.save_state(waypoint)
                reward += self.getReward()
                done = self.isEnd()
                if done==True:
                    break

            return self.state, reward, done

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

                self.save_state(waypoint)
                self.local_path_history.append([cx, cy][:])
                
                reward += self.getReward()
                done = self.isEnd()
                if done == True:
                    break
   
                i+=1
            return self.state, reward, done
    
    def save_state(self, waypoint):
        
        self.state = [self.x, self.y, self.theta, self.delta, self.v, self.x_to_goal, self.y_to_goal]
        self.state_history.append(self.state[:])
        self.action_history.append(waypoint)
        self.goal_history.append(self.goals[self.current_goal])
    
    
    def convert_action_to_coord(self, strategy, action):
        if strategy=='global':
            waypoint = [int((action+1)%3), int((action+1)/3)]

        if strategy=='local':
            waypoint_relative_angle = self.theta+math.pi/2-math.pi*(action/8)
            waypoint = [self.x + math.cos(waypoint_relative_angle), self.y + math.sin(waypoint_relative_angle)]
        
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
            if (ind+1)>=len(self.cx):
                    print('index error 3')

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
            self.current_goal+=1
            return 1

        #if (self.x>self.goal1[0]-self.s and self.x<self.goal1[0]+self.s) and (self.y>self.goal1[1]-self.s and self.y<self.goal1[1]+self.s) and self.goal1_reached==False:
        #    self.goal1_reached=True
        #    return 1
        
        #elif (self.x>self.goal2[0]-self.s and self.x<self.goal2[0]+self.s) and (self.y>self.goal2[1]-self.s and self.y<self.goal2[1]+self.s) and self.goal1_reached==True and self.goal2_reached==False:
        #    self.goal2_reached=True
        #    return 1
            
        elif self.x>self.right_bound or self.x<self.left_bound or self.y>self.upper_bound or self.y<self.lower_bound:        
            self.out_of_bounds=True
            return -1
        
        elif self.steps >= self.max_steps:
            self.max_steps_reached=True
            return 0
        
        else:
            return -0.0001
            
    
    def isEnd(self):
        if self.current_goal==(len(self.goals)):
            return True
        elif self.out_of_bounds==True:       
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
    n = 1
    state_history = []
    while n<=1:

        #action = int(input())
        action = 7
        state, reward, done = env.take_action(action)
        #print(f"state = {state}")
        #print(f"reward =  {reward}")
        #print(f"done = {done}")
        #print(f'Position = {env.position}')
        #print(f'State = {env.state}')

        if done==True:
            n+=1
            print(f"n = {n}")

            plt.plot([x[0] for x in env.state_history], [x[1] for x in env.state_history])
            plt.plot(env.local_path_history[0][0], env.local_path_history[0][1])
            plt.legend(["car trajectory", "local path"])
            
            plt.xlim([-0.5,2.5])
            plt.ylim([-0.5, 2.5])
            plt.show()
            pass

#test_environment()

        
