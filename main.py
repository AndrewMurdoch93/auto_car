from audioop import avg
import numpy as np
import agent_dqn
import agent_reinforce
import agent_actor_critic
import agent_actor_critic_continuous
import agent_dueling_dqn
import agent_dueling_ddqn
import agent_rainbow
import agent_ddpg
from environment import environment
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
import functions
import sys
import matplotlib.patches as patches
import matplotlib.animation as animation
import math
from matplotlib import image
from PIL import Image
import time
import seaborn as sns
import display_results
import random

class trainingLoop():
   def __init__(self, main_dict, agent_dict, env_dict, load_agent):
      
      self.main_dict = main_dict
      self.agent_dict = agent_dict
      self.env_dict = env_dict
      self.load_agent = load_agent
      self.learning_method = main_dict['learning_method']
      
      self.agent_name = main_dict['name']
      self.max_episodes = main_dict['max_episodes']
      self.comment = main_dict['comment']

      #Initialise file names for saving data
      self.train_results_file_name = 'train_results/' + self.agent_name
      self.environment_name = 'environments/' + self.agent_name
      self.train_parameters_name = 'train_parameters/' + self.agent_name
      self.action_durations_name = 'action_durations/' + self.agent_name + '_train'


   def train(self):

      print('Training agent: ', self.main_dict['name'])
      
      self.env_dict['name'] = self.main_dict['name']

      #self.env = environment(self.env_dict, start_condition={'x':15,'y':5,'theta':0,'goal':0})
      self.env = environment(self.env_dict)
      self.env.reset(save_history=False, start_condition=[])
      
      self.agent_dict['name'] = self.main_dict['name']
      self.agent_dict['input_dims'] = len(self.env.observation)
      self.agent_dict['n_actions'] = self.env.num_actions

      self.old_ave_score = -np.inf
      self.old_ave_progress = -np.inf
      self.best_ave_progress = -np.inf
      self.best_ave_score = -np.inf

      if self.learning_method=='dqn':
         self.agent = agent_dqn.agent(self.agent_dict)
         self.agent_dict['epsilon'] = 1
      if self.learning_method=='reinforce':
         self.agent = agent_reinforce.PolicyGradientAgent(self.agent_dict)
      if self.learning_method=='actor_critic_sep':
         self.agent = agent_actor_critic.actor_critic_separated(self.agent_dict)
      if self.learning_method=='actor_critic_com':
         self.agent = agent_actor_critic.actor_critic_combined(self.agent_dict)
      if self.learning_method=='actor_critic_cont':
          self.agent = agent_actor_critic_continuous.agent_separate(self.agent_dict) 
      if self.learning_method == 'dueling_dqn':
         self.agent_dict['epsilon'] = 1
         self.agent = agent_dueling_dqn.agent(self.agent_dict)
      if self.learning_method == 'dueling_ddqn':
         self.agent_dict['epsilon'] = 1
         self.agent = agent_dueling_ddqn.agent(self.agent_dict)
      if self.learning_method == 'rainbow':
         self.agent_dict['epsilon'] = 1
         self.replay_beta_0=self.agent_dict['replay_beta_0']
         self.replay_beta=self.replay_beta_0
         self.agent = agent_rainbow.agent(self.agent_dict)
      if self.learning_method == 'ddpg':
         self.agent = agent_ddpg.agent(self.agent_dict)
      
      if self.load_agent:
         self.agent.load_weights(self.load_agent)

      outfile = open('environments/' + self.agent_name, 'wb')
      pickle.dump(self.env_dict, outfile)
      outfile.close()

      outfile = open('train_parameters/' + self.agent_name, 'wb')
      pickle.dump(self.main_dict, outfile)
      outfile.close()

      scores = []
      progress = []
      times = []
      steps = []
      durations = []

      for episode in range(self.max_episodes):
         
         self.env.reset(save_history=True, start_condition=[])  #Reset the environment every episode
         obs = self.env.observation      #Records starting state
         done = False
         score = 0                       #Initialise score counter for every episode
         start_time = time.time()
         
         #For every planning time step in the episode
         
         if self.learning_method == 'dqn':
            while not done: 
               action = self.agent.choose_action(obs)    #Select an action
               next_obs, reward, done = self.env.take_action(action) #Environment executes action
               score += reward 
               self.agent.store_transition(obs, action, reward, next_obs, done)
               self.agent.learn()  #Learn using state transition history
               obs = next_obs  #Update the state
            self.agent.decrease_epsilon()

         if self.learning_method == 'dueling_dqn' or self.learning_method == 'dueling_ddqn':
            while not done:
               action = self.agent.choose_action(obs)
               next_obs, reward, done = self.env.take_action(action)
               score += reward
               self.agent.store_transition(obs, action, reward, next_obs, int(done))
               self.agent.learn()
               obs = next_obs
            self.agent.decrease_epsilon()

         if self.learning_method == 'rainbow':
            while not done:
               
               start = time.time()
               action = self.agent.choose_action(obs)

               next_obs, reward, done = self.env.take_action(action)
               score += reward
               self.agent.store_transition(obs, action, reward, next_obs, int(done))
               self.agent.learn(self.replay_beta)
               obs = next_obs
               end = time.time()
               durations.append(end-start)
            
            self.agent.decrease_epsilon()
            self.replay_beta += (1-self.replay_beta)*(episode/self.max_episodes)
         
         if self.learning_method == 'reinforce':
            while not done:
               action = self.agent.choose_action(obs)
               next_obs, reward, done = self.env.take_action(action)
               self.agent.store_rewards(reward)
               obs = next_obs
               score += reward
            self.agent.learn()
         
         if self.learning_method == 'actor_critic_sep' or self.learning_method == 'actor_critic_com':
            while not done: 
               action = self.agent.choose_action(obs)    #Select an action
               next_obs, reward, done = self.env.take_action(action) #Environment executes action
               self.agent.learn(obs, reward, next_obs, done)  #Learn using state transition history
               obs = next_obs  #Update the state
               score+=reward
         
         if self.learning_method == 'actor_critic_cont':
              while not done:
                action = np.array(self.agent.choose_action(obs)).reshape((1,))
                next_obs, reward, done = self.env.take_action(action)
                self.agent.learn(obs, reward, next_obs, done)
                obs = next_obs
                score += reward
         
              
         if self.learning_method == 'ddpg':
            while not done: 
               
               start = time.time()

               action = self.agent.choose_action(obs)    #Select an action
               next_obs, reward, done = self.env.take_action(action) #Environment executes action
               self.agent.store_transition(obs, action, reward, next_obs, int(done))
               self.agent.learn()  #Learn using state transition history
               obs = next_obs  #Update the state
               score+=reward

               end = time.time()
               durations.append(end-start)
         
         
         end_time = time.time()
         times.append(end_time-start_time)

         scores.append(score)
         avg_score = np.mean(scores[-100:])

         progress.append(self.env.progress)
         avg_progress = np.mean(progress[-100:])

         steps.append(self.env.steps)

         if episode%500==0 and episode!=0:

            ave_progress = self.test_while_train(n_episodes=10)
            self.save_agent(ave_progress)

            outfile=open(self.train_results_file_name, 'wb')
            pickle.dump(scores, outfile)
            pickle.dump(progress, outfile)
            pickle.dump(times, outfile)
            pickle.dump(steps, outfile)
            outfile.close()


         if episode%10==0:
            if self.learning_method=='dqn' or self.learning_method=='dueling_dqn' or self.learning_method=='dueling_ddqn' or self.learning_method=='rainbow':
               print(f"{'Episode':8s} {episode:5.0f} {'| Score':8s} {score:6.2f} {'| Progress':12s} {self.env.progress:3.2f} {'| Average score':15s} {avg_score:6.2f} {'| Average progress':18s} {avg_progress:3.2f} {'| Epsilon':9s} {self.agent.epsilon:.2f}")
            if self.learning_method=='reinforce' or self.learning_method=='actor_critic_sep' or self.learning_method=='actor_critic_com' or self.learning_method=='actor_critic_cont' or self.learning_method=='ddpg':
               print(f"{'Episode':8s} {episode:5.0f} {'| Score':8s} {score:6.2f} {'| Progress':12s} {self.env.progress:3.2f} {'| Average score':15s} {avg_score:6.2f} {'| Average progress':18s} {avg_progress:3.2f}")
            
      ave_progress = self.test_while_train(n_episodes=10)
      self.save_agent(ave_progress)

      outfile=open(self.train_results_file_name, 'wb')
      pickle.dump(scores, outfile)
      pickle.dump(progress, outfile)
      pickle.dump(times, outfile)
      pickle.dump(steps, outfile)
      outfile.close()

      outfile=open(self.action_durations_name, 'wb')
      pickle.dump(durations, outfile)
      outfile.close()

   def test_while_train(self, n_episodes):
      
      print(f"{'Testing agent for '}{n_episodes}{' episodes'}")
      
      
      test_progress = []
      test_score = []
      
      for episode in range(n_episodes):
         self.env.reset(save_history=True, start_condition=[])
         obs = self.env.observation
         done = False
         score=0

         while not done:
            action = self.agent.choose_action(obs)
            next_obs, reward, done = self.env.take_action(action)
            score += reward
            obs = next_obs
         
         test_progress.append(self.env.progress)
         test_score.append(score)
         
      ave_progress = np.average(test_progress)
      ave_score = np.average(test_score)

      if ave_progress >= self.old_ave_progress:
         print(f"{'Average progress increased from '}{self.old_ave_progress:.2f}{' to '}{ave_progress:.2f}")
      else:
         print(f"{'Average progress decreased from '}{self.old_ave_progress:.2f}{' to '}{ave_progress:.2f}")

      if ave_score >= self.old_ave_score:
         print(f"{'Average score increased from '}{self.old_ave_score:.2f}{' to '}{ave_score:.2f}")
      else:
         print(f"{'Average score decreased from '}{self.old_ave_score:.2f}{' to '}{ave_score:.2f}")
      
      self.old_ave_progress = ave_progress 
      self.old_ave_score = ave_score

      return test_score
      

   def save_agent(self, test_score):
      
      ave_score = np.average(test_score)
      std = np.std(test_score)

      #if  ave_score >= (self.best_ave_score-std):
      if True:
         self.best_ave_score = ave_score
         self.agent.save_agent()
         print("Agent was saved")
      else:
         print("Agent was not saved")

   
def test(agent_name, n_episodes, detect_issues, initial_conditions):

   results_file_name = 'test_results/' + agent_name 
   replay_episode_name = 'replay_episodes/' + agent_name

   
   infile = open('environments/' + agent_name, 'rb')
   env_dict = pickle.load(infile)
   infile.close()

   if initial_conditions==True:
      start_condition_file_name = 'test_initial_condition/' + env_dict['map_name']
   else:
      start_condition_file_name = 'test_initial_condition/none' 
   
   infile = open(start_condition_file_name, 'rb')
   start_conditions = pickle.load(infile)
   infile.close()

   env_dict['max_steps'] = 1000

   #env = environment(env_dict, start_condition={'x':15,'y':5,'theta':0,'goal':0})
   env = environment(env_dict)
   env.reset(save_history=False, start_condition=[])
   
   action_history = []
   
   infile = open('agents/' + agent_name + '_hyper_parameters', 'rb')
   agent_dict = pickle.load(infile)
   infile.close()
   
   infile = open('train_parameters/' + agent_name, 'rb')
   main_dict = pickle.load(infile)
   infile.close()


   if main_dict['learning_method']=='dqn':
      agent_dict['epsilon'] = 0
      a = agent_dqn.agent(agent_dict)
   if main_dict['learning_method']=='reinforce':
      a = agent_reinforce.PolicyGradientAgent(agent_dict)
   if main_dict['learning_method']=='actor_critic_sep':
      a = agent_actor_critic.actor_critic_separated(agent_dict)
   if main_dict['learning_method']=='actor_critic_com':
      a = agent_actor_critic.actor_critic_combined(agent_dict)
   if main_dict['learning_method']=='actor_critic_cont':
      a = agent_actor_critic_continuous.agent_separate(agent_dict) 
   if main_dict['learning_method'] == 'dueling_dqn':
      agent_dict['epsilon'] = 0
      a = agent_dueling_dqn.agent(agent_dict)
   if main_dict['learning_method'] == 'dueling_ddqn':
      agent_dict['epsilon'] = 0
      a = agent_dueling_ddqn.agent(agent_dict)
   if main_dict['learning_method'] == 'rainbow':
      agent_dict['epsilon'] = 0
      a = agent_rainbow.agent(agent_dict)
   if main_dict['learning_method'] == 'ddpg':
      a = agent_ddpg.agent(agent_dict)
   
   a.load_weights(agent_name)

   test_progress = []
   test_score = []
   test_collision = []
   test_max_steps = []
   terminal_poses = []

   for episode in range(n_episodes):

      env.reset(save_history=True, start_condition=start_conditions[episode])
      action_history = []
      obs = env.observation
      done = False
      score = 0

      while not done:
         if main_dict['learning_method'] !='ddpg':
            action = a.choose_action(obs)
         elif main_dict['learning_method'] == 'ddpg':
            action = a.choose_greedy_action(obs)
         
         action_history.append(action)
         
         next_obs, reward, done = env.take_action(action)
         score += reward
         obs = next_obs
         
      test_progress.append(env.progress)
      test_score.append(score)
      test_collision.append(env.collision)
      test_max_steps.append(env.max_steps_reached)
      terminal_poses.append(env.pose)
         
      if episode%10==0:
         print('Test episode', episode, '| Progress = %.2f' % env.progress, '| Score = %.2f' % score)

      if detect_issues==True and (score<1):
         print('Stop condition met')
         print('Progress = ', env.progress)
         print('score = ', score)

         outfile = open(replay_episode_name, 'wb')
         pickle.dump(action_history, outfile)
         pickle.dump(env.initial_condition_dict, outfile)
         outfile.close()
         break
      
   outfile=open(results_file_name, 'wb')
   pickle.dump(test_score, outfile)
   pickle.dump(test_progress, outfile)
   pickle.dump(test_collision, outfile)
   pickle.dump(test_max_steps, outfile)
   pickle.dump(terminal_poses, outfile)
   outfile.close()


if __name__=='__main__':

   
   agent_name = 'rb_col_end_to_end_0'
   
   main_dict = {'name': agent_name, 'max_episodes':10000, 'learning_method': 'rainbow', 'comment': ''}

   agent_dqn_dict = {'gamma':0.99, 'epsilon':1, 'eps_end':0.01, 'eps_dec':1/1000, 'lr':0.001, 'batch_size':64, 'max_mem_size':500000, 
                  'fc1_dims': 64, 'fc2_dims': 64, 'fc3_dims':64}

   agent_dueling_dqn_dict = {'gamma':0.99, 'epsilon':1, 'eps_end':0.01, 'eps_dec':1/1000, 'alpha':0.001, 'batch_size':64, 'max_mem_size':500000, 
                           'replace':100, 'fc1_dims':64, 'fc2_dims':64}
   
   agent_dueling_ddqn_dict = {'gamma':0.99, 'epsilon':1, 'eps_end':0.01, 'eps_dec':1/1000, 'alpha':0.001, 'batch_size':64, 'max_mem_size':500000, 
                           'replace':100, 'fc1_dims':64, 'fc2_dims':64}
   
   agent_rainbow_dict = {'gamma':0.99, 'epsilon':1, 'eps_end':0.01, 'eps_dec':1/1000, 'alpha':0.001/4, 'batch_size':64, 'max_mem_size':500000, 
                           'replay_alpha':0.6, 'replay_beta_0':0.7, 'replace':100, 'fc1_dims':100, 'fc2_dims':100}
   
   agent_reinforce_dict = {'alpha':0.001, 'gamma':0.99, 'fc1_dims':256, 'fc2_dims':256}

   agent_actor_critic_sep_dict = {'gamma':0.99, 'actor_dict': {'alpha':0.00001, 'fc1_dims':2048, 'fc2_dims':512}, 'critic_dict': {'alpha': 0.00001, 'fc1_dims':2048, 'fc2_dims':512}}
   
   agent_actor_critic_com_dict = {'gamma':0.99, 'alpha':0.00001, 'fc1_dims':2048, 'fc2_dims':512}
   
   agent_actor_critic_cont_dict = {'gamma':0.99, 'alpha':0.000005, 'beta':0.00001, 'fc1_dims':256, 'fc2_dims':256}
   
   agent_ddpg_dict = {'alpha':0.000025, 'beta':0.00025, 'tau':0.001, 'gamma':0.99, 'max_size':1000000, 'layer1_size':400, 'layer2_size':300, 'batch_size':64}
   
   
   
   
   
   reward_signal = {'goal_reached':0, 'out_of_bounds':-1, 'max_steps':0, 'collision':-1, 'backwards':-1, 'park':-0.5, 'time_step':-0.005, 'progress':0, 'distance':0.3}    
   
   action_space_dict = {'action_space': 'continuous', 'vel_select':[1,7], 'R':[3]}
   
   action_space_dict = {'action_space': 'discrete', 'n_waypoints': 10, 'vel_select':[7], 'R':[3]}

   path_dict = {'local_path':True, 'waypoint_strategy':'local', 'wpt_arc':np.pi/2}

   if path_dict['local_path'] == True:
      path_dict['path_strategy'] = 'circle' #circle or linear
      path_dict['track_dict'] = {'k':0.1, 'Lfc':1}
   
   lidar_dict = {'is_lidar':True, 'lidar_res':0.1, 'n_beams':8, 'max_range':20, 'fov':np.pi}

   #env_dict = {'sim_conf': functions.load_config(sys.path[0], "config"), 'save_history': False, 'map_name': 'columbia_1'
   #         , 'max_steps': 500, 'local_path': False, 'waypoint_strategy': 'local', 'wpt_arc': np.pi/2, 'action_space': 'discrete'
   #         , 'reward_signal': {'goal_reached':0, 'out_of_bounds':-1, 'max_steps':0, 'collision':-1, 'backwards':-1, 'park':-0.5, 'time_step':-0.005, 'progress':0, 'distance':0.3}
   #         , 'n_waypoints': 10, 'vel_select':[7], 'control_steps': 15, 'display': False, 'R':3, 'track_dict':{'k':0.1, 'Lfc':1}
   #         , 'lidar_dict': {'is_lidar':True, 'lidar_res':0.1, 'n_beams':8, 'max_range':20, 'fov':np.pi} } 
   
   env_dict = {'sim_conf': functions.load_config(sys.path[0], "config"), 'save_history': False, 'map_name': 'columbia_1'
            , 'max_steps': 500
            , 'control_steps': 15
            , 'display': False
            , 'reward_signal':reward_signal
            , 'lidar_dict':lidar_dict
            , 'action_space_dict':  action_space_dict
            , 'path_dict': path_dict
            } 


   a = trainingLoop(main_dict, agent_rainbow_dict, env_dict, load_agent='')
   a.train()
   test(agent_name=agent_name, n_episodes=300, detect_issues=False, initial_conditions=False)
   
   
   
   '''
   agent_name = 'rb_col_fc_1'
   main_dict['name'] = agent_name
   agent_rainbow_dict['fc1_dims'] = 100
   agent_rainbow_dict['fc2_dims'] = 100
   a = trainingLoop(main_dict, agent_rainbow_dict, env_dict, load_agent='')
   a.train()
   test(agent_name=agent_name, n_episodes=300, detect_issues=False, initial_conditions=False)

   agent_name = 'rb_col_fc_2'
   main_dict['name'] = agent_name
   agent_rainbow_dict['fc1_dims'] = 200
   agent_rainbow_dict['fc2_dims'] = 200
   a = trainingLoop(main_dict, agent_rainbow_dict, env_dict, load_agent='')
   a.train()
   test(agent_name=agent_name, n_episodes=300, detect_issues=False, initial_conditions=False)
   
   agent_name = 'rb_col_R_0'
   main_dict['name'] = agent_name
   agent_rainbow_dict['fc1_dims'] = 64
   agent_rainbow_dict['fc2_dims'] = 64
   env_dict['R'] = 2
   a = trainingLoop(main_dict, agent_rainbow_dict, env_dict, load_agent='')
   a.train()
   test(agent_name=agent_name, n_episodes=300, detect_issues=False, initial_conditions=False)

   agent_name = 'rb_col_R_1'
   main_dict['name'] = agent_name
   env_dict['R'] = 5
   a = trainingLoop(main_dict, agent_rainbow_dict, env_dict, load_agent='')
   a.train()
   test(agent_name=agent_name, n_episodes=300, detect_issues=False, initial_conditions=False)

   agent_name = 'rb_col_R_2'
   main_dict['name'] = agent_name
   env_dict['R'] = 6
   a = trainingLoop(main_dict, agent_rainbow_dict, env_dict, load_agent='')
   a.train()
   test(agent_name=agent_name, n_episodes=300, detect_issues=False, initial_conditions=False)

   agent_name = 'rb_col_R_3'
   main_dict['name'] = agent_name
   env_dict['R'] = 6
   a = trainingLoop(main_dict, agent_rainbow_dict, env_dict, load_agent='')
   a.train()
   test(agent_name=agent_name, n_episodes=300, detect_issues=False, initial_conditions=False)
   
   agent_name = 'rb_col_cs_0'
   main_dict['name'] = agent_name
   env_dict['R'] = 3
   env_dict['control_steps'] = 10
   a = trainingLoop(main_dict, agent_rainbow_dict, env_dict, load_agent='')
   a.train()
   test(agent_name=agent_name, n_episodes=300, detect_issues=False, initial_conditions=False)

   agent_name = 'rb_col_cs_1'
   main_dict['name'] = agent_name
   env_dict['control_steps'] = 15
   a = trainingLoop(main_dict, agent_rainbow_dict, env_dict, load_agent='')
   a.train()
   test(agent_name=agent_name, n_episodes=300, detect_issues=False, initial_conditions=False)

   
   agent_name = 'rb_col_cs_2'
   main_dict['name'] = agent_name
   env_dict['control_steps'] = 25
   a = trainingLoop(main_dict, agent_rainbow_dict, env_dict, load_agent='')
   a.train()
   test(agent_name=agent_name, n_episodes=300, detect_issues=False, initial_conditions=False)

   agent_name = 'rb_col_rs_0'
   main_dict['name'] = agent_name
   env_dict['control_steps'] = 20
   env_dict['reward_signal']['distance'] = 0.05
   a = trainingLoop(main_dict, agent_rainbow_dict, env_dict, load_agent='')
   a.train()
   test(agent_name=agent_name, n_episodes=300, detect_issues=False, initial_conditions=False)


   agent_name = 'rb_col_rs_1'
   main_dict['name'] = agent_name
   env_dict['control_steps'] = 20
   env_dict['reward_signal']['distance'] = 0.1
   a = trainingLoop(main_dict, agent_rainbow_dict, env_dict, load_agent='')
   a.train()
   test(agent_name=agent_name, n_episodes=300, detect_issues=False, initial_conditions=False)

   agent_name = 'rb_col_rs_2'
   main_dict['name'] = agent_name
   env_dict['control_steps'] = 20
   env_dict['reward_signal']['distance'] = 0.3
   a = trainingLoop(main_dict, agent_rainbow_dict, env_dict, load_agent='')
   a.train()
   test(agent_name=agent_name, n_episodes=300, detect_issues=False, initial_conditions=False)

   agent_name = 'rb_col_rs_3'
   main_dict['name'] = agent_name
   env_dict['control_steps'] = 20
   env_dict['reward_signal']['distance'] = 0.5
   a = trainingLoop(main_dict, agent_rainbow_dict, env_dict, load_agent='')
   a.train()
   test(agent_name=agent_name, n_episodes=300, detect_issues=False, initial_conditions=False)

   '''

   #agent_name = 'ddpg_end_to_end'
   #main_dict['name'] = agent_name
   #main_dict['learning_method'] = 'ddpg'
   #main_dict['max_episodes'] = 2000
   #env_dict['action_space'] = 'continuous'
   #env_dict['n_waypoints'] = 1
   #a = trainingLoop(main_dict, agent_ddpg_dict, env_dict, load_agent='')
   #a.train()
   #test(agent_name=agent_name, n_episodes=300, detect_issues=False, initial_conditions=True)
   
   
   #agent_names = ['rb_col_standard', 'rb_col_fc_1', 'rb_col_fc_2']
   #legend_title = 'agent'
   #legend = ['rb_col_standard',  'rb_col_fc_1', 'rb_col_fc_2']
   #display_results.compare_learning_curves_progress(agent_names, legend, legend_title, show_average=True, show_median=False, xaxis='episodes')
   #display_results.compare_learning_curves_progress(agent_names, legend, legend_title, show_average=True, show_median=False, xaxis='steps')
   #display_results.compare_learning_curves_progress(agent_names, legend, legend_title, show_average=True, show_median=False, xaxis='times')
   #display_results.density_plot_progress(agent_names, legend, legend_title)

   #agent_names = [ 'rb_col_R_0', 'rb_col_standard',  'rb_col_R_1',  'rb_col_R_2',  'rb_col_R_3']
   #legend_title = 'agent'
   #legend = [ 'rb_col_R_0', 'rb_col_standard',  'rb_col_R_1',  'rb_col_R_2',  'rb_col_R_3']
   #display_results.compare_learning_curves_progress(agent_names, legend, legend_title, show_average=True, show_median=False, xaxis='episodes')
   #display_results.compare_learning_curves_progress(agent_names, legend, legend_title, show_average=True, show_median=False, xaxis='steps')
   #display_results.compare_learning_curves_progress(agent_names, legend, legend_title, show_average=True, show_median=False, xaxis='times')
   #display_results.density_plot_progress(agent_names, legend, legend_title)

   #agent_names = [ 'rb_col_cs_0', 'rb_col_cs_1', 'rb_col_standard',  'rb_col_cs_2']
   #legend_title = 'agent'
   #legend = ['rb_col_cs_0', 'rb_col_cs_1', 'rb_col_standard',  'rb_col_cs_2']
   #display_results.compare_learning_curves_progress(agent_names, legend, legend_title, show_average=True, show_median=False, xaxis='episodes')
   #display_results.compare_learning_curves_progress(agent_names, legend, legend_title, show_average=True, show_median=False, xaxis='steps')
   #display_results.compare_learning_curves_progress(agent_names, legend, legend_title, show_average=True, show_median=False, xaxis='times')
   #display_results.density_plot_progress(agent_names, legend, legend_title)

   #agent_names = [ 'rb_col_cs_0', 'rb_col_cs_1', 'rb_col_standard',  'rb_col_cs_2']
   #legend_title = 'agent'
   #legend = ['rb_col_cs_0', 'rb_col_cs_1', 'rb_col_standard',  'rb_col_cs_2']
   #display_results.compare_learning_curves_progress(agent_names, legend, legend_title, show_average=True, show_median=False, xaxis='episodes')
   #display_results.compare_learning_curves_progress(agent_names, legend, legend_title, show_average=True, show_median=False, xaxis='steps')
   #display_results.compare_learning_curves_progress(agent_names, legend, legend_title, show_average=True, show_median=False, xaxis='times')
   #display_results.density_plot_progress(agent_names, legend, legend_title)
   
   #agent_names = ['rb_col_standard', 'rb_col_rs_0', 'rb_col_rs_1', 'rb_col_rs_2',  'rb_col_rs_3']
   #legend_title = 'agent'
   #legend = ['rb_col_standard', 'rb_col_rs_0', 'rb_col_rs_1', 'rb_col_rs_2',  'rb_col_rs_3']
   #display_results.compare_learning_curves_progress(agent_names, legend, legend_title, show_average=True, show_median=False, xaxis='episodes')
   #display_results.compare_learning_curves_progress(agent_names, legend, legend_title, show_average=True, show_median=False, xaxis='steps')
   #display_results.compare_learning_curves_progress(agent_names, legend, legend_title, show_average=True, show_median=False, xaxis='times')
   #display_results.density_plot_progress(agent_names, legend, legend_title)

   #agent_name = 'rb_col_max_steps_1'
   #display_results.durations(agent_name, show_average=True, show_median=True)
   #display_results.display_collision_distribution(agent_name)
   #test(agent_name=agent_name, n_episodes=300, detect_issues=False, initial_conditions=False)
   #display_results.display_train_parameters(agent_name=agent_name)
   #display_results.agent_progress_statistics(agent_name=agent_name)
   #display_results.learning_curve_progress(agent_name=agent_name, show_average=True, show_median=True)
   #display_results.density_plot_progress([agent_name], legend=[''], legend_title='')
   #display_results.display_moving_agent(agent_name=agent_name, load_history=False)
   #display_results.display_path(agent_name=agent_name, load_history=False)
   
   #agent_names = ['rainbow_fc_0', 'rainbow_fc_1', 'rainbow_fc_2', 'rainbow_fc_3', 'rainbow_fc_4']
   #legend = ['fc_0', 'fc_1',  'fc_2',  'fc_3',  'fc_4']
   #legend_title = 'agent name'
   #display_results.density_plot_action_duration(agent_names, legend, legend_title)

   #agent_names = ['ddpg_fc_0', 'ddpg_fc_1', 'ddpg_fc_2', 'ddpg_fc_3', 'ddpg_fc_4']
   #legend = ['fc_0', 'fc_1',  'fc_2',  'fc_3',  'fc_4']
   #legend_title = 'agent name'
   #display_results.density_plot_action_duration(agent_names, legend, legend_title)

   #display_results.compare_learning_curves_progress([agent_name], [''], [''], xaxis='times')
   #display_results.display_train_parameters(agent_name=agent_name)
   #display_results.learning_curve_score(agent_name=agent_name, show_average=True, show_median=True)
   #display_results.agent_score_statistics(agent_name=agent_name)

   #display_results.histogram_score(agent_name=agent_name)
   #display_results.histogram_progress(agent_name=agent_name)

