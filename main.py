from audioop import avg
import numpy as np
import agent
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


class trainingLoop():
   def __init__(self, main_dict, agent_dict, env_dict):
      
      self.main_dict = main_dict
      self.agent_dict = agent_dict
      self.env_dict = env_dict
      
      self.agent_name = main_dict['name']
      self.max_episodes = main_dict['max_episodes']
      self.comment = main_dict['comment']

      #Initialise file names for saving data
      self.train_results_file_name = 'train_results/' + self.agent_name
      self.environment_name = "environments/" + self.agent_name
      self.train_parameters_name = 'train_parameters/' + self.agent_name
      self.results_file_name = 'test_results/' + self.agent_name
      self.replay_episode_name = 'replay_episodes/' + self.agent_name


   def train(self):
      
      #Initialising environment
      self.env_dict['name'] = self.main_dict['name']
      self.env = environment(self.env_dict, start_condition=[])
      self.agent_dict['name'] = self.main_dict['name']
      self.agent_dict['input_dims'] = len(self.env.observation)
      self.agent_dict['n_actions'] = self.env_dict['n_actions']
      
      self.agent = agent.agent(self.agent_dict, new_agent=True)
      print("Agent = ", self.agent_name)

      outfile = open('environments/' + self.agent_name, 'wb')
      pickle.dump(self.env_dict, outfile)
      outfile.close()

      outfile = open('train_parameters/' + self.agent_name, 'wb')
      pickle.dump(self.main_dict, outfile)
      outfile.close()

      scores = []
      progress = []
      times = []

      for episode in range(self.max_episodes):
         
         self.env.reset(save_history=False)                #Reset the environment every episode
         obs = self.env.observation      #Records starting state
         done = False
         score = 0                       #Initialise score counter for every episode
         start_time = time.time()
         
         #For every planning time step in the episode
         while not done:
               
            action = self.agent.choose_action(obs)    #Select an action
            next_obs, reward, done = self.env.take_action(action) #Environment executes action
            score += reward 
            self.agent.store_transition(obs, action, reward, next_obs, done)
            self.agent.learn()  #Learn using state transition history
            obs = next_obs  #Update the state
      
         end_time = time.time()
         times.append(end_time-start_time)

         scores.append(score)
         avg_score = np.mean(scores[-100:])

         progress.append(self.env.progress)
         avg_progress = np.mean(progress[-100:])

         self.agent.decrease_epsilon()

         if episode%1000==0 and episode!=0:

               self.agent.save_agent()

               outfile=open(self.train_results_file_name, 'wb')
               pickle.dump(scores, outfile)
               pickle.dump(progress, outfile)
               outfile.close()


         if episode%10==0:
            print(f"{'Episode':8s} {episode:5.0f} {'| Score':8s} {score:6.2f} {'| Progress':12s} {self.env.progress:3.2f} {'| Average score':15s} {avg_score:6.2f} {'| Average progress':18s} {avg_progress:3.2f} {'| Epsilon':9s} {self.agent.epsilon:.2f}")
   
      
      self.agent.save_agent()

      outfile=open(self.train_results_file_name, 'wb')
      pickle.dump(scores, outfile)
      pickle.dump(progress, outfile)
      outfile.close()


   def test(self, n_episodes, detect_issues):

      self.env = environment(self.env_dict, start_condition=[])
      
      action_history = []

      infile = open('agents/' + self.agent_name + '_hyper_parameters', 'rb')
      self.agent_dict = pickle.load(infile)
      infile.close()

      self.agent = agent.agent(self.agent_dict, new_agent=False)

      test_progress = []
      test_score = []

      for episode in range(n_episodes):

         self.env.reset(save_history=True)
         action_history = []
         obs = self.env.observation
         done = False
         score=0

         while not done:
            action = self.agent.choose_action(obs)
            action_history.append(action)
            
            next_obs, reward, done = self.env.take_action(action)
            score += reward
            obs = next_obs
            
            test_progress.append(self.env.progress)
            test_score.append(score)
            
         if episode%50==0:
            print('Test episode', episode, '| Progress = %.2f' % self.env.progress, '| Score = %.2f' % score)

         if detect_issues==True and (self.env.progress>0.05 or self.env.progress<-0.05):
            print('Simulation is broken')
            print('Progress = ', self.env.progress)

            outfile = open(self.replay_episode_name, 'wb')
            pickle.dump(action_history, outfile)
            pickle.dump(self.env.initial_condition_dict, outfile)
            outfile.close()
            break
         
      outfile=open(self.results_file_name, 'wb')
      pickle.dump(test_score, outfile)
      pickle.dump(test_progress, outfile)
      outfile.close()

             
if __name__=='__main__':
   
   agent_name = 'progress_10'

   
   main_dict = {'name': agent_name, 'max_episodes':5000, 'comment':''}

   agent_dict = {'gamma':0.99, 'epsilon':1, 'eps_end':0.01, 'eps_dec':1e-3, 'lr':0.001, 'batch_size':64, 'max_mem_size':100000}

   env_dict = {'sim_conf': functions.load_config(sys.path[0], "config"), 'save_history': False, 'map_name': 'circle'
            , 'max_steps': 1000, 'local_path': False, 'waypoint_strategy': 'local'
            , 'reward_signal': [0, -1, 0, -1, -0.001, 10, 0, 0, 0], 'n_actions': 8, 'control_steps': 20 
            , 'display': False} 
 
   

   #a = trainingLoop(main_dict, agent_dict, env_dict)
   #a.train()
   #a.test(n_episodes=1000, detect_issues=False)

   
   
   
   #display_results.density_plot_progress(['bench_agent'])
   #display_results.density_plot_progress(['progress_01', 'progress_05','progress_1'])
   #display_results.density_plot_progress(['progress_1', 'angle_01', 'angle_05'])
   #display_results.density_plot_progress(['progress_1', 'velocity_01', 'velocity_1'])
   #display_results.density_plot_progress(['progress_1', 'dist_01'])
   

   #display_results.display_train_parameters(agent_name=agent_name)
   #display_results.learning_curve_score(agent_name=agent_name, show_average=True, show_median=True)
   #display_results.learning_curve_progress(agent_name=agent_name, show_average=True, show_median=True)
   #display_results.agent_score_statistics(agent_name=agent_name)
   #display_results.agent_progress_statistics(agent_name=agent_name)
   #display_results.histogram_score(agent_name=agent_name)
   #display_results.histogram_progress(agent_name=agent_name)
   display_results.display_moving_agent(agent_name=agent_name, load_history=False)
   #display_results.display_path(agent_name=agent_name, load_history=False)





