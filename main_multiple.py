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
import agent_td3
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
import os
import display_results_multiple



class trainingLoop():
   def __init__(self, main_dict, agent_dict, env_dict, load_agent):
      
      self.main_dict = main_dict
      self.agent_dict = agent_dict
      self.env_dict = env_dict
      self.load_agent = load_agent
      self.learning_method = main_dict['learning_method']
      self.runs = main_dict['runs']
      self.agent_name = main_dict['name']
      self.max_episodes = main_dict['max_episodes']
      self.max_steps = main_dict['max_steps']
      self.comment = main_dict['comment']

      #Initialise file names for saving data
      self.train_results_file_name = 'train_results/' + self.agent_name
      self.environment_name = 'environments/' + self.agent_name
      self.train_parameters_name = 'train_parameters/' + self.agent_name
      self.action_durations_name = 'action_durations/' + self.agent_name + '_train'

      self.parent_dir = os.path.dirname(os.path.abspath(__file__))
      self.agent_dir = self.parent_dir + '/agents/' + self.main_dict['name']

      self.agent_params_file = self.agent_dir + '/' + self.agent_name + '_params'

      try:
         os.mkdir(self.agent_dir)
      except OSError as error:
         print(error)
         print("Warning: Files will be overwritten")


   def train(self):

      print('Training agent: ', self.main_dict['name'])
      
      self.env_dict['name'] = self.main_dict['name']

      #self.env = environment(self.env_dict, start_condition={'x':15,'y':5,'theta':0,'goal':0})
      self.env = environment(self.env_dict)
      self.env.reset(save_history=False, start_condition=[], car_params=self.env_dict['car_params'], get_lap_time=False)
      
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
      if self.learning_method == 'td3':
         self.agent = agent_td3.agent(self.agent_dict)
      if self.load_agent:
         self.agent.load_weights(self.load_agent)

      outfile = open('environments/' + self.agent_name, 'wb')
      pickle.dump(self.env_dict, outfile)
      outfile.close()

      outfile = open('train_parameters/' + self.agent_name, 'wb')
      pickle.dump(self.main_dict, outfile)
      outfile.close()

      outfile = open(self.agent_params_file, 'wb')
      pickle.dump(self.agent_dict, outfile)
      outfile.close()

      scores = np.zeros([self.runs, self.max_episodes])
      progress = np.zeros([self.runs, self.max_episodes])
      times = np.zeros([self.runs, self.max_episodes])
      steps = np.zeros([self.runs, self.max_episodes])
      #durations = []

      for n in range(self.runs):
         
         #if self.learning_method == 'rainbow':
         #   self.agent_dict['epsilon'] = 1
         #   self.replay_beta_0=self.agent_dict['replay_beta_0']
         #   self.replay_beta=self.replay_beta_0
         self.agent = agent_td3.agent(self.agent_dict)
         
         car_params = self.env_dict['car_params']

         episode=0
         
         while np.sum(steps[n,:])<=self.max_steps and episode<=self.max_episodes:
            
            self.env.reset(save_history=True, start_condition=[], car_params=car_params, get_lap_time=False)  #Reset the environment every episode
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
                  
                  #start = time.time()
                  action = self.agent.choose_action(obs)

                  next_obs, reward, done = self.env.take_action(action)
                  score += reward
                  self.agent.store_transition(obs, action, reward, next_obs, int(done))
                  self.agent.learn(self.replay_beta)
                  obs = next_obs
                  #end = time.time()
                  #durations.append(end-start)
               
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
            
               
            if self.learning_method == 'ddpg' or self.learning_method == 'td3':
               while not done: 
                  action = self.agent.choose_action(obs)    #Select an action
                  next_obs, reward, done = self.env.take_action(action) #Environment executes action
                  self.agent.store_transition(obs, action, reward, next_obs, int(done))
                  self.agent.learn()  #Learn using state transition history
                  obs = next_obs  #Update the state
                  score+=reward
                  #self.agent.decrease_noise_factor()
             
            
            end_time = time.time()
            
            times[n, episode] = end_time-start_time

            scores[n, episode] = score
            #scores.append(score)
            avg_score = np.mean(scores[n, episode-100:episode])

            progress[n, episode] = self.env.progress
            #progress.append(self.env.progress)
            avg_progress = np.mean(progress[n, episode-100:episode])

            steps[n, episode] = self.env.steps

            if episode%100==0 and episode!=0:

               self.save_agent(n)

               outfile=open(self.train_results_file_name, 'wb')
               pickle.dump(scores, outfile)
               pickle.dump(progress, outfile)
               pickle.dump(times, outfile)
               pickle.dump(steps, outfile)
               outfile.close()


            if episode%10==0:
               if self.learning_method=='dqn' or self.learning_method=='dueling_dqn' or self.learning_method=='dueling_ddqn' or self.learning_method=='rainbow':
                  print(f"{'Run':3s} {n:2.0f} {'| Episode':8s} {episode:5.0f} {'| Score':8s} {score:6.2f} {'| Progress':12s} {self.env.progress:3.2f} {'| collision ':14s} {self.env.collision} {'| Average score':15s} {avg_score:6.2f} {'| Average progress':18s} {avg_progress:3.2f} {'| Epsilon':9s} {self.agent.epsilon:.2f}")
               if self.learning_method=='reinforce' or self.learning_method=='actor_critic_sep' or self.learning_method=='actor_critic_com' or self.learning_method=='actor_critic_cont' or self.learning_method=='ddpg' or self.learning_method=='td3':
                  print(f"{'Run':3s} {n:2.0f} {'| Episode':8s} {episode:5.0f} {'| Score':8s} {score:6.2f} {'| Progress':12s} {self.env.progress:3.2f} {'| collision ':14s} {self.env.collision} {'| Average score':15s} {avg_score:6.2f} {'| Average progress':18s} {avg_progress:3.2f}")

            episode+=1
               
         self.save_agent(n)

      outfile=open(self.train_results_file_name, 'wb')
      pickle.dump(scores, outfile)
      pickle.dump(progress, outfile)
      pickle.dump(times, outfile)
      pickle.dump(steps, outfile)
      outfile.close()

      #outfile=open(self.action_durations_name, 'wb')
      #pickle.dump(durations, outfile)
      #outfile.close()

   def test_while_train(self, n_episodes):
      
      print(f"{'Testing agent for '}{n_episodes}{' episodes'}")
      
      
      test_progress = []
      test_score = []
      
      for episode in range(n_episodes):
         
         self.env.reset(save_history=True, start_condition=[], car_params=self.env_dict['car_params'],get_lap_time=False)
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
      

   def save_agent(self, n):
      self.agent.save_agent(self.main_dict['name'], n)
      print("Agent " + self.main_dict['name'] + ", n = " + str(n) + " was saved")

   
def test(agent_name, n_episodes, detect_issues, initial_conditions):

   results_file_name = 'test_results/' + agent_name 
   replay_episode_name = 'replay_episodes/' + agent_name

   parent_dir = os.path.dirname(os.path.abspath(__file__))
   agent_dir = parent_dir + '/agents/' + agent_name
   agent_params_file = agent_dir + '/' + agent_name + '_params'


   infile = open('environments/' + agent_name, 'rb')
   env_dict = pickle.load(infile)
   infile.close()
   #env_dict['architecture'] = 'pete'

   if initial_conditions==True:
      start_condition_file_name = 'test_initial_condition/' + env_dict['map_name']
   else:
      start_condition_file_name = 'test_initial_condition/none' 
   
   infile = open(start_condition_file_name, 'rb')
   start_conditions = pickle.load(infile)
   infile.close()

   env_dict['max_steps'] = 2000

   #env = environment(env_dict, start_condition={'x':15,'y':5,'theta':0,'goal':0})
   env = environment(env_dict)
   car_params = env_dict['car_params']

   env.reset(save_history=False, start_condition=[], car_params=car_params, get_lap_time=False)
   
   action_history = []
   
   infile = open(agent_params_file, 'rb')
   agent_dict = pickle.load(infile)
   infile.close()
   agent_dict['layer3_size'] = 300
   
   infile = open('train_parameters/' + agent_name, 'rb')
   main_dict = pickle.load(infile)
   infile.close()

   runs = main_dict['runs']

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
   if main_dict['learning_method'] == 'td3':
      a = agent_td3.agent(agent_dict)
   

   test_progress = np.zeros([runs, n_episodes])
   test_score = np.zeros([runs, n_episodes])
   test_collision = np.zeros([runs, n_episodes])
   test_max_steps = np.zeros([runs, n_episodes])
   terminal_poses = np.zeros([runs, n_episodes])

   for n in range(runs):
      
      a = agent_td3.agent(agent_dict)
      #if main_dict['learning_method'] == 'rainbow':
      #   agent_dict['epsilon'] = 0
      #   a = agent_rainbow.agent(agent_dict)
      
      a.load_weights(agent_name, n)
      
      print("Testing agent " + agent_name + ", n = " + str(n))

      for episode in range(n_episodes):
         
         env.reset(save_history=True, start_condition=start_conditions[episode], car_params=car_params,get_lap_time=False)
         action_history = []
         obs = env.observation
         done = False
         score = 0

         while not done:
            if main_dict['learning_method']=='ddpg' or main_dict['learning_method']=='td3':
               action = a.choose_greedy_action(obs)
            else:
               action = a.choose_action(obs)
            
            
            action_history.append(action)
            
            next_obs, reward, done = env.take_action(action)
            score += reward
            obs = next_obs 
         
         test_progress[n, episode] = env.progress
         test_score[n, episode] = score
         test_collision[n, episode] = env.collision or env.park or env.backwards   
         test_max_steps[n, episode] = env.max_steps_reached
         #terminal_poses[n, episode] = env.pose
            
         if episode%10==0:
            print('Progress test episode', episode, '| Progress = %.2f' % env.progress, '| Score = %.2f' % score)

         if detect_issues==True and (env.progress<0.5):
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
   #pickle.dump(terminal_poses, outfile)
   outfile.close()


def lap_time_test(agent_name, n_episodes, detect_issues, initial_conditions):

   results_file_name = 'lap_results/' + agent_name
   parent_dir = os.path.dirname(os.path.abspath(__file__))
   agent_dir = parent_dir + '/agents/' + agent_name
   agent_params_file = agent_dir + '/' + agent_name + '_params'
   replay_episode_name = 'replay_episodes/' + agent_name

   infile = open('environments/' + agent_name, 'rb')
   env_dict = pickle.load(infile)
   infile.close()
   car_params = env_dict['car_params']
   

   if initial_conditions==True:
      start_condition_file_name = 'test_initial_condition/' + env_dict['map_name']
   else:
      start_condition_file_name = 'test_initial_condition/none' 
   
   infile = open(start_condition_file_name, 'rb')
   start_conditions = pickle.load(infile)
   infile.close()

   env_dict['max_steps'] = 2000
   

   #env = environment(env_dict, start_condition={'x':15,'y':5,'theta':0,'goal':0})
   
   #env_dict['architecture'] = 'pete'
   env = environment(env_dict)
   env.reset(save_history=False, start_condition=[], car_params=car_params, get_lap_time=True)
   
   action_history = []
   
   infile = open(agent_params_file, 'rb')
   agent_dict = pickle.load(infile)
   infile.close()
   agent_dict['layer3_size']=300

   infile = open('train_parameters/' + agent_name, 'rb')
   main_dict = pickle.load(infile)
   infile.close()

   runs = main_dict['runs']

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
   if main_dict['learning_method'] == 'td3':
      a = agent_td3.agent(agent_dict)
   
   times = np.zeros([runs, n_episodes])
   collisions = np.zeros([runs, n_episodes])

   for n in range(runs):

      a = agent_td3.agent(agent_dict)
      a.load_weights(agent_name, n)
      print("Testing agent " + agent_name + ", n = " + str(n))

      for episode in range(n_episodes):

         env.reset(save_history=True, start_condition=start_conditions[episode], get_lap_time=True, car_params=car_params)
         action_history = []
         obs = env.observation
         done = False
         score = 0

         while not done:
            if main_dict['learning_method']=='ddpg' or main_dict['learning_method']=='td3':
               action = a.choose_greedy_action(obs)
            else:
               action = a.choose_action(obs)

            action_history.append(action)
            
            next_obs, reward, done = env.take_action(action)
            score += reward
            obs = next_obs
            
         time = env.steps*0.01
         collision = env.collision or env.park or env.backwards
         
         if detect_issues==True and (collision==True):
            print('Stop condition met')
            print('Progress = ', env.progress)
            print('score = ', score)

            outfile = open(replay_episode_name, 'wb')
            pickle.dump(action_history, outfile)
            pickle.dump(env.initial_condition_dict, outfile)
            pickle.dump(n, outfile)
            pickle.dump(env_dict, outfile)
            outfile.close()
            break

         times[n, episode] = time
         collisions[n, episode] = collision 
            
         if episode%10==0:
            print('Lap test episode', episode, '| Lap time = %.2f' % time, '| Score = %.2f' % score)

   outfile=open(results_file_name, 'wb')
   pickle.dump(times, outfile)
   pickle.dump(collisions, outfile)
   outfile.close()

def lap_time_test_mismatch(agent_name, n_episodes, detect_issues, initial_conditions, parameter,  frac_variation):

   results_dir = 'lap_results_mismatch/' + agent_name
   results_file = results_dir + '/' + parameter
   parent_dir = os.path.dirname(os.path.abspath(__file__))
   agent_dir = parent_dir + '/agents/' + agent_name
   agent_params_file = agent_dir + '/' + agent_name + '_params'
   replay_episode_name = 'replay_episodes/' + agent_name

   try:
      os.mkdir(results_dir)
   except OSError as error:
      print(error)
      print("Warning: Files will be overwritten")

   
   infile = open('environments/' + agent_name, 'rb')
   env_dict = pickle.load(infile)
   infile.close()
   init_car_params = env_dict['car_params']
   

   if initial_conditions==True:
      start_condition_file_name = 'test_initial_condition/' + env_dict['map_name']
   else:
      start_condition_file_name = 'test_initial_condition/none' 
   
   infile = open(start_condition_file_name, 'rb')
   start_conditions = pickle.load(infile)
   infile.close()

   env_dict['max_steps'] = 2000
   

   #env = environment(env_dict, start_condition={'x':15,'y':5,'theta':0,'goal':0})
   
   #env_dict['architecture'] = 'pete'
   env = environment(env_dict)
   env.reset(save_history=False, start_condition=[], car_params=init_car_params, get_lap_time=True)
   
   action_history = []
   
   infile = open(agent_params_file, 'rb')
   agent_dict = pickle.load(infile)
   infile.close()
   agent_dict['layer3_size']=300

   infile = open('train_parameters/' + agent_name, 'rb')
   main_dict = pickle.load(infile)
   infile.close()

   runs = main_dict['runs']

   # if main_dict['learning_method']=='dqn':
   #    agent_dict['epsilon'] = 0
   #    a = agent_dqn.agent(agent_dict)
   # if main_dict['learning_method']=='reinforce':
   #    a = agent_reinforce.PolicyGradientAgent(agent_dict)
   # if main_dict['learning_method']=='actor_critic_sep':
   #    a = agent_actor_critic.actor_critic_separated(agent_dict)
   # if main_dict['learning_method']=='actor_critic_com':
   #    a = agent_actor_critic.actor_critic_combined(agent_dict)
   # if main_dict['learning_method']=='actor_critic_cont':
   #    a = agent_actor_critic_continuous.agent_separate(agent_dict) 
   # if main_dict['learning_method'] == 'dueling_dqn':
   #    agent_dict['epsilon'] = 0
   #    a = agent_dueling_dqn.agent(agent_dict)
   # if main_dict['learning_method'] == 'dueling_ddqn':
   #    agent_dict['epsilon'] = 0
   #    a = agent_dueling_ddqn.agent(agent_dict)
   # if main_dict['learning_method'] == 'rainbow':
   #    agent_dict['epsilon'] = 0
   #    a = agent_rainbow.agent(agent_dict)
   # if main_dict['learning_method'] == 'ddpg':
   #    a = agent_ddpg.agent(agent_dict)
   # if main_dict['learning_method'] == 'td3':
   #    a = agent_td3.agent(agent_dict)
   
   #parameter = 'C_Sf'
   #frac_variation = np.array([-0.05, 0.05])

   param_dict = {'parameter': parameter
               , 'original_value':init_car_params[parameter]
               , 'frac_variation': frac_variation
               , 'times_results': np.zeros((runs, len(frac_variation), n_episodes))
               , 'collision_results': np.zeros((runs, len(frac_variation), n_episodes))
               }

   for v_i, frac_vary in enumerate(frac_variation):
      
      car_params = init_car_params.copy()
      car_params[parameter] *= 1+frac_vary
   
      for n in range(runs):

         a = agent_td3.agent(agent_dict) 
         a.load_weights(agent_name, n)
         print("Testing agent " + agent_name + ", n = " + str(n) + ", parameter = " + parameter + ", variation = " + str(round(frac_vary,2)))

         for episode in range(n_episodes):

            env.reset(save_history=True, start_condition=start_conditions[episode], get_lap_time=True, car_params=car_params)
            action_history = []
            obs = env.observation
            done = False
            score = 0

            while not done:
               if main_dict['learning_method']=='ddpg' or main_dict['learning_method']=='td3':
                  action = a.choose_greedy_action(obs)
               else:
                  action = a.choose_action(obs)

               action_history.append(action)
               
               next_obs, reward, done = env.take_action(action)
               score += reward
               obs = next_obs
               
            time = env.steps*0.01
            collision = env.collision or env.park or env.backwards
            
            if detect_issues==True and (collision==True):
               print('Stop condition met')
               print('Progress = ', env.progress)
               print('score = ', score)

               outfile = open(replay_episode_name, 'wb')
               pickle.dump(action_history, outfile)
               pickle.dump(env.initial_condition_dict, outfile)
               pickle.dump(n, outfile)
               env_dict['car_params'] = car_params
               pickle.dump(env_dict, outfile)
               outfile.close()
               break

            param_dict['times_results'][n, v_i, episode] = time
            param_dict['collision_results'][n, v_i, episode] = collision

            if episode%10==0:
               print('Lap test episode', episode, '| Lap time = %.2f' % time, '| Score = %.2f' % score, '| Fail = %.2f' %collision)

   outfile=open(results_file, 'wb')
   pickle.dump(param_dict, outfile)
   outfile.close()

   

if __name__=='__main__':

   agent_name = 'pete_sv_berlin_3'

   main_dict = {'name':agent_name, 'max_episodes':50000, 'max_steps':5e6, 'learning_method':'td3', 'runs':5, 'comment':''}

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
   
   agent_td3_dict = {'alpha':0.001, 'beta':0.001, 'tau':0.005, 'gamma':0.99, 'update_actor_interval':2, 'warmup':100, 
                        'max_size':1000000, 'layer1_size':400, 'layer2_size':300, 'layer3_size':300, 'batch_size':100, 'noise':0.1}
   
   car_params =   {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145
                  , 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2
                  , 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}
   
   reward_signal = {'goal_reached':0, 'out_of_bounds':-1, 'max_steps':0, 'collision':-1, 
                     'backwards':-1, 'park':-1, 'time_step':-0.01, 'progress':0, 'distance':0.3, 
                     'max_progress':0}    
   
   action_space_dict = {'action_space': 'continuous', 'vel_select':[3,7], 'R_range':[2]}
   
   #action_space_dict = {'action_space': 'discrete', 'n_waypoints': 10, 'vel_select':[7], 'R_range':[6]}

   path_dict = {'local_path':True, 'waypoint_strategy':'local', 'wpt_arc':np.pi/2}
   
   if path_dict['local_path'] == True: #True or false
        path_dict['path_strategy'] = 'circle' #circle or linear or polynomial or gradient
        path_dict['control_strategy'] = 'pure_pursuit' #pure_pursuit or stanley
        
        if path_dict['control_strategy'] == 'pure_pursuit':
            path_dict['track_dict'] = {'k':0.1, 'Lfc':1}
        if path_dict['control_strategy'] == 'stanley':
            path_dict['track_dict'] = {'l_front': car_params['lf'], 'k':5, 'max_steer':car_params['s_max']}
   
   lidar_dict = {'is_lidar':True, 'lidar_res':0.1, 'n_beams':8, 'max_range':20, 'fov':np.pi}
   
   env_dict = {'sim_conf': functions.load_config(sys.path[0], "config")
            , 'save_history': False
            , 'map_name': 'berlin'
            , 'max_steps': 3000
            , 'control_steps': 20
            , 'display': False
            , 'architecture': 'pete'    #pete, ete
            , 'car_params':car_params
            , 'reward_signal':reward_signal
            , 'lidar_dict':lidar_dict
            , 'action_space_dict':action_space_dict
            , 'path_dict': path_dict
            } 
   
   a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
   a.train()
   test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)
   lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)
   lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.array([-0.05, 0.05]))
   
   agent_name = 'pete_sv_circle_3'
   main_dict['name'] = agent_name
   env_dict['map_name'] = 'circle'
   env_dict['architecture'] = 'pete'
   a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
   a.train()
   lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

   agent_name = 'pete_sv_torino_3'
   main_dict['name'] = agent_name
   env_dict['map_name'] = 'torino'
   env_dict['architecture'] = 'pete'
   a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
   a.train()
   lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

   agent_name = 'pete_sv_columbia_3'
   main_dict['name'] = agent_name
   env_dict['map_name'] = 'columbia_1'
   env_dict['architecture'] = 'pete'
   a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
   a.train()
   lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

   agent_name = 'pete_sv_porto_3'
   main_dict['name'] = agent_name
   env_dict['map_name'] = 'porto_1'
   env_dict['architecture'] = 'pete'
   a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
   a.train()
   lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)
   
   agent_name = 'pete_sv_redbull_ring_3'
   main_dict['name'] = agent_name
   env_dict['map_name'] = 'redbull_ring'
   env_dict['architecture'] = 'pete'
   env_dict['reward_signal']['time_step']=-0.005
   a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
   a.train()
   lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

   
   

   agent_name = 'pete_v_berlin_3'
   main_dict['name'] = agent_name
   env_dict['map_name'] = 'berlin'
   env_dict['architecture'] = 'pete'
   env_dict['path_dict']['local_path'] = False
   env_dict['reward_signal']['time_step']=-0.01
   a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
   a.train()
   lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

   agent_name = 'pete_v_circle_3'
   main_dict['name'] = agent_name
   env_dict['map_name'] = 'circle'
   env_dict['architecture'] = 'pete'
   env_dict['path_dict']['local_path'] = False
   a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
   a.train()
   lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

   agent_name = 'pete_v_torino_3'
   main_dict['name'] = agent_name
   env_dict['map_name'] = 'torino'
   env_dict['architecture'] = 'pete'
   env_dict['path_dict']['local_path'] = False
   a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
   a.train()
   lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

   agent_name = 'pete_v_columbia_3'
   main_dict['name'] = agent_name
   env_dict['map_name'] = 'columbia_1'
   env_dict['architecture'] = 'pete'
   env_dict['path_dict']['local_path'] = False
   a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
   a.train()
   lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

   agent_name = 'pete_v_porto_3'
   main_dict['name'] = agent_name
   env_dict['map_name'] = 'porto_1'
   env_dict['architecture'] = 'pete'
   env_dict['path_dict']['local_path'] = False
   a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
   a.train()
   lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

   agent_name = 'pete_v_redbull_ring_3'
   main_dict['name'] = agent_name
   env_dict['map_name'] = 'redbull_ring'
   env_dict['architecture'] = 'pete'
   env_dict['path_dict']['local_path'] = False
   env_dict['reward_signal']['time_step']=-0.005
   a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
   a.train()
   lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)





   agent_name = 'ete_berlin_3'
   main_dict['name'] = agent_name
   env_dict['map_name'] = 'berlin'
   env_dict['architecture'] = 'ete'
   env_dict['path_dict']['local_path'] = False
   env_dict['reward_signal']['time_step']=-0.01
   a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
   a.train()
   lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

   agent_name = 'ete_circle_3'
   main_dict['name'] = agent_name
   env_dict['map_name'] = 'circle'
   env_dict['architecture'] = 'ete'
   a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
   a.train()
   lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

   agent_name = 'ete_torino_3'
   main_dict['name'] = agent_name
   env_dict['map_name'] = 'torino'
   env_dict['architecture'] = 'ete'
   a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
   a.train()
   lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

   agent_name = 'ete_columbia_3'
   main_dict['name'] = agent_name
   env_dict['map_name'] = 'columbia_1'
   env_dict['architecture'] = 'ete'
   a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
   a.train()
   lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

   agent_name = 'ete_porto_3'
   main_dict['name'] = agent_name
   env_dict['map_name'] = 'porto_1'
   env_dict['architecture'] = 'ete'
   a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
   a.train()
   lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)
   
   agent_name = 'ete_redbull_ring_3'
   main_dict['name'] = agent_name
   env_dict['map_name'] = 'redbull_ring'
   env_dict['architecture'] = 'ete'
   env_dict['reward_signal']['time_step']=-0.005
   a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
   a.train()
   lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)



   # agent_name = 'ete_new__porto'
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='lf', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='lr', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='h', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='m', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='I', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='s_min', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='s_max', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='sv_min', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='sv_max', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='a_max', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='v_max', frac_variation=np.linspace(-0.2,0.2,21) )
  
   # agent_name = 'ete_porto'
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='lf', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='lr', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='h', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='m', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='I', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='s_min', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='s_max', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='sv_min', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='sv_max', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='a_max', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='v_max', frac_variation=np.linspace(-0.2,0.2,21) )
  
   # agent_name = 'pete_porto'
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='lf', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='lr', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='h', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='m', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='I', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='s_min', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='s_max', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='sv_min', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='sv_max', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='a_max', frac_variation=np.linspace(-0.2,0.2,21) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='v_max', frac_variation=np.linspace(-0.2,0.2,21) )
  
   # agent_name = 'pete_sv_berlin_1'
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,5) )
   # agent_name = 'pete_v_berlin_1'
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,5) )
   # agent_name = 'ete_berlin_1'
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,5) )
   # agent_name = 'pete_sv_berlin_1'
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,5) )
   # agent_name = 'pete_v_berlin_1'
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,5) )
   # agent_name = 'ete_berlin_1'
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,5) )
   
   # agent_name = 'pete_sv_circle_1'
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,5) )
   # agent_name = 'pete_v_circle_1'
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,5) )
   # agent_name = 'ete_circle_1'
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,5) )
   # agent_name = 'pete_sv_circle_1'
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,5) )
   # agent_name = 'pete_v_circle_1'
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,5) )
   # agent_name = 'ete_circle_1'
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,5) )
   
   # agent_name = 'pete_sv_torino_2'
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,5) )
   # agent_name = 'pete_v_torino_2'
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,5) )
   # agent_name = 'ete_torino_1'
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,5) )
   # agent_name = 'pete_sv_torino_2'
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,5) )
   # agent_name = 'pete_v_torino_2'
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,5) )
   # agent_name = 'ete_torino_1'
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,5) )
   
   # agent_name = 'pete_sv_redbull_ring'
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,5) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,5) )
   # agent_name = 'pete_v_redbull_ring_2'
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,5) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,5) )
   # agent_name = 'ete_redbull_ring_1'
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,5) )
   # lap_time_test_mismatch(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,5) )
   

   # agent_name = 'ete_porto'
   # environment_name = 'environments/' + agent_name
   
   # infile = open(environment_name, 'rb')
   # env_dict = pickle.load(infile)
   # infile.close()
   # env_dict['architecture'] = 'pete'

   # outfile=open(environment_name, 'wb')
   # pickle.dump(env_dict, outfile)
   # outfile.close()

   #agent_names = ['ete_new__porto', 'ete_porto', 'pete_porto']
   #agent_names = ['ete_berlin_1', 'pete_v_berlin_1', 'pete_sv_berlin_1']
   #agent_names = ['ete_circle_1', 'pete_v_circle_1', 'pete_sv_circle_1']
   #agent_names = ['ete_torino_1', 'pete_v_torino_2', 'pete_sv_torino_2']
   #agent_names = ['ete_redbull_ring_1', 'pete_v_redbull_ring_2', 'pete_sv_redbull_ring']

   #legend_title = 'architecture description'
   #legend = ['No controllers', 'Velocity controller', 'Steering and velocity controllers']
   #parameters = ['lf', 'a_max', 'C_Sf', 'C_Sr']
   #parameters = ['C_Sf', 'C_Sr']
   #plot_titles = ['Distance between CoG and front axle', 'Maximum acceleration', 'Front tyres cornering stiffness', 'Rear tyres cornering stiffness']
   #plot_titles = ['Front tyres cornering stiffness', 'Rear tyres cornering stiffness']
   #display_results_multiple.display_lap_mismatch_results(agent_names, parameters, legend_title, legend, plot_titles)
   #display_results_multiple.display_lap_mismatch_results_box(agent_names, parameters, legend_title, legend)
   
   # agent_name = 'pete_sv_circle_1'
   # agent_name = 'pete_sv_berlin_1' #failed - must redo
   # agent_name = 'pete_sv_torino'
   # agent_name = 'pete_sv_redbull_ring'
   
   # agent_name = 'pete_v_circle_1'
   # agent_name = 'pete_v_berlin_1'
   # agent_name = 'pete_v_torino_1'
   # agent_name = 'pete_v_redbull_ring_2'

   # agent_name = 'ete_circle_1'
   # agent_name = 'ete_berlin_1'
   # agent_name = 'ete_torino_1'
   # agent_name = 'ete_redbull_ring_1'

   #legend = [agent_name]
   #legend_title = 'agent name'
   #display_results_multiple.compare_learning_curves_progress(agent_names=[agent_name], legend=legend, legend_title=legend_title, 
   #   show_average=True, show_median=False, xaxis='steps')
   #display_results_multiple.display_train_parameters(agent_name=agent_name)
   #display_results_multiple.agent_progress_statistics(agent_name=agent_name)
   #display_results_multiple.display_lap_results(agent_name=agent_name)
   #display_results_multiple.display_moving_agent(agent_name=agent_name, load_history=False, n=0)
   #display_results_multiple.display_path(agent_name=agent_name, load_history=False, n=0)

   #agent_name = 'pete_sv_redbull_ring_1'
   #test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)
   #lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)
   #display_results_multiple.display_train_parameters(agent_name=agent_name)
   #display_results_multiple.display_lap_results(agent_name=agent_name)
   #display_results_multiple.display_moving_agent(agent_name=agent_name, load_history=False, n=4)


   # agent_names = ['ete_redbull_ring', 'pete_v_redbull_ring', 'pete_sv_redbull_ring']
   # agent_names = ['ete_torino', 'pete_v_torino', 'pete_sv_torino']
   # legend_title = 'Architecture'
   # legend = ['fully end-to-end', 'partial end-to-end, velocity controller', 'partial end-to-end, steering and velocity controllers']
   # ns = [0, 0, 0]
   # display_results_multiple.compare_learning_curves_progress(agent_names, legend, legend_title, show_average=True, show_median=True, xaxis='episodes')
   # display_results_multiple.display_path_multiple(agent_names=agent_names, ns=ns, legend_title=legend_title, legend=legend)
   
   
   #agent_names = ['ete_new__porto', 'ete_porto', 'pete_porto']
   #agent_names = ['ete_berlin_1', 'pete_v_berlin_1', 'pete_sv_berlin_1']
   #agent_names = ['ete_circle_1', 'pete_v_circle_1', 'pete_sv_circle_1']
   #agent_names = ['ete_torino_1', 'pete_v_torino_2', 'pete_sv_torino_2']
   #agent_names = ['ete_redbull_ring_1', 'pete_v_redbull_ring_2', 'pete_sv_redbull_ring']
   #legend_title = 'Architecture'
   #legend = ['no controllers', 'only velocity control', 'steering and velocity control']
   #ns = [0, 0, 0]
   #mismatch_parameters = ['C_Sr']
   #frac_vary = [-0.2]
   #display_results_multiple.compare_learning_curves_progress(agent_names, legend, legend_title, show_average=True, show_median=True, xaxis='episodes')
   #display_results_multiple.display_path_multiple(agent_names=agent_names, ns=ns, legend_title=legend_title, 
   #           legend=legend, mismatch_parameters=mismatch_parameters, frac_vary=frac_vary)



   # agent_name = 'pete_sv_redbull_ring_2'
   # infile = open('train_parameters/' + agent_name, 'rb')
   # main_dict = pickle.load(infile)
   # infile.close()
   # main_dict['runs']=1

   # outfile=open('train_parameters/' + agent_name, 'wb')
   # pickle.dump(main_dict, outfile)
   # outfile.close()


