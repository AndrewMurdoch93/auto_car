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
    def __init__(self, agent_name, gamma=0.99, epsilon=1, eps_end=0.01, eps_dec=1e-3, batch_size=64, lr=0.001, max_episodes=10000, max_mem_size=1000000,
                    map_name='circle', max_steps=1500,local_path=False, waypoint_strategy='local', reward_signal=[1,-1,-1,-1,-0.001], control_steps=10):
        
        #Initialise file names for saving data
        self.agent_file_name = 'agents/' + agent_name
        self.train_results_file_name = 'train_results/' + agent_name
        self.environment_name = 'environments/' + agent_name

        #Hyper parameters for training the agent
        self.gamma=gamma
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.batch_size = batch_size
        self.lr = lr
        self.max_episodes = max_episodes
        self.max_mem_size = max_mem_size

        #Agent constraints
        self.n_actions = 8

        #Initialising environment
        self.env = environment(functions.load_config(sys.path[0], "config"), save_history=True, map_name=map_name, max_steps=max_steps, 
        local_path=local_path, waypoint_strategy=waypoint_strategy, reward_signal=reward_signal, num_actions=self.n_actions, control_steps=control_steps)

        self.input_dims = len(self.env.observation)
        
        #Display constraints
        self.window=100

        
    def train(self):
        
        #Save the environment
        outfile=open(self.environment_name, 'wb')
        pickle.dump(self.env, outfile)
        outfile.close()
        
        #Initialise the agent
        self.agent = agent.agent(gamma=self.gamma, epsilon=self.epsilon, lr=self.lr, input_dims=self.input_dims, 
        batch_size=self.batch_size, n_actions=self.n_actions, max_mem_size=self.max_mem_size, eps_end=self.eps_end, 
        eps_dec=self.eps_dec)

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
                outfile=open(self.agent_file_name, 'wb')
                pickle.dump(self.agent, outfile)
                outfile.close()

                outfile=open(self.train_results_file_name, 'wb')
                pickle.dump(scores, outfile)
                pickle.dump(progress, outfile)
                outfile.close()


            if episode%10==0:
                print(f"{'Episode':8s} {episode:5.0f} {'| Score':8s} {score:6.2f} {'| Progress':12s} {self.env.progress:3.2f} {'| Average score':15s} {avg_score:6.2f} {'| Average progress':18s} {avg_progress:3.2f} {'| Epsilon':9s} {self.agent.epsilon:.2f}")
      
        
        outfile=open(self.agent_file_name, 'wb')
        pickle.dump(self.agent, outfile)
        outfile.close()

        outfile=open(self.train_results_file_name, 'wb')
        pickle.dump(scores, outfile)
        pickle.dump(progress, outfile)
        outfile.close()


def test(agent_name, n_episodes=1000):
    
    agent_file_name = 'agents/' + agent_name
    environment_name = 'environments/' + agent_name
    results_file_name = 'test_results/' + agent_name


    infile = open(agent_file_name, 'rb')
    agent = pickle.load(infile)
    infile.close()
   
    infile = open(environment_name, 'rb')
    env = pickle.load(infile)
    infile.close()

    test_progress = []
    test_score = []

    for episode in range(n_episodes):

        env.reset(save_history=True)
        obs = env.observation
        done = False
        score=0

        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done = env.take_action(action)
            score += reward
            obs = next_obs
        
        test_progress.append(env.progress)
        test_score.append(score)
        
        if episode%50==0:
            print('Test episode', episode, '| Progress = %.2f' % env.progress, '| Score = %.2f' % score)
        
    outfile=open(results_file_name, 'wb')
    pickle.dump(test_score, outfile)
    pickle.dump(test_progress, outfile)
    outfile.close()


             

if __name__=='__main__':
    
    agent_name='standard'
    #a = trainingLoop (agent_name=agent_name, gamma=0.99, epsilon=1, eps_end=0.01, eps_dec=1e-3, batch_size=64, lr=0.001, max_episodes=50000, max_mem_size=2000000,
    #                map_name='circle', max_steps=1500,local_path=False, waypoint_strategy='local', reward_signal=[1,-1,-1,-1,-0.001], control_steps=10)
    #a.train()
    #test(agent_name=agent_name)

    #agent_name='local_path'
    #b = trainingLoop (agent_name=agent_name, gamma=0.99, epsilon=1, eps_end=0.01, eps_dec=1e-3, batch_size=64, lr=0.001, max_episodes=50000, max_mem_size=2000000,
    #                map_name='circle', max_steps=1500,local_path=True, waypoint_strategy='local', reward_signal=[1,-1,-1,-1,-0.001], control_steps=10)
    #b.train()
    #test(agent_name=agent_name)

    #agent_name='standard'
    display_results.learning_curve_score(agent_name=agent_name, show_average=True, show_median=True)
    display_results.learning_curve_progress(agent_name=agent_name, show_average=True, show_median=True)
    display_results.agent_score_statistics(agent_name=agent_name)
    display_results.agent_progress_statistics(agent_name=agent_name)
    display_results.histogram_score(agent_name=agent_name)
    display_results.histogram_progress(agent_name=agent_name)
    display_results.display_moving_agent(agent_name=agent_name, load_history=False, save_history=False)
