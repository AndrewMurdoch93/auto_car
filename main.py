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

class trainingLoop():
    def __init__(self, agent_name):
        
        #Initialise file names for saving data
        self.agent_file_name = 'agents/' + agent_name
        self.history_file_name = 'test_history/' + agent_name

        #Initialising environment
        self.env = environment(functions.load_config(sys.path[0], "config"))

        #Hyper parameters for training the agent
        self.gamma=0.99
        self.epsilon = 1
        self.eps_end = 0.1
        self.eps_dec = 1e-3
        self.batch_size = 64
        self.lr = 0.001
        self.max_episodes = 10000
        self.max_mem_size = 1000000

        #Agent constraints
        self.n_actions = self.env.num_actions
        self.input_dims = len(self.env.observation)

    def train(self, save_agent):
        
        #Initialise the agent
        self.agent = agent.agent(gamma=self.gamma, epsilon=self.epsilon, lr=self.lr, input_dims=self.input_dims, 
        batch_size=self.batch_size, n_actions=self.n_actions, max_mem_size=self.max_mem_size, eps_end=self.eps_end, 
        eps_dec=self.eps_dec)
        
        scores = []

        for episode in range(self.max_episodes):
            
            #Reset the environment every episode
            self.env.reset()
            obs = self.env.observation  #Records starting state
            #state = [self.env.state[0], self.env.state[1], self.env.state[2]]
            done = False
            score=0 #Initialise score counter for every episode

            #For every planning time step in the episode
            while not done:
                
                action = self.agent.choose_action(obs)    #Select an action
                next_obs, reward, done = self.env.take_action(action) #Environment executes action
                score += reward 
                self.agent.store_transition(obs, action, reward, next_obs, done)
                self.agent.learn()  #Learn using state transition history
                obs = next_obs  #Update the state
            
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            self.agent.decrease_epsilon()

            if save_agent==True and episode%1000==0:
                outfile=open(self.agent_file_name, 'wb')
                pickle.dump(self.agent, outfile)
                outfile.close()  

            print('episode ', episode, 'score %.2f' % score, 'average score %.2f' % avg_score, 'epsilon %.2f' % self.agent.epsilon)
        
        #Save the agent
        if save_agent==True:
            outfile=open(self.agent_file_name, 'wb')
            pickle.dump(self.agent, outfile)
            outfile.close()


    def test(self, save_history, display, verbose):
        
        #Load agent from memory if agent doesn't exist 
        if not hasattr(self, 'agent'):
            infile = open(self.agent_file_name, 'rb')
            self.agent = pickle.load(infile)
            infile.close()

        self.env.reset(save_history=True)
        
        obs = self.env.observation
        done = False
        score=0

        while not done:
            action = self.agent.choose_action(obs)
            next_obs, reward, done = self.env.take_action(action)
            score += reward
            obs = next_obs
        
        #Record histories
        if save_history==True:
            outfile=open(self.history_file_name, 'wb')
            pickle.dump(self.env.state_history, outfile)
            pickle.dump(self.env.action_history, outfile)
            pickle.dump(self.env.goal_history, outfile)
            pickle.dump(self.env.observation_history, outfile)
            if self.env.local_path==True:
                pickle.dump(self.env.local_path_history, outfile)
            outfile.close()

        if verbose==True:
            print(self.env.steps)
        
        if display==True:
            self.state_history = self.env.state_history
            self.action_history = self.env.action_history
            self.goal_history = self.env.goal_history
            self.observation_history = self.env.observation_history
            if self.env.local_path==True:
                self.local_path_history = self.env.local_path_history
            
            self.display_history(load_history=False)
        
    #Display a history of the episode
    def display_history(self, load_history=True):

        if load_history==True:
            infile = open(self.history_file_name, 'rb')
            self.state_history = pickle.load(infile)
            self.action_history = pickle.load(infile)
            self.goal_history = pickle.load(infile)
            self.observation_history = pickle.load(infile)
            if self.env.local_path==True:
                self.local_path_history = pickle.load(infile)
            infile.close()

        image_path = sys.path[0] + '/maps/' + 'berlin' + '.png'
        im = image.imread(image_path)
        plt.imshow(im, extent=(0,30,0,30))

        if self.env.local_path==True:
            for sh, ah, gh, lph in zip(self.state_history, self.action_history, self.goal_history, self.local_path_history):
                plt.cla()
                # Stop the simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
                plt.imshow(im, extent=(0,30,0,30))
                plt.arrow(sh[0], sh[1], 0.5*math.cos(sh[2]), 0.1*math.sin(sh[2]), head_length=0.5, head_width=0.5, shape='full', ec='None', fc='blue')
                plt.arrow(sh[0], sh[1], 0.5*math.cos(sh[2]+sh[3]), 0.1*math.sin(sh[2]+sh[3]), head_length=0.5, head_width=0.5, shape='full',ec='None', fc='red')
                plt.plot(sh[0], sh[1], 'o')
                plt.plot(ah[0], ah[1], 'x')
                plt.plot(lph[0], lph[1], 'g')
                plt.plot([gh[0]-self.env.s, gh[0]+self.env.s, gh[0]+self.env.s, gh[0]-self.env.s, gh[0]-self.env.s], [gh[1]-self.env.s, gh[1]-self.env.s, gh[1]+self.env.s, gh[1]+self.env.s, gh[1]-self.env.s], 'r')
                plt.legend(["position", "waypoint", "local path", "goal area", "heading", "steering angle"])
                plt.xlabel('x coordinate')
                plt.ylabel('y coordinate')
                plt.xlim([0,30])
                plt.ylim([0,30])
                plt.grid(True)
                plt.title('Episode history')
                plt.pause(0.01)
        else:
            for sh, ah, gh in zip(self.state_history, self.action_history, self.goal_history):
                plt.cla()
                # Stop the simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
                plt.imshow(im, extent=(0,30,0,30))
                plt.arrow(sh[0], sh[1], 0.5*math.cos(sh[2]), 0.1*math.sin(sh[2]), head_length=0.5, head_width=0.5, shape='full', ec='None', fc='blue')
                plt.arrow(sh[0], sh[1], 0.5*math.cos(sh[2]+sh[3]), 0.1*math.sin(sh[2]+sh[3]), head_length=0.5, head_width=0.5, shape='full',ec='None', fc='red')
                plt.plot(sh[0], sh[1], 'o')
                plt.plot(ah[0], ah[1], 'x')
                plt.plot([gh[0]-self.env.s, gh[0]+self.env.s, gh[0]+self.env.s, gh[0]-self.env.s, gh[0]-self.env.s], [gh[1]-self.env.s, gh[1]-self.env.s, gh[1]+self.env.s, gh[1]+self.env.s, gh[1]-self.env.s], 'r')
                #plt.legend(["position", "waypoint", "goal area", "heading", "steering angle"])
                plt.xlabel('x coordinate')
                plt.ylabel('y coordinate')
                plt.xlim([0,30])
                plt.ylim([0,30])
                plt.grid(True)
                plt.title('Episode history')
                plt.pause(0.01)

                    

if __name__=='__main__':
    
    a = trainingLoop(agent_name='track_agent')
    #a.train(save_agent=True)
    a.test(save_history=True, display=True, verbose=True)
    #a.display_history()
