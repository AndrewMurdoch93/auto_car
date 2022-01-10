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

fileName = 'agents/bicycle_dynamics_agent'

class trainingLoop():
    def __init__(self):
        
        #Parameters for training the agent
        self.gamma=0.99
        self.epsilon = 1
        self.eps_end = 0.01
        self.eps_dec = 5e-4
        self.batch_size = 64
        self.n_actions = 8
        self.input_dims = 3
        self.lr = 0.001
        self.max_episodes = 4000
        self.max_mem_size = 100000

        #Parameters for the environment
        self.env = environment(functions.load_config(sys.path[0], "config"))
        

    def train(self, save):
        
        self.agent = agent.agent(gamma=self.gamma, epsilon=self.epsilon, lr=self.lr, input_dims=self.input_dims, 
        batch_size=self.batch_size, n_actions=self.n_actions, max_mem_size=self.max_mem_size, eps_end=self.eps_end, 
        eps_dec=self.eps_dec)
        
        scores = []

        for episode in range(self.max_episodes):
            self.env.reset()
            state = [self.env.state[0], self.env.state[1], self.env.state[2]]
            done = False
            score=0

            while not done:
                #state = state.flatten()
                action = self.agent.choose_action(state)
                #action=2
                #print(action)
                #action = int(input())
                #action=8

                next_state, reward, done = self.env.take_action(action)
                next_state = [self.env.state[0], self.env.state[1], self.env.state[2]]
                #next_state = next_state.flatten()
                score += reward

                #print(f"next_state = {next_state}")
                #print(f"reward = {reward}")
                #print(f"done = {done}")
                self.agent.store_transition(state, action, reward, next_state, done)
                self.agent.learn()
                state = next_state
            
            scores.append(score)
            avg_score = np.mean(scores[-100:])

            print('episode ', episode, 'score %.2f' % score,
            'average score %.2f' % avg_score,
            'epsilon %.2f' % self.agent.epsilon)
        
        if save==True:
            outfile=open(fileName, 'wb')
            pickle.dump(self.agent, outfile)
            outfile.close()

    
    def test(self):
        
        if not hasattr(self, 'agent'):
            infile = open(fileName, 'rb')
            self.agent = pickle.load(infile)

        self.env.reset()
        state = self.env.state
        done = False
        score=0

        state_history = []
        action_history = []
        state_history.append(self.env.state[:])
        #actions = [3, 2]
        i=0
        while not done:
            #state = state.flatten()
            action = self.agent.choose_action([state[0], state[1], state[2]])
            #action = 7
            #print(action)
            next_state, reward, done = self.env.take_action(action)
            
            #next_state = next_state.flatten()
            score += reward

            if done==True:
                print('Done')

            #print(f"next_state = {next_state}")
            #print(f"reward = {reward}")
            #print(f"done = {done}")
            #self.agent.store_transition(state, action, reward, next_state, done)
            #self.agent.learn()
            state = next_state
            i+=1

        #pass 
        #print(action_memory)
        #print(state_history)
        i=0
        for sh, ah, lph in zip(self.env.state_history, self.env.action_history, self.env.local_path_history):
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            #plt.plot([x[0] for x in state_history], [x[1] for x in state_history])
            #plt.plot([x[0] for x in action_history], [x[1] for x in action_history], 'x')
            plt.arrow(sh[0], sh[1], 0.1*math.cos(sh[2]), 0.1*math.sin(sh[2]), head_length=0.04,head_width=0.02, ec='None', fc='blue')
            plt.arrow(sh[0], sh[1], 0.1*math.cos(sh[2]+sh[3]), 0.1*math.sin(sh[2]+sh[3]), head_length=0.04,head_width=0.02, ec='None', fc='red')
            plt.plot(sh[0], sh[1], 'o')
            plt.plot(ah[0], ah[1], 'x')
            plt.plot(lph[0], lph[1], 'g')
            plt.plot([1.5, 2.5, 2.5, 1.5, 1.5], [1.5, 1.5, 2.5, 2.5, 1.5], 'r')
            plt.legend(["vehicle trajectory", "predicted waypoints", "goal area"])
            plt.xlabel('x coordinate')
            plt.ylabel('y coordinate')
            plt.xlim([-0.5,3])
            plt.ylim([-0.5,3])
            plt.grid(True)
            plt.title('Vehicle trajectory')
            plt.pause(0.01)
            #print(self.env.state_history[i][3])
            i+=1

        print(self.env.steps)
                

if __name__=='__main__':
    a = trainingLoop()
    
    #a.train(save=True)
    a.test()