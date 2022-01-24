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

class trainingLoop():
    def __init__(self, agent_name):
        
        #Initialise file names for saving data
        self.agent_file_name = 'agents/' + agent_name
        self.train_results_file_name = 'train_results/' + agent_name
        self.history_file_name = 'test_history/' + agent_name
        self.results_file_name = 'test_results/' + agent_name

        #Initialising environment
        self.env = environment(functions.load_config(sys.path[0], "config"))

        #Hyper parameters for training the agent
        self.gamma=0.99
        self.epsilon = 1
        self.eps_end = 0.01
        self.eps_dec = 1e-3
        self.batch_size = 64
        self.lr = 0.001
        self.max_episodes = 10000
        self.max_mem_size = 1000000

        #Agent constraints
        self.n_actions = self.env.num_actions
        self.input_dims = len(self.env.observation)

        #Display constraints
        self.window=100

        
    def train(self, save_agent=True, save_score=True):
        
        #Initialise the agent
        self.agent = agent.agent(gamma=self.gamma, epsilon=self.epsilon, lr=self.lr, input_dims=self.input_dims, 
        batch_size=self.batch_size, n_actions=self.n_actions, max_mem_size=self.max_mem_size, eps_end=self.eps_end, 
        eps_dec=self.eps_dec)

        scores = []
        progress = []
        times = []
 
        
        for episode in range(self.max_episodes):
            
            self.env.reset()                #Reset the environment every episode
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

            if save_agent==True and episode%1000==0:
                outfile=open(self.agent_file_name, 'wb')
                pickle.dump(self.agent, outfile)
                outfile.close()  

            if episode%10==0:
                print(f"{'Episode':8s} {episode:5.0f} {'| Score':8s} {score:6.2f} {'| Average score':15s} {avg_score:6.2f} {'| Progress':12s} {self.env.progress:3.2f} {'| Average progress':18s} {avg_progress:3.2f} {'| Epsilon':9s} {self.agent.epsilon:.2f}")
      
            

        if save_agent==True:
            outfile=open(self.agent_file_name, 'wb')
            pickle.dump(self.agent, outfile)
            outfile.close()

        if save_score==True:
            outfile=open(self.train_results_file_name, 'wb')
            pickle.dump(scores, outfile)
            pickle.dump(progress, outfile)
            outfile.close()


    def learning_curve_score(self, show_average=False, show_median=True):
        
        infile = open(self.train_results_file_name, 'rb')
        scores = pickle.load(infile)
        progress = pickle.load(infile)
        infile.close()
    
        avg_scores = []
        std_dev = []
        percentile_25 = []
        median = []
        percentile_75 = []

        
        for i in range(self.max_episodes):
            if i <= self.window:
                x = 0
            else:
                x = i-self.window 
            
            avg_scores.append(np.mean(scores[x:i+1]))
            std_dev.append(np.std(scores[x:i+1]))
            percentile_25.append(np.percentile(scores[x:i+1], 25))
            median.append(np.percentile(scores[x:i+1], 50))
            percentile_75.append( np.percentile(scores[x:i+1], 75))
      
        if show_median==True:
            plt.plot(scores)
            plt.plot(median, color='black')
            plt.fill_between(np.arange(len(scores)), percentile_25, percentile_75, color='lightblue')
            plt.title('Learning curve')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.legend(['Episode score', 'Median score', '25th to 75th percentile'])
            plt.show()

        if show_average==True:
            plt.plot(scores)
            plt.plot(avg_scores, color='black')
            plt.fill_between(np.arange(len(scores)), np.add(avg_scores,std_dev), np.subtract(avg_scores,std_dev), color='lightblue')
            plt.title('Learning curve')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.legend(['Episode score', 'Average score', 'Standard deviation from mean'])
            plt.show()
        

    def learning_curve_progress(self, show_average=False, show_median=True):
        
        infile = open(self.train_results_file_name, 'rb')
        scores = pickle.load(infile)
        progress = pickle.load(infile)
        infile.close()
    
        avg_scores = []
        std_dev = []
        percentile_25 = []
        median = []
        percentile_75 = []

        
        for i in range(self.max_episodes):
            if i <= self.window:
                x = 0
            else:
                x = i-self.window 
            
            avg_scores.append(np.mean(progress[x:i+1]))
            std_dev.append(np.std(progress[x:i+1]))
            percentile_25.append(np.percentile(progress[x:i+1], 25))
            median.append(np.percentile(progress[x:i+1], 50))
            percentile_75.append( np.percentile(progress[x:i+1], 75))
      
        if show_median==True:
            plt.plot(progress)
            plt.plot(median, color='black')
            plt.fill_between(np.arange(len(progress)), percentile_25, percentile_75, color='lightblue')
            plt.title('Learning curve')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.legend(['Episode score', 'Median score', '25th to 75th percentile'])
            plt.show()

        if show_average==True:
            plt.plot(progress)
            plt.plot(avg_scores, color='black')
            plt.fill_between(np.arange(len(progress)), np.add(avg_scores,std_dev), np.subtract(avg_scores,std_dev), color='lightblue')
            plt.title('Learning curve')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.legend(['Episode score', 'Average score', 'Standard deviation from mean'])
            plt.show()


    def test(self, n_episodes=1000):
        

        infile = open(self.agent_file_name, 'rb')
        agent = pickle.load(infile)
        infile.close()

        
        test_progress = []
        test_score = []

        for episode in range(n_episodes):

            self.env.reset(save_history=True)
            obs = self.env.observation
            done = False
            score=0

            while not done:
                action = agent.choose_action(obs)
                next_obs, reward, done = self.env.take_action(action)
                score += reward
                obs = next_obs
            
            test_progress.append(self.env.progress)
            test_score.append(score)
            if episode%50==0:
                print('Test episode', episode, '| Progress = %.2f' % self.env.progress, '| Score = %.2f' % score)
            
        outfile=open(self.results_file_name, 'wb')
        pickle.dump(test_score, outfile)
        pickle.dump(test_progress, outfile)
        outfile.close()
        

    def histogram_score(self):

        infile = open(self.results_file_name, 'rb')
        test_score = pickle.load(infile)
        test_progress = pickle.load(infile)
        infile.close()

        #sns.displot(test_progress)
        percentile25 = np.percentile(test_score, 25)
        percentile50 = np.percentile(test_score, 50)
        percentile75 = np.percentile(test_score, 75)

        y, x, _ = plt.hist(test_score, bins=12)
        plt.plot([percentile25, percentile25], [0, y.max()])
        plt.plot([percentile50, percentile50], [0, y.max()])
        plt.plot([percentile75, percentile75], [0, y.max()])
        
        plt.title('Score distribution')
        plt.xlabel('Score')
        plt.ylabel('Number of agents')
        plt.legend(['25th percentile', 'Median', '75th percentile', 'Agent scores'])
        plt.show()
    
    
    def histogram_progress(self):

        infile = open(self.results_file_name, 'rb')
        test_score = pickle.load(infile)
        test_progress = pickle.load(infile)
        infile.close()

        #sns.displot(test_progress)
        percentile25 = np.percentile(test_progress, 25)
        percentile50 = np.percentile(test_progress, 50)
        percentile75 = np.percentile(test_progress, 75)

        y, x, _ = plt.hist(test_progress, bins=12)
        plt.plot([percentile25, percentile25], [0, y.max()])
        plt.plot([percentile50, percentile50], [0, y.max()])
        plt.plot([percentile75, percentile75], [0, y.max()])
        
        plt.title('Progress distribution')
        plt.xlabel('Score')
        plt.ylabel('Number of agents')
        plt.legend(['25th percentile', 'Median', '75th percentile', 'Agent progress'])
        plt.show()
    

    def agent_score_statistics(self):
        infile = open(self.results_file_name, 'rb')
        test_score = pickle.load(infile)
        test_progress = pickle.load(infile)
        infile.close()

        minimum = np.min(test_score)
        percentile_25 = np.percentile(test_score, 25)
        percentile_50 = np.percentile(test_score, 50)
        percentile_75 = np.percentile(test_score, 75)
        maximum = np.max(test_score)
        average = np.average(test_score)
        std_dev = np.std(test_score)

        print('\n')
        print('Agent score statistics: \n')
        print(f"{'Minimum':20s} {minimum:6.2f}")
        print(f"{'25th percentile':20s} {percentile_25:6.2f}")
        print(f"{'Median':20s} {percentile_50:6.2f}")
        print(f"{'75th percentile':20s} {percentile_75:6.2f}")
        print(f"{'Maximum':20s} {maximum:6.2f}")
        print(f"{'Average':20s} {average:6.2f}")
        print(f"{'Standard deviation':20s} {std_dev:6.2f}")
        

    def agent_progress_statistics(self):
        infile = open(self.results_file_name, 'rb')
        test_score = pickle.load(infile)
        test_progress = pickle.load(infile)
        infile.close()

        minimum = np.min(test_progress)
        percentile_25 = np.percentile(test_progress, 25)
        percentile_50 = np.percentile(test_progress, 50)
        percentile_75 = np.percentile(test_progress, 75)
        maximum = np.max(test_progress)
        average = np.average(test_progress)
        std_dev = np.std(test_progress)

        print('\n')
        print('Agent progress statistics: \n')
        print(f"{'Minimum':20s} {minimum:6.2f}")
        print(f"{'25th percentile':20s} {percentile_25:6.2f}")
        print(f"{'Median':20s} {percentile_50:6.2f}")
        print(f"{'75th percentile':20s} {percentile_75:6.2f}")
        print(f"{'Maximum':20s} {maximum:6.2f}")
        print(f"{'Average':20s} {average:6.2f}")
        print(f"{'Standard deviation':20s} {std_dev:6.2f}")


    #Display a history of the episode
    def display_agent(self, load_history=False, save_history=False):
        
        if not hasattr(self, 'agent'):
            
            infile = open(self.agent_file_name, 'rb')
            self.agent = pickle.load(infile)
            infile.close()

        if load_history==True:
            
            infile = open(self.history_file_name, 'rb')
            self.state_history = pickle.load(infile)
            self.action_history = pickle.load(infile)
            self.goal_history = pickle.load(infile)
            self.observation_history = pickle.load(infile)
            #if self.env.local_path==True:
            #    self.local_path_history = pickle.load(infile)
            infile.close()
        
        else:

            self.env.reset(save_history=True)
            obs = self.env.observation
            done = False
            score=0

            while not done:
                action = self.agent.choose_action(obs)
                next_obs, reward, done = self.env.take_action(action)
                score += reward
                obs = next_obs
            
            self.state_history = self.env.state_history
            self.action_history = self.env.action_history
            self.goal_history = self.env.goal_history
            self.observation_history = self.env.observation_history
            self.reward_history = self.env.reward_history
            #if self.env.local_path==True:
            #    self.local_path_history = self.env.local_path_history

            if save_history==True:
                outfile = open(self.history_file_name, 'wb')
                pickle.dump(self.state_history, outfile)
                pickle.dump(self.action_history, outfile)
                pickle.dump(self.goal_history, outfile)
                pickle.dump(self.observation_history, outfile)
                pickle.dump(self.reward_history)
                #if self.env.local_path==True:
                #    pickle.dump(self.state_history, outfile)
                outfile.close()
           
        print('\nTotal reward = ', np.sum(self.reward_history))
        print('\nTotal score = ', score)

        image_path = sys.path[0] + '/maps/' + 'circle' + '.png'
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
            
            for sh, ah, gh, rh in zip(self.state_history, self.action_history, self.goal_history, self.reward_history):
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
                #print(rh)
            
                    

if __name__=='__main__':
    
    a = trainingLoop(agent_name='goals_agent_full_state')
    #a.train(save_agent=True)
    #a.learning_curve_score(show_average=True, show_median=True)
    #a.learning_curve_progress(show_average=True, show_median=True)
    #a.test(n_episodes=1000)
    #a.agent_score_statistics()
    #a.agent_progress_statistics()
    #a.histogram_score()
    #a.histogram_progress()

    a.display_agent(load_history=False, save_history=False)
