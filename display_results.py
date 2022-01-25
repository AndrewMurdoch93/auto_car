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


def learning_curve_score(agent_name, show_average=False, show_median=True):
    window = 100
    
    train_results_file_name = 'train_results/' + agent_name
    infile = open(train_results_file_name, 'rb')
    scores = pickle.load(infile)
    progress = pickle.load(infile)
    infile.close()

    avg_scores = []
    std_dev = []
    percentile_25 = []
    median = []
    percentile_75 = []

    
    for i in range(len(scores)):
        if i <= window:
            x = 0
        else:
            x = i-window 
        
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
        

def learning_curve_progress(agent_name, show_average=False, show_median=True):
    window=100

    train_results_file_name = 'train_results/' + agent_name
    infile = open(train_results_file_name, 'rb')
    scores = pickle.load(infile)
    progress = pickle.load(infile)
    infile.close()

    avg_scores = []
    std_dev = []
    percentile_25 = []
    median = []
    percentile_75 = []

    for i in range(len(progress)):
        if i <= window:
            x = 0
        else:
            x = i-window 
        
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


def histogram_score(agent_name):

    results_file_name = 'test_results/' + agent_name
    infile = open(results_file_name, 'rb')
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
    

def histogram_progress(agent_name):

    results_file_name = 'test_results/' + agent_name
    infile = open(results_file_name, 'rb')
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
    plt.xlabel('Fraction of track completed')
    plt.ylabel('Number of agents')
    plt.legend(['25th percentile', 'Median', '75th percentile', 'Agent progress'])
    plt.show()


def agent_score_statistics(agent_name):
    
    results_file_name = 'test_results/' + agent_name
    infile = open(results_file_name, 'rb')
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
    

def agent_progress_statistics(agent_name):
    
    results_file_name = 'test_results/' + agent_name
    infile = open(results_file_name, 'rb')
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
    frac_complete = np.sum(np.array(test_progress)>=1)/len(test_progress)

    print('\n')
    print('Agent progress statistics: \n')
    print(f"{'Minimum':20s} {minimum:6.2f}")
    print(f"{'25th percentile':20s} {percentile_25:6.2f}")
    print(f"{'Median':20s} {percentile_50:6.2f}")
    print(f"{'75th percentile':20s} {percentile_75:6.2f}")
    print(f"{'Maximum':20s} {maximum:6.2f}")
    print(f"{'Average':20s} {average:6.2f}")
    print(f"{'Standard deviation':20s} {std_dev:6.2f}")
    print(f"{'Fraction completed':20s} {frac_complete:6.2f}")


def display_moving_agent(agent_name, load_history=False, save_history=False):
    
    agent_file_name = 'agents/' + agent_name
    history_file_name = 'test_history/' + agent_name
    environment_name = 'environments/' + agent_name

    infile = open(agent_file_name, 'rb')
    agent = pickle.load(infile)
    infile.close()

    infile = open(environment_name, 'rb')
    env = pickle.load(infile)
    infile.close()

    if load_history==True:
        infile = open(history_file_name, 'rb')
        state_history = pickle.load(infile)
        action_history = pickle.load(infile)
        goal_history = pickle.load(infile)
        observation_history = pickle.load(infile)
        reward_history = pickle.load(infile)
        score = pickle.load(infile)
        if env.local_path==True:
            local_path_history = pickle.load(infile)
        infile.close()
    
    else:
        env.reset(save_history=True)
        obs = env.observation
        done = False
        score=0

        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done = env.take_action(action)
            score += reward
            obs = next_obs
        
        state_history = env.state_history
        action_history = env.action_history
        goal_history = env.goal_history
        observation_history = env.observation_history
        reward_history = env.reward_history
        if env.local_path==True:
            local_path_history = env.local_path_history

        if save_history==True:
            outfile = open(history_file_name, 'wb')
            pickle.dump(state_history, outfile)
            pickle.dump(action_history, outfile)
            pickle.dump(goal_history, outfile)
            pickle.dump(observation_history, outfile)
            pickle.dump(reward_history, outfile)
            pickle.dump(score)
            if env.local_path==True:
                pickle.dump(env.local_path_history, outfile)
            outfile.close()
        
    print('\nTotal score = ', score)

    image_path = sys.path[0] + '/maps/' + env.map_name + '.png'
    im = image.imread(image_path)
    plt.imshow(im, extent=(0,30,0,30))

    if env.local_path==True:
        
        for sh, ah, gh, lph in zip(state_history, action_history, goal_history, local_path_history):
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
        
        for sh, ah, gh, rh in zip(state_history, action_history, goal_history, reward_history):
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
    
    print(score)