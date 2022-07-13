from audioop import avg
from statistics import median
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
from environment import environment
import pandas as pd
import time
import random



#from numpy import unique
#from numpy import where
#from sklearn.datasets import make_classification
#from sklearn.mixture import GaussianMixture

def compare_learning_curves_progress(agent_names, legend, legend_title, show_average=True, show_median=True, xaxis='episodes'):
    
    #window = 300

    progress = [[] for _ in range(len(agent_names))]
    #avg = [[] for _ in range(len(agent_names))]
    #std_dev = [[] for _ in range(len(agent_names))]
    #percentile_25 = [[] for _ in range(len(agent_names))]
    median = [[] for _ in range(len(agent_names))]
    #percentile_75 = [[] for _ in range(len(agent_names))]
    steps = [[] for _ in range(len(agent_names))]
    times = [[] for _ in range(len(agent_names))]

    if show_median==True:
        for i in range(len(agent_names)):
            agent_name = agent_names[i]
            train_results_file_name = 'train_results/' + agent_name
            infile = open(train_results_file_name, 'rb')
            
            _ = pickle.load(infile)
            progress = pickle.load(infile)
            times = pickle.load(infile)
            steps = pickle.load(infile)
            infile.close()
        
            median = np.median(progress, axis=0)
            if xaxis=='episodes':
                plt.plot(median)
                plt.xlabel('Episode')
            elif xaxis=='times':
                plt.plot(np.cumsum(np.array(times)), median)
                plt.xlabel('Time')
            elif xaxis=='steps':
                plt.plot(np.cumsum(np.array(steps)), median)
                plt.xlabel('Steps')

        plt.title('Learning curve for median progress')
        plt.ylabel('Progress')
        plt.legend(legend, title=legend_title, loc='lower right')
        #plt.xlim([0,6000])
        plt.show()

    if show_average==True:
        for i in range(len(agent_names)):
            agent_name = agent_names[i]
            train_results_file_name = 'train_results/' + agent_name
            infile = open(train_results_file_name, 'rb')
            
            _ = pickle.load(infile)
            progress = pickle.load(infile)
            times = pickle.load(infile)
            steps = pickle.load(infile)
            infile.close()

            avg = np.average(progress, axis=0)
            if xaxis=='episodes':
                plt.plot(avg)
                plt.xlabel('Episode')
            elif xaxis=='times':
                plt.plot(np.cumsum(np.array(times)), avg)
                plt.xlabel('Time')
            elif xaxis=='steps':
                plt.plot(np.cumsum(np.array(steps)), avg)
                plt.xlabel('Steps')
    
        plt.title('Learning curve for average progress')
        plt.ylabel('Progress')
        plt.legend(legend, title=legend_title, loc='lower right')
        #plt.xlim([0,5000])
        plt.show()
        

def learning_curve_progress(agent_name, show_average=False, show_median=True):
    train_results_file_name = 'train_results/' + agent_name
    infile = open(train_results_file_name, 'rb')
    scores = pickle.load(infile)
    progress = pickle.load(infile)
    infile.close()

    median = np.median(progress, axis=0)
    percentile_25 = np.percentile(progress, 25,axis=0)
    percentile_75 = np.percentile(progress, 75,axis=0)

    avg = np.average(progress, axis=0)
    std_dev = np.std(progress, axis=0)

    if show_median==True:
        #plt.plot(progress)
        plt.plot(median, color='black')
        plt.fill_between(np.arange(len(progress[0])), percentile_25, percentile_75, color='lightblue')
        plt.title('Learning curve')
        plt.xlabel('Episode')
        plt.ylabel('Progress')
        plt.legend(['Median Progress', '25th to 75th percentile'])
        plt.show()

    if show_average==True:
        #plt.plot(progress)
        plt.plot(avg, color='black')
        plt.fill_between(np.arange(len(progress[0])), np.add(avg,std_dev*0.5), np.subtract(avg,std_dev*0.5), color='lightblue')
        plt.title('Learning curve')
        plt.xlabel('Episode')
        plt.ylabel('Progress')
        plt.legend(['Average progress', 'Standard deviation from mean'])
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


def density_plot_score(agent_names, legend, legend_title):
    
    test_score = []
    for a in agent_names:
        results_file_name = 'test_results/' + a
        infile = open(results_file_name, 'rb')
        test_score.append(pickle.load(infile))
        _ = pickle.load(infile)
        infile.close()
    
    sns.displot(test_score,legend=False, kind="kde")
    leg=legend.copy()
    leg.reverse()
    plt.legend(leg, title=legend_title, loc='upper left')
    plt.title('Agent score distribution in testing')
    plt.xlabel('Score')
    plt.ylabel('Density probability')
    plt.show()
    

def density_plot_progress(agent_names, legend, legend_title):
    
    test_progress = []
    for a in agent_names:
        results_file_name = 'test_results/' + a
        infile = open(results_file_name, 'rb')
        _ = pickle.load(infile)
        test_progress.append(pickle.load(infile))
        infile.close()
    
    sns.displot(test_progress,legend=False, kind="kde")
    leg=legend.copy()
    leg.reverse()
    plt.legend(leg, title=legend_title, loc='upper left')
    plt.title('Agent progress distribution in testing')
    plt.xlabel('Progress')
    plt.ylabel('Density probability')
    #plt.xlim([0.8, 1.4])
    plt.show()

def density_plot_action_duration(agent_names, legend, legend_title):
    
    durations = []
    for a in agent_names:
        results_file_name = 'action_durations/' + a
        infile = open(results_file_name, 'rb')
        durations.append(pickle.load(infile))
        infile.close()
    
    for d, a in zip(durations, agent_names):
        print('agent name = ', a)
        print('Average duration without zeros: ', np.average(np.array(d)[np.nonzero(np.array(d))][1:-1])) 
        print('Average duration with zeros: ', np.average(np.array(d)[1:-1]))
        print('Number of non-zero actions: ', len(np.nonzero(np.array(d))[0]), '\n')
    
    sns.displot(durations,legend=False, kind="kde")
    leg=legend.copy()
    leg.reverse()
    plt.legend(leg, title=legend_title, loc='upper left')
    plt.title('Agent action duration')
    plt.xlabel('Action duration')
    plt.ylabel('Density probability')
    #plt.xlim([0.8, 1.4])
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
    test_collision = pickle.load(infile)
    test_max_steps = pickle.load(infile)
    infile.close()

    #Agent statistics over all n
    minimum = np.min(test_progress)
    percentile_25 = np.percentile(test_progress, 25)
    percentile_50 = np.percentile(test_progress, 50)
    percentile_75 = np.percentile(test_progress, 75)
    maximum = np.max(test_progress)
    average = np.average(test_progress)
    std_dev = np.std(test_progress)
    frac_max_steps_reached = np.sum(np.array(test_max_steps))/len(test_max_steps)
    frac_collision = np.sum(np.array(test_collision))/len(test_collision)

    print('Agent progress statistics over all runs: \n')
    print(f"{'Minimum':20s} {minimum:6.3f}")
    print(f"{'25th percentile':20s} {percentile_25:6.3f}")
    print(f"{'Median':20s} {percentile_50:6.3f}")
    print(f"{'75th percentile':20s} {percentile_75:6.3f}")
    print(f"{'Maximum':20s} {maximum:6.3f}")
    print(f"{'Average':20s} {average:6.3f}")
    print(f"{'Standard deviation':20s} {std_dev:6.3f}")
    print(f"{'Percent completed':20s} {frac_max_steps_reached:6.3f}")
    print(f"{'Percent collided':20s}{frac_collision:6.3f}")
    
    #Agent statistics for each n
    print("")
    print("Agent progress statistics for individual runs: \n")
    print(f"{'n':3s}{'| average':10s}{'| median':10s}{'| maximum':10s}{'| minimum':10s}{'| deviation':10s}")
    for n in range(len(test_progress[:,0])):
        print(f"{n:3d}", end='')
        print(f"{np.average(np.round(test_progress, 1), axis=1)[n]:10.3f}", end='')
        print(f"{np.median(np.round(test_progress, 1), axis=1)[n]:10.3f}", end='')
        print(f"{np.max(np.round(test_progress, 1), axis=1)[n]:10.3f}", end='')
        print(f"{np.min(np.round(test_progress, 1), axis=1)[n]:10.3f}", end='')
        print(f"{np.std(np.round(test_progress, 1), axis=1)[n]:10.3f}", end='')
        print("")


def display_train_parameters(agent_name):
    
    infile = open('train_parameters/' + agent_name, 'rb')
    train_parameters_dict = pickle.load(infile)
    infile.close()
    
    print('\nTraining Parameters')
    for key in train_parameters_dict:
        print(key, ': ', train_parameters_dict[key])

    infile = open('environments/' + agent_name, 'rb')
    env_dict = pickle.load(infile)
    infile.close()
    
    print('\nEnvironment Parameters')
    for key in env_dict:
        print(key, ': ', env_dict[key])

    infile = open('agents/' + agent_name + '/' + agent_name + '_params', 'rb')
    agent_dict = pickle.load(infile)
    infile.close()
    
    print('\nAgent Parameters')
    for key in agent_dict:
        print(key, ': ', agent_dict[key])

def display_lap_results(agent_name):

    infile = open('lap_results/' + agent_name, 'rb')
    times = pickle.load(infile)
    collisions = pickle.load(infile)
    infile.close() 
    
    collisions_ = []
    times_ = []
    for n in range(len(times[:,0])):
        frac_succ_ =  np.sum(np.logical_not(np.array(collisions[n])))/len(np.array(collisions[n]))
        if frac_succ_ >= 0.6:
            collisions_.append(collisions[n,:])
            times_.append(times[n,:])
    collisions_ = np.array(collisions_)
    times_ = np.array(times_)

    ave_times = np.average(times[np.logical_not(np.array(collisions))])
    perc_success = np.sum(np.logical_not(np.array(collisions)))/len(np.array(collisions).flatten())
    std_dev = np.std(times[np.logical_not(np.array(collisions))])

    ave_times_ = np.average(times_[np.logical_not(np.array(collisions_))])
    perc_success_ = np.sum(np.logical_not(np.array(collisions_)))/len(np.array(collisions_).flatten())
    std_dev_ = np.std(times_[np.logical_not(np.array(collisions_))])
    
    print('\nLap results over all n, including failed runs:')
    print('Average lap time: ', ave_times)
    print('Lap time std deviation:', std_dev)
    print('Fraction successful laps: ', perc_success)

    print('\nLap results over all n, excluding failed runs:')
    print('Average lap time: ', ave_times_)
    print('Lap time std deviation:', std_dev_)
    print('Fraction successful laps: ', perc_success_)
    
    print("\nAgent lap statistics for individual runs:")

    print(f"{'n':3s}{'| fraction success':20s}{'|avg lap time':12s}")
    for n in range(len(times[:,0])):
        avg_time = np.average(times[n, np.logical_not(np.array(collisions[n]))])
        frac_succ =  np.sum(np.logical_not(np.array(collisions[n])))/len(np.array(collisions[n]))
        print(f"{n:3d}", end='')
        print(f"{frac_succ:20.3f}{avg_time:12.3f}")
        
        #print(f"{np.average(np.round(, 1), axis=1)[n]:10.3f}", end='')
        #print(f"{np.median(np.round(test_progress, 1), axis=1)[n]:10.3f}", end='')
        #print(f"{np.max(np.round(test_progress, 1), axis=1)[n]:10.3f}", end='')
        #print(f"{np.min(np.round(test_progress, 1), axis=1)[n]:10.3f}", end='')
        #print(f"{np.std(np.round(test_progress, 1), axis=1)[n]:10.3f}", end='')
        #print("")

    

def display_moving_agent(agent_name, load_history=False, n=0):

    durations = []

    infile = open('environments/' + agent_name, 'rb')
    env_dict = pickle.load(infile)
    infile.close()
    env_dict['max_steps']=3000
    env = environment(env_dict)
    env.reset(save_history=True, start_condition=[])
    
    infile = open('agents/' + agent_name + '/' + agent_name + '_params', 'rb')
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
    if  main_dict['learning_method']=='actor_critic_com':
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
       
    a.load_weights(agent_name, n)

    if load_history==True:
        env.load_history_func()
    
    else:
        env.reset(save_history=True, start_condition=[])
        obs = env.observation
        done = False
        score=0

        while not done:
            
            start_action = time.time()
            
            if main_dict['learning_method']=='ddpg' or main_dict['learning_method']=='td3':
                action = a.choose_greedy_action(obs)
            else:
                action = a.choose_action(obs)
            
            end_action = time.time()
            
            action_duration = end_action - start_action
            durations.append(action_duration)

            next_obs, reward, done = env.take_action(action)
            score += reward
            obs = next_obs
        
        print('\nTotal score = ', score)

    image_path = sys.path[0] + '/maps/' + env.map_name + '.png'
    im = image.imread(image_path)
    plt.imshow(im, extent=(0,30,0,30))
    
    #outfile=open('action_durations/' + agent_name + 'debug', 'wb')
    #pickle.dump(durations, outfile)
    #outfile.close()
    
    print(np.average(np.array(durations)[np.nonzero(np.array(durations))][1:-1]))  #duration without zeros and first action
    print(np.average(np.array(durations)[1:-1]))
    
    for i in range(len(env.pose_history)):
        plt.cla()
        plt.imshow(im, extent=(0,env.map_width,0,env.map_height))
        sh=env.pose_history[i]
        plt.arrow(sh[0], sh[1], 0.5*math.cos(sh[2]), 0.5*math.sin(sh[2]), head_length=0.5, head_width=0.5, shape='full', ec='None', fc='blue')
        plt.arrow(sh[0], sh[1], 0.5*math.cos(sh[2]+sh[3]), 0.5*math.sin(sh[2]+sh[3]), head_length=0.5, head_width=0.5, shape='full',ec='None', fc='red')
        plt.plot(sh[0], sh[1], 'o')
        wh = env.waypoint_history[i]
        plt.plot(wh[0], wh[1], 'x')
        #gh = env.goal_history[i]
        #plt.plot([gh[0]-env.s, gh[0]+env.s, gh[0]+env.s, gh[0]-env.s, gh[0]-env.s], [gh[1]-env.s, gh[1]-env.s, gh[1]+env.s, gh[1]+env.s, gh[1]-env.s], 'r')
        
        if env.local_path==True:
            lph = env.local_path_history[i]
            plt.plot(lph[0], lph[1])
        
        plt.plot(env.rx, env.ry)

        cph = env.closest_point_history[i]
        plt.plot(env.rx[cph], env.ry[cph], 'x')
        
        if env_dict['lidar_dict']['is_lidar']==True:
            lh = env.lidar_coords_history[i]
            for coord in lh:
                plt.plot(coord[0], coord[1], 'xb')
        
        plt.plot(np.array(env.pose_history)[0:i,0], np.array(env.pose_history)[0:i,1])
        
        ph = env.progress_history[i]
        ah = env.action_step_history[i]
        
        #wpt, v_ref = env.convert_action_to_coord(strategy='local', action=ah)
        #print('v =', sh[4])

        #plt.legend(["position", "waypoint", "goal area", "heading", "steering angle"])
        plt.xlabel('x coordinate')
        plt.ylabel('y coordinate')
        plt.xlim([0,env.map_width])
        plt.ylim([0,env.map_height])
        #plt.grid(True)
        plt.title('Episode history')
        #print('Progress = ', ph)
        plt.pause(0.001)

def display_path(agent_name, load_history=False, n=0):
    
    agent_file_name = 'agents/' + agent_name
    environment_name = 'environments/' + agent_name
    #initial_condition={'x':16, 'y':28, 'v':7, 'delta':0, 'theta':np.pi, 'goal':1}
    initial_condition = []

    infile = open('environments/' + agent_name, 'rb')
    env_dict = pickle.load(infile)
    infile.close()
    env_dict['max_steps']=3000
    env = environment(env_dict)
    env.reset(save_history=True, start_condition=[])

    infile = open('agents/' + agent_name + '/' + agent_name + '_params', 'rb')
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
    if  main_dict['learning_method']=='actor_critic_com':
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
        
    a.load_weights(agent_name, n)
    
    if load_history==True:
        env.load_history_func()
        
    else:
        env.reset(save_history=True, start_condition=[])
        obs = env.observation
        done = False
        score=0

        while not done:
            if main_dict['learning_method']=='ddpg' or main_dict['learning_method']=='td3':
                action = a.choose_greedy_action(obs)
            else:
                action = a.choose_action(obs)

            next_obs, reward, done = env.take_action(action)
            score += reward
            obs = next_obs
        
        print('Total score = ', score)
        print('Progress = ', env.progress)

    image_path = sys.path[0] + '/maps/' + env.map_name + '.png'
    im = image.imread(image_path)
    plt.imshow(im, extent=(0,env.map_width,0,env.map_height))
    plt.plot(np.array(env.pose_history)[:,0], np.array(env.pose_history)[:,1])
    plt.plot(env.rx, env.ry)
    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    plt.xlim([0,env.map_width])
    plt.ylim([0,env.map_height])
    #plt.grid(True)
    plt.title('Agent path')
    plt.show()


def display_collision_distribution(agent_name):

    results_file_name = 'test_results/' + agent_name

    infile = open('environments/' + agent_name, 'rb')
    env_dict = pickle.load(infile)
    infile.close()

    infile = open(results_file_name, 'rb')
    test_score = pickle.load(infile)
    test_progress = pickle.load(infile)
    test_collision = pickle.load(infile)
    test_max_steps = pickle.load(infile)
    terminal_poses = pickle.load(infile)
    infile.close()

    image_path = sys.path[0] + '/maps/' + env_dict['map_name'] + '.png'
    im = image.imread(image_path)
    plt.imshow(im)
    plt.plot(np.array(terminal_poses)[:,0], np.array(terminal_poses)[:,1], 'x')
    #sns.jointplot(x=np.array(terminal_poses)[:,0],y=np.array(terminal_poses)[:,1], kind="hex", alpha=0.5)
    plt.show()


def display_path_multiple(agent_names, ns, legend_title, legend):
    

    for agent_name, n, i in zip(agent_names, ns, range(len(agent_names))):

        infile = open('environments/' + agent_name, 'rb')
        env_dict = pickle.load(infile)
        infile.close()
        #env_dict['max_steps']=3000
        env = environment(env_dict)
        env.reset(save_history=True, start_condition=[])

        infile = open('agents/' + agent_name + '/' + agent_name + '_params', 'rb')
        agent_dict = pickle.load(infile)
        infile.close()

        infile = open('train_parameters/' + agent_name, 'rb')
        main_dict = pickle.load(infile)
        infile.close()
          
        if i==0:
            infile = open('test_initial_condition/' + env_dict['map_name'], 'rb')
            start_conditions = pickle.load(infile)
            infile.close()
            start_pose = random.choice(start_conditions)

        if main_dict['learning_method']=='dqn':
            agent_dict['epsilon'] = 0
            a = agent_dqn.agent(agent_dict)
        if main_dict['learning_method']=='reinforce':
            a = agent_reinforce.PolicyGradientAgent(agent_dict)
        if main_dict['learning_method']=='actor_critic_sep':
            a = agent_actor_critic.actor_critic_separated(agent_dict)
        if  main_dict['learning_method']=='actor_critic_com':
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
            
        a.load_weights(agent_name, n)

        env.reset(save_history=True, start_condition=start_pose)
        obs = env.observation
        done = False
        score=0

        while not done:
            if main_dict['learning_method']=='ddpg' or main_dict['learning_method']=='td3':
                action = a.choose_greedy_action(obs)
            else:
                action = a.choose_action(obs)

            next_obs, reward, done = env.take_action(action)
            score += reward
            obs = next_obs

            if env.progress>=0.95:
                done=True
            

        print('Total score = ', score)
        print('Progress = ', env.progress)

        image_path = sys.path[0] + '/maps/' + env.map_name + '.png'
        im = image.imread(image_path)
        plt.imshow(im, extent=(0,env.map_width,0,env.map_height))
        plt.plot(np.array(env.pose_history)[:,0], np.array(env.pose_history)[:,1], linewidth=2.5)
    
    #plt.legend(legend, title=legend_title)
    #plt.xlabel('x [m]')
    #plt.ylabel('y [m]')
    plt.xticks([])
    plt.yticks([])
    plt.xlim([0,env.map_width])
    plt.ylim([0,env.map_height])
    #plt.grid(True)
    #plt.title('Agent path')
    plt.show()