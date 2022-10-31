from audioop import avg
from configparser import BasicInterpolation
from re import S
from statistics import median
from xmlrpc.client import ProtocolError
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
import math
from matplotlib import image
from PIL import Image
import time
import seaborn as sns
from environment import environment
import pandas as pd
import time
import random
from matplotlib.ticker import FormatStrFormatter
import mapping
from PIL import Image, ImageOps, ImageDraw, ImageFilter


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


def evaluation(agent_names, legend, legend_title):
    n = 0
    fig, axs = plt.subplots(2, sharex=True)
    for i in range(len(agent_names)):
        
        agent_name = agent_names[i]
        train_results_file_name = 'evaluation_results/' + agent_name
        
        infile = open(train_results_file_name, 'rb')
        eval_steps = pickle.load(infile)
        eval_lap_times = pickle.load(infile)
        eval_collisions = pickle.load(infile)
        infile.close()

        last_row = np.where(eval_lap_times[n]==0)[0][0]

        eval_steps = eval_steps[n][0:last_row]
        eval_lap_times = eval_lap_times[n][0:last_row]
        eval_collisions = eval_collisions[n][0:last_row]

        avg_lap_time = np.zeros(len(eval_lap_times))
        avg_collision = np.zeros(len(eval_lap_times))

        for i in range(len(eval_lap_times)):
            avg_lap_time[i] = np.average(eval_lap_times[i][np.where(eval_collisions[i]==False)])
            avg_collision[i] = np.average(eval_collisions[i])
      
    
        axs[0].plot(eval_steps, avg_lap_time)
        axs[1].plot(eval_steps, avg_collision)

    axs[0].set(ylabel='Lap time [s]', title='Average lap time')
    axs[1].set(xlabel='Training simulation step', ylabel='fraction collisions', title='Collision rate')
    axs[1].legend(legend, title=legend_title, loc='upper right')

    plt.show()
        

def learning_curve_lap_time(agent_names, legend, legend_title, show_average=True, show_median=True):
    
    window = 500
    steps = [[] for _ in range(len(agent_names))]
    steps_x_axis = [[] for _ in range(len(agent_names))]
    steps_no_coll = [[] for _ in range(len(agent_names))]
    avg_steps_no_coll = [[] for _ in range(len(agent_names))]
    collisions = [[] for _ in range(len(agent_names))]
    avg_time = [[] for _ in range(len(agent_names))]
    avg_coll = [[] for _ in range(len(agent_names))]

    for i in range(len(agent_names)):
        agent_name = agent_names[i]
        train_results_file_name = 'train_results/' + agent_name
        infile = open(train_results_file_name, 'rb')
        
        _ = pickle.load(infile)
        _ = pickle.load(infile)
        _ = pickle.load(infile)
        steps[i] = pickle.load(infile)
        collisions[i] = pickle.load(infile)
        infile.close()
        
        for j in range(len(collisions[0][0])):
            if j <= window:
                x = 0
            else:
                x = j-window 
            avg_coll[i].append(np.mean(collisions[i][0][x:j+1]))
            avg_time[i].append(np.mean(steps[i][0][x:j+1]))
    
        steps_x_axis[i] = np.cumsum(steps[i])[np.logical_not(collisions[i][0])]
        steps_no_coll[i] = steps[i][0][np.logical_not(collisions[i][0])]

        for j in range(len(steps_x_axis[0])):
            if j <= window:
                x = 0
            else:
                x = j-window 
            avg_steps_no_coll[i].append(np.mean(steps_no_coll[i][x:j+1]))
    
    plt.figure(1)
    for i in range(len(agent_names)):
        end_episode =  np.where(steps[i][0]==0)[0][0]
        #plt.plot(np.cumsum(steps[0]), steps[0], 'x')
        #plt.plot(steps[0][np.where(np.logical_not(collisions[0]))], 'x')
        plt.plot(np.cumsum(steps[0])[0:end_episode],  avg_coll[i][0:end_episode])
        plt.xlabel('Steps')
        plt.title('Average collision rate per episode')
        plt.ylabel('Average collision rate per episode')
        plt.legend(legend, title=legend_title, loc='upper right')
        #plt.xlim([0,6000])

    plt.figure(2)
    for i in range(len(agent_names)):
        end_episode_no_coll = np.where(steps_no_coll[i]==0)[0][0]
        #plt.plot(np.cumsum(steps[0]), steps[0], 'x')
        #plt.plot(steps[0][np.where(np.logical_not(collisions[0]))], 'x')
        plt.plot(steps_x_axis[i][0:end_episode_no_coll],   np.array(avg_steps_no_coll[i][0:end_episode_no_coll])*0.01 )
        plt.xlabel('Steps')
        plt.title('Average time per episode without collisions')
        plt.ylabel('time [s]')
        plt.legend(legend, title=legend_title, loc='upper right')
        #plt.xlim([0,6000])
    
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

def graph_lap_results(agent_names):
    
    results_dict = {}
    results = {}
    envs = {}
    lap_data = []
    success_data = []
    data = []

    for agent in agent_names:
        infile = open('lap_results/' + agent, 'rb')
        results_dict['times'] = pickle.load(infile)
        results_dict['collisions'] = pickle.load(infile)
        results[agent] = results_dict
        infile.close()

        infile = open('environments/' + agent, 'rb')
        env_dict = pickle.load(infile)
        envs[agent] = env_dict
        infile.close()

        pass
        
        if env_dict['velocity_control']==True and env_dict['steer_control_dict']['steering_control']==True:
            architecture = 'steering and velocity control'

        elif env_dict['velocity_control']==True and env_dict['steer_control_dict']['steering_control']==False:
            architecture = 'velocity control'

        elif env_dict['velocity_control']==False and env_dict['steer_control_dict']['steering_control']==True:
            architecture = 'steering control'

        elif env_dict['velocity_control']==False and env_dict['steer_control_dict']['steering_control']==False:
            architecture = 'no controller'
        

        for lap_time, collision in zip(results_dict['times'].flatten(), results_dict['collisions'].flatten()):
            if collision==False:
                lap_data.append({'agent_name':agent, 'architecture':architecture, 'map':env_dict['map_name'], 'lap_time':lap_time})
            
            success_data.append({'agent_name':agent, 'architecture':architecture, 'map':env_dict['map_name'], 'success':np.logical_not(collision)})
            data.append({'agent_name':agent, 'architecture':architecture, 'map':env_dict['map_name'], 'success':np.logical_not(collision), 'lap_time':lap_time})
    
    pass
    df_lap = pd.DataFrame(lap_data)
    df_success = pd.DataFrame(success_data)
    df = pd.DataFrame(data)

    archs = df['architecture'].unique()
    maps = df['map'].unique()

    arch_lap_results = np.zeros((len(archs), len(maps)))
    arch_success_results = np.zeros((len(archs), len(maps)))

    for a_idx, a in enumerate(archs):
        for m_idx, m in enumerate(maps): 
            arch_lap_results[a_idx, m_idx] = np.average(np.array(df[np.logical_and(np.logical_and(df['architecture']==a, df['map']==m), df['success']==True)]['lap_time']))
            arch_success_results[a_idx, m_idx] = np.average(np.array(df[np.logical_and(df['architecture']==a, df['map']==m)]['success']))

    x = ['circle', 'columbia', 'porto', 'berlin', 'torino', 'redbull ring']
    
    w=0.2
    bar1 = np.arange(len(maps))
    bar2 = [i+w for i in bar1]
    bar3 = [i+w for i in bar2]
    bar4 = [i+w for i in bar3]
    

    fig, axs = plt.subplots(2, sharex=True)
    axs[0].bar(bar1, arch_lap_results[0], w, label=archs[0])
    axs[0].bar(bar2, arch_lap_results[1], w, label=archs[1])
    axs[0].bar(bar3, arch_lap_results[2], w, label=archs[2])
    axs[0].bar(bar4, arch_lap_results[3], w, label=archs[3])
    axs[0].set(ylabel='Lap time [s]')
    #axs[0].set_ylim([0,24])
    #axs[0].legend(archs, loc='lower right')


    axs[1].bar(bar1, arch_success_results[0], w, label=archs[0])
    axs[1].bar(bar2, arch_success_results[1], w, label=archs[1])
    axs[1].bar(bar3, arch_success_results[2], w, label=archs[2])
    axs[1].bar(bar4, arch_success_results[3], w, label=archs[3])
    axs[1].set_xticks(bar1+w, x)
    axs[1].set(xlabel='Track', ylabel='Fraction successful laps')
    axs[1].set_ylim([0,1])
    axs[1].legend(archs, loc='lower right')
    plt.show()

    plt.figure(1)
    plt.bar(bar1, arch_lap_results[0], w, label=archs[0])
    plt.bar(bar2, arch_lap_results[1], w, label=archs[1])
    plt.bar(bar3, arch_lap_results[2], w, label=archs[2])
    plt.bar(bar4, arch_lap_results[3], w, label=archs[3])
    plt.ylabel('Lap time [s]')
    plt.xlabel('Track')
    #plt.ylim([0,24])
    plt.xticks(bar1+w, x)
    plt.legend(archs, loc='lower right')
    
    plt.figure(2)
    plt.bar(bar1, arch_success_results[0], w, label=archs[0])
    plt.bar(bar2, arch_success_results[1], w, label=archs[1])
    plt.bar(bar3, arch_success_results[2], w, label=archs[2])
    plt.bar(bar4, arch_success_results[3], w, label=archs[3])
    plt.xticks(bar1+w, x)
    plt.xlabel('Track')
    plt.ylabel('Fraction successful laps')
    plt.ylim([0,1])
    plt.legend(archs, loc='lower right')
    plt.show()

    #axs[0].barplot(x='map', y='lap_time', hue='architecture', capsize=.2, data=df_lap)
    # p.set_xlabel('% variation from original ' + param + ' value')
    # p.set_ylabel('fraction successful laps')
    # handles, _ = p.get_legend_handles_labels()
    # p.legend(handles, legend ,title=legend_title, loc='lower left')
    # plt.title('Lap success rate for ' + param + ' mismatch')
    #plt.show()

    #p = sns.barplot(x='map', y='success', hue='architecture', capsize=.2, data=df_success)
    # p.set_xlabel('% variation from original ' + param + ' value')
    # p.set_ylabel('fraction successful laps')
    # handles, _ = p.get_legend_handles_labels()
    # p.legend(handles, legend ,title=legend_title, loc='lower left')
    # plt.title('Lap success rate for ' + param + ' mismatch')
    #plt.show()

agent_names = ['circle_pete_sv', 'circle_pete_s', 'circle_pete_v', 'circle_ete', 
            'columbia_pete_sv', 'columbia_pete_s', 'columbia_pete_v', 'columbia_ete',
            'porto_pete_sv', 'porto_pete_s', 'porto_pete_v', 'porto_ete',
            'berlin_pete_sv', 'berlin_pete_s', 'berlin_pete_v', 'berlin_ete',
            'torino_pete_sv', 'torino_pete_s', 'torino_pete_v', 'torino_ete',
            'redbull_ring_pete_sv', 'redbull_ring_pete_s', 'redbull_ring_pete_v', 'redbull_ring_ete']      

# agent_names = ['circle_pete_sv_1', 'circle_pete_s_1', 'circle_pete_v_1', 'circle_ete_1', 
#          'columbia_pete_sv_1', 'columbia_pete_s_1', 'columbia_pete_v_1', 'columbia_ete_1',
#          'porto_pete_sv_1', 'porto_pete_s_1', 'porto_pete_v_1', 'porto_ete_1',
#          'berlin_pete_sv_1', 'berlin_pete_s_1', 'berlin_pete_v_1', 'berlin_ete_1',
#          'torino_pete_sv_1', 'torino_pete_s_1', 'torino_pete_v_1', 'torino_ete_1',
#          'redbull_ring_pete_sv_1', 'redbull_ring_pete_s_1', 'redbull_ring_pete_v_1', 'redbull_ring_ete_1']   

#agent_names = ['porto_pete_sv', 'porto_pete_s', 'porto_pete_v', 'porto_ete']

#graph_lap_results(agent_names)



def graph_lap_results_mismatch(agent_names, mismatch_parameter, title):
    
    results_dict = {}
    results = {}
    envs = {}
    lap_data = []
    success_data = []
    data = []

    for agent in agent_names:
        
        infile = open('lap_results_mismatch/' + agent + '/' + mismatch_parameter, 'rb')
        results_dict = pickle.load(infile)
        infile.close() 

        infile = open('environments/' + agent, 'rb')
        env_dict = pickle.load(infile)
        envs[agent] = env_dict
        infile.close()
    
        if env_dict['velocity_control']==True and env_dict['steer_control_dict']['steering_control']==True:
            architecture = 'steering and velocity control'

        elif env_dict['velocity_control']==True and env_dict['steer_control_dict']['steering_control']==False:
            architecture = 'velocity control'

        elif env_dict['velocity_control']==False and env_dict['steer_control_dict']['steering_control']==True:
            architecture = 'steering control'

        elif env_dict['velocity_control']==False and env_dict['steer_control_dict']['steering_control']==False:
            architecture = 'no controller'
        
        for i in range(len(results_dict['times_results'][0])):
            for j in range(len(results_dict['times_results'][0][0])):
                data.append({'agent_name':agent, 'architecture':architecture, 'map':env_dict['map_name'], 'mismatch_parameter':mismatch_parameter,
                        'frac_vary':results_dict['frac_variation'][i], 'success':np.logical_not(results_dict['collision_results'][0][i][j]), 
                        'lap_time':results_dict['times_results'][0][i][j]})

    df = pd.DataFrame(data)
    
    s = 20
    archs = df['architecture'].unique()
    maps = df['map'].unique()
    vary = df['frac_vary'].unique()
    vary_select = np.array([10, s])

    arch_lap_results = np.zeros((len(vary_select), len(archs), len(maps)))
    arch_success_results = np.zeros((len(vary_select), len(archs), len(maps)))

    for v_idx, v in  enumerate(vary_select):
        for a_idx, a in enumerate(archs):
             for m_idx, m in enumerate(maps): 
                condition_time = np.logical_and(np.logical_and(np.logical_and(df['architecture']==a, df['map']==m), df['success']==True), df['frac_vary']==vary[v])
                condition_succ = np.logical_and(np.logical_and(df['architecture']==a, df['map']==m), df['frac_vary']==vary[v])
                
                arch_lap_results[v_idx, a_idx, m_idx] = np.average(np.array(df[condition_time]['lap_time']))
                arch_success_results[v_idx, a_idx, m_idx] = np.average(np.array(df[condition_succ]['success']))

    x = ['circle', 'columbia', 'porto', 'berlin', 'torino', 'redbull ring']
    
    time_errors = arch_lap_results[1]-arch_lap_results[0]
    succ_errors = arch_success_results[1]-arch_success_results[0]

    #x = maps

    w=0.2
    bar1 = np.arange(len(maps))
    bar2 = [i+w for i in bar1]
    bar3 = [i+w for i in bar2]
    bar4 = [i+w for i in bar3]
    
    #error bar method
    #fig, axs = plt.subplots(2, sharex=True)
    # axs[0].bar(bar1, arch_lap_results[0,0], w, label=archs[0], yerr=[np.zeros(len(time_errors[0])), time_errors[0]])
    # axs[0].bar(bar2, arch_lap_results[0,1], w, label=archs[1], yerr=[np.zeros(len(time_errors[0])), time_errors[1]])
    # axs[0].bar(bar3, arch_lap_results[0,2], w, label=archs[2], yerr=[np.zeros(len(time_errors[0])), time_errors[2]])
    # axs[0].set(ylabel='Lap time [s]')
    # axs[0].set_ylim([5,18])
    # axs[0].legend(archs)
    # axs[1].bar(bar1, arch_success_results[0,0], w, label=archs[0], yerr=[np.zeros(len(succ_errors[0])), succ_errors[0]])
    # axs[1].bar(bar2, arch_success_results[0,1], w, label=archs[1], yerr=[np.zeros(len(succ_errors[0])), succ_errors[1]])
    # axs[1].bar(bar3, arch_success_results[0,2], w, label=archs[2], yerr=[np.zeros(len(succ_errors[0])), succ_errors[2]])
    # axs[1].set_xticks(bar1+w, x)
    # axs[1].set(xlabel='Track', ylabel='Fraction successful laps')
    # axs[1].set_ylim([0,1.1])
    # plt.show()

    #new figures for every parameter
    # for idx, _ in enumerate(vary_select):
    #     fig, axs = plt.subplots(2, sharex=True)
    #     axs[0].bar(bar1, arch_lap_results[idx,0], w, label=archs[0])
    #     axs[0].bar(bar2, arch_lap_results[idx,1], w, label=archs[1])
    #     axs[0].bar(bar3, arch_lap_results[idx,2], w, label=archs[2])
    #     axs[0].set(ylabel='Lap time [s]')
    #     axs[0].set(ylabel='Lap time [s]')
    #     axs[0].set_ylim([5,18])
    #     axs[0].legend(archs)

    #     axs[1].bar(bar1, arch_success_results[idx,0], w, label=archs[0])
    #     axs[1].bar(bar2, arch_success_results[idx,1], w, label=archs[1])
    #     axs[1].bar(bar3, arch_success_results[idx,2], w, label=archs[2])
    #     axs[1].set_xticks(bar1+w, x)
    #     axs[1].set(xlabel='Track', ylabel='Fraction successful laps')
    #     axs[1].set_ylim([0,1.1])
    #     axs[0].set(ylabel='Lap time [s]')
    #     axs[0].set_ylim([5,18])
    #     axs[0].legend(archs)
    
    #     plt.show()
    
    #Only plot fraction success
    fig, axs = plt.subplots(2, sharex=True)
    axs[0].bar(bar1, arch_success_results[0,0], w, label=archs[0])
    axs[0].bar(bar2, arch_success_results[0,1], w, label=archs[1])
    axs[0].bar(bar3, arch_success_results[0,2], w, label=archs[2])
    axs[0].bar(bar4, arch_success_results[0,3], w, label=archs[3])
    axs[0].set_title('(a) No model-mismatch')
    axs[0].set_ylabel('Fraction successful laps')
    axs[0].legend(archs, loc='lower right')
    #axs[0].set_ylim([0,1.8])

    axs[1].bar(bar1, arch_success_results[1,0], w, label=archs[0])
    axs[1].bar(bar2, arch_success_results[1,1], w, label=archs[1])
    axs[1].bar(bar3, arch_success_results[1,2], w, label=archs[2])
    axs[1].bar(bar4, arch_success_results[1,3], w, label=archs[3])
    axs[1].set_xticks(bar1+w, x)
    axs[1].set_title('(b)' + title)
    axs[1].set_xlabel('Track')
    axs[1].set_ylabel('Fraction successful laps')
    #axs[1].legend(archs, loc='lower right')
    plt.show()


    plt.bar(bar1, arch_success_results[1,0], w, label=archs[0])
    plt.bar(bar2, arch_success_results[1,1], w, label=archs[1])
    plt.bar(bar3, arch_success_results[1,2], w, label=archs[2])
    plt.bar(bar4, arch_success_results[1,3], w, label=archs[3])
    plt.xticks(bar1+w, x)
    plt.xlabel('Track')
    plt.ylabel('Fraction successful laps')
    plt.ylim([0,1])
    plt.legend(archs, loc='lower right')
    plt.show()


agent_names = ['circle_pete_sv', 'circle_pete_s', 'circle_pete_v', 'circle_ete', 
            'columbia_pete_sv', 'columbia_pete_s', 'columbia_pete_v', 'columbia_ete',
            'porto_pete_sv', 'porto_pete_s', 'porto_pete_v', 'porto_ete',
            'berlin_pete_sv', 'berlin_pete_s', 'berlin_pete_v', 'berlin_ete',
            'torino_pete_sv', 'torino_pete_s', 'torino_pete_v', 'torino_ete',
            'redbull_ring_pete_sv', 'redbull_ring_pete_s', 'redbull_ring_pete_v', 'redbull_ring_ete']      

# agent_names = ['circle_pete_sv_1', 'circle_pete_s_1', 'circle_pete_v_1', 'circle_ete_1', 
#          'columbia_pete_sv_1', 'columbia_pete_s_1', 'columbia_pete_v_1', 'columbia_ete_1',
#          'porto_pete_sv_1', 'porto_pete_s_1', 'porto_pete_v_1', 'porto_ete_1',
#          'berlin_pete_sv_1', 'berlin_pete_s_1', 'berlin_pete_v_1', 'berlin_ete_1',
#          'torino_pete_sv_1', 'torino_pete_s_1', 'torino_pete_v_1', 'torino_ete_1',
#          'redbull_ring_pete_sv_1', 'redbull_ring_pete_s_1', 'redbull_ring_pete_v_1', 'redbull_ring_ete_1']   

#agent_names = ['porto_pete_sv', 'porto_pete_s', 'porto_pete_v', 'porto_ete']


#graph_lap_results_mismatch(agent_names, 'C_Sf', title='Front tyre stiffness coefficient 20% higher than expected')




def display_lap_mismatch_results(agent_names, parameters, legend_title, legend, plot_titles):
    
    fig, axs = plt.subplots(len(parameters), sharex=True)
    numbering = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']

    for j, parameter in enumerate(parameters):
        for agent in agent_names:
            
            #infile = open('lap_results_mismatch/' + agent + '_new/' + parameter, 'rb')
            infile = open('lap_results_mismatch/' + agent + '/' + parameter, 'rb')
            results_dict = pickle.load(infile)
            infile.close() 

            n_episodes = len(results_dict['collision_results'][0,0,:])
            n_param = len(results_dict['collision_results'][0,:,0])
            n_runs = len(results_dict['collision_results'][:,0,0])

            avg = np.zeros(n_param)
            dev = np.zeros(n_param)

            for i in range(n_param):
                avg[i] = np.round(np.sum(np.logical_not(results_dict['collision_results'][:,i,:]))/(n_episodes*n_runs), 2)
                failures = np.count_nonzero(results_dict['collision_results'][:,0,:].flatten())
                successes = n_episodes - failures
                dev[i] = np.sqrt(n_episodes*(successes/n_episodes)*((failures)/n_episodes))/(n_episodes*n_runs)

            axs[j].grid()
            axs[j].set_ylim([0,1.1])
            axs[j].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axs[j].plot(results_dict['frac_variation']*100, avg)
            #plt.fill_between(results_dict['frac_variation']*100, avg-dev, avg+dev, alpha=0.25)
            axs[j].set(ylabel='fraction successful laps')
            #axs.yaxis.set_major_formatter(plt.ticker.FormatStrFormatter('%.2f'))

        axs[j].set_title('(' + numbering[j] + ') ' + plot_titles[j])
    axs[j].set(xlabel='% variation from original value')
    axs[j].legend(legend, title=legend_title, loc='lower right')
    plt.show()

    # results = []
    # data = []

    # for agent in agent_names:
    #     for param in parameters:
    #         infile = open('lap_results_mismatch/' + agent + '_new/' + param, 'rb')
    #         results_dict = pickle.load(infile)
    #         results.append(results_dict)
    #         infile.close() 
            
    #         n_episodes = len(results_dict['collision_results'][0,0,:])
    #         n_param = len(results_dict['collision_results'][0,:,0])
    #         n_runs = len(results_dict['collision_results'][:,0,0])

    #         y = np.zeros((n_param, n_runs))

    #         for i in range(n_param):
    #             y = np.sum(np.logical_not(results_dict['collision_results'][:,i,:]))/(n_episodes*n_runs)
    #             data.append({'agent_name':agent, 'parameter':param,'frac_variation':round(results_dict['frac_variation'][i],2)*100, 'success_rate':y})

    # df = pd.DataFrame(data)
    
    # if len(parameters)>1:
    #     p = sns.factorplot(x='frac_variation', y='success_rate', hue='agent_name', row='parameter',data=df)
    # else:
    #     p = sns.lineplot(x='frac_variation', y='success_rate', hue='agent_name', data=df)
    #     p.set_xlabel('% variation from original ' + param + ' value')
    #     p.set_ylabel('fraction successful laps')
    #     handles, _ = p.get_legend_handles_labels()
    #     p.legend(handles, legend ,title=legend_title, loc='lower left')
    #     plt.title('Lap success rate for ' + param + ' mismatch')
    # plt.show()

    

def display_lap_mismatch_results_box(agent_names, parameters, legend_title, legend):

    results = []
    results_data = []
    data = []

    for agent in agent_names:
        for param in parameters:
            infile = open('lap_results_mismatch/' + agent + '/' + param, 'rb')
            results_dict = pickle.load(infile)
            results.append(results_dict)
            infile.close() 
            
            n_episodes = len(results_dict['collision_results'][0,0,:])
            n_param = len(results_dict['collision_results'][0,:,0])
            n_runs = len(results_dict['collision_results'][:,0,0])

            y = np.zeros((n_param, n_runs))

            for i in range(n_param):
                y = np.sum(np.logical_not(results_dict['collision_results'][:,i,:]), axis=1)/(n_episodes)

                for n, success in enumerate(y):
                    data.append({'agent_name':agent, 'parameter':param,'frac_variation':round(results_dict['frac_variation'][i],2)*100, 'n':n, 'success_rate':success})

    df = pd.DataFrame(data)
    df['frac_var_cut'] = pd.cut(df['frac_variation'], 5)
    
    if len(parameters)>1:
        p = sns.factorplot(x='frac_var_cut', y='success_rate', hue='agent_name', row='parameter', kind='box',data=df)
    else:
        p = sns.boxplot(x='frac_var_cut', y='success_rate', hue='agent_name',data=df)
        p.set_xlabel('% variation from original ' + param + ' value')
        p.set_ylabel('fraction successful laps')
        handles, _ = p.get_legend_handles_labels()
        p.legend(handles, legend)
        plt.title('Lap success rate for ' + param + ' mismatch')
    plt.show()

    #plt.title('Lap success result for ' + parameter + ' parameter mismatch')
    #plt.xlabel('% variation from original ' + parameter + ' value')
    #plt.ylabel('fraction successful laps')
    #plt.legend(legend, title=legend_title, loc='lower left')
    #plt.show()




def display_moving_agent(agent_name, load_history=False, n=0):

    durations = []

    infile = open('environments/' + agent_name, 'rb')
    env_dict = pickle.load(infile)
    infile.close()
    env_dict['max_steps']=3000
    #env_dict['architecture'] = 'pete'
    env_dict['reward_signal']['max_progress'] = 0

    env = environment(env_dict)
    env.reset(save_history=True, start_condition=[], car_params=env_dict['car_params'])
    
    infile = open('agents/' + agent_name + '/' + agent_name + '_params', 'rb')
    agent_dict = pickle.load(infile)
    infile.close()
    agent_dict['layer3_size']=300

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
        env.reset(save_history=True, start_condition=[], car_params=env_dict['car_params'])
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
        #wh = env.waypoint_history[i]
        #plt.plot(wh[0], wh[1], 'x')
        #gh = env.goal_history[i]
        #plt.plot([gh[0]-env.s, gh[0]+env.s, gh[0]+env.s, gh[0]-env.s, gh[0]-env.s], [gh[1]-env.s, gh[1]-env.s, gh[1]+env.s, gh[1]+env.s, gh[1]-env.s], 'r')
        
        if env_dict['velocity_control']==True and env_dict['steer_control_dict']['steering_control']==True:
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
    env.reset(save_history=True, start_condition=[], car_params=env_dict['car_params'])

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
        env.reset(save_history=True, start_condition=[], car_params=env_dict['car_params'])
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


def display_path_multiple(agent_names, ns, legend_title, legend, mismatch_parameters, frac_vary):
    
    pose_history = []
    progress_history = []
    state_history = []
    
    for agent_name, n, i in zip(agent_names, ns, range(len(agent_names))):

        infile = open('environments/' + agent_name, 'rb')
        env_dict = pickle.load(infile)
        infile.close()
        # Compensate for changes to reward structure
        env_dict['reward_signal']['max_progress'] = 0
        
        # Model mismatches
        if mismatch_parameters:
            for par, var in zip(mismatch_parameters, frac_vary):
                env_dict['car_params'][par] *= 1+var 


        env = environment(env_dict)
        env.reset(save_history=True, start_condition=[], car_params=env_dict['car_params'])

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

        #start_pose = {'x':11.2, 'y':7.7, 'v':0, 'delta':0, 'theta':0, 'goal':1}
        env.reset(save_history=True, start_condition=start_pose, car_params=env_dict['car_params'])
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

            if env.progress>=0.98:
                done=True
            

        print('Total score = ', score)
        print('Progress = ', env.progress)

        state_history.append(env.state_history)
        pose_history.append(env.pose_history)
        progress_history.append(env.progress_history)
        
        
        # image_path = sys.path[0] + '/maps/' + env.map_name + '.png'
        # im = image.imread(image_path)
        # plt.imshow(im, extent=(0,env.map_width,0,env.map_height))
        # plt.plot(np.array(env.pose_history)[:,0], np.array(env.pose_history)[:,1], linewidth=1.5)
        # plt.legend(legend, title=legend_title)

    #plt.legend(legend, title=legend_title)
    #plt.xlabel('x [m]')
    #plt.ylabel('y [m]')
    # plt.xticks([])
    # plt.yticks([])
    # plt.xlim([0,env.map_width])
    # plt.ylim([0,env.map_height])
    # #plt.grid(True)
    # #plt.title('Agent path')
    # plt.show()

    # plt.plot(np.array(env.progress_history)[:], np.array(env.pose_history)[:,4], linewidth=1.5)
    # plt.show()

    # print('done')
    
    plt.figure(1)
    #fig, axs = plt.subplots(4) 
    image_path = sys.path[0] + '/maps/' + env.map_name + '.png'
    im = image.imread(image_path)
    plt.imshow(im, extent=(0,env.map_width,0,env.map_height))
    for i in range(len(agent_names)):
        plt.plot(np.array(pose_history[i])[:,0], np.array(pose_history[i])[:,1], linewidth=1.5)
    plt.xlabel('x coordinate [m]') 
    plt.ylabel('y coordinate [m]')
    #axs[0].legend(legend, title=legend_title, bbox_to_anchor=[1,1.6])
    #axs[0].legend(legend, title=legend_title)
    #axs[0].show()

    plt.figure(2)
    for i in range(len(agent_names)):
        plt.plot(np.array(progress_history[i])*100, np.array(pose_history[i])[:,4], linewidth=1.5)
        #plt.plot(np.array(pose_history[i])[:,4], linewidth=1.5)
    plt.xlabel('progress along centerline [%]')
    plt.ylabel('Longitudinal velocity [m/s]')
    #axs[1].set(xlabel='progress along centerline [%]', ylabel='Longitudinal velocity [m/s]')
    plt.legend(legend, title=legend_title, loc='lower right')
    #axs[1].set_ylim([0, 15])
    #plt.ylabel('Longitudinal velocity')
    #plt.xlabel('progress along centerline [%]')

    plt.figure(3)
    for i in range(len(agent_names)):
        plt.plot(np.array(progress_history[i])*100, np.array(pose_history[i])[:,3], linewidth=1.5)
        #plt.plot(np.array(pose_history[i])[:,4], linewidth=1.5)
    plt.xlabel('progress along centerline [%]')
    plt.ylabel('steering angle [rad]')
    #axs[1].set(xlabel='progress along centerline [%]', ylabel='Longitudinal velocity [m/s]')
    plt.legend(legend, title=legend_title, loc='lower right')
    #axs[1].set_ylim([0, 15])
    #plt.ylabel('Longitudinal velocity')
    #plt.xlabel('progress along centerline [%]')

    plt.figure(4)
    for i in range(len(agent_names)):
        plt.plot(np.array(progress_history[i])*100, np.array(state_history[i])[:,6], linewidth=1.5)
        #plt.plot(np.array(pose_history[i])[:,4], linewidth=1.5)
    plt.xlabel('progress along centerline [%]')
    plt.ylabel('Slip angle')
    #axs[2].set(xlabel='progress along centerline [%]', ylabel='Slip angle')
    plt.legend(legend, title=legend_title, loc='lower right')
    #axs[1].set_ylim([0, 15])
    #plt.ylabel('Longitudinal velocity')
    #plt.xlabel('progress along centerline [%]')
    
    plt.figure(5)
    for i in range(len(agent_names)):
        plt.plot(np.arange(len(progress_history[i])), np.array(progress_history[i])*100, linewidth=1.5)
        #plt.plot(np.array(pose_history[i])[:,4], linewidth=1.5)
    plt.xlabel('Simulation step')
    plt.ylabel('progress along centerline [%]')
    #axs[3].set(ylabel='progress along centerline [%]', xlabel='Simulation step')
    plt.legend(legend, title=legend_title, loc='lower right')
    #axs[1].set_ylim([0, 15])
    #plt.ylabel('Longitudinal velocity')
    #plt.xlabel('progress along centerline [%]')
    
    plt.show()


def display_maps():
    names = ['Circle', 'Redbull ring', 'Berlin', 'Columbia', 'Porto', 'Torino']
    circle = mapping.map('circle')
    columbia = mapping.map('columbia_1')
    porto = mapping.map('porto_1')
    berlin = mapping.map('berlin')
    torino = mapping.map('torino')
    redbull_ring = mapping.map('redbull_ring')

    fig, axs = plt.subplots(2, 3)
    axs[0,0].imshow(circle.map_array, extent=(0,circle.map_width,0,circle.map_height), cmap="gray")
    axs[0,1].imshow(redbull_ring.map_array, extent=(0,redbull_ring.map_width,0,redbull_ring.map_height), cmap="gray")
    axs[0,2].imshow(berlin.map_array, extent=(0,berlin.map_width,0,berlin.map_height), cmap="gray")
    
    axs[1,0].imshow(columbia.map_array, extent=(0,columbia.map_width,0,columbia.map_height), cmap="gray")
    axs[1,1].imshow(porto.map_array, extent=(0,porto.map_width,0,porto.map_height), cmap="gray")
    axs[1,2].imshow(torino.map_array, extent=(0,torino.map_width,0,torino.map_height), cmap="gray")

    for i in range(2):
        for j in range(3):
                #axs[i,j].set(ylabel='y [m]', xlabel='x [m]')
                axs[i,j].axis('off')
                axs[i,j].set_title(names[j+i*3])
    plt.show()

def display_all_maps_outline():
    names = ['Circle', 'Redbull ring', 'Berlin', 'Columbia', 'Porto', 'Torino']
    circle = mapping.map('circle')
    columbia = mapping.map('columbia_1')
    porto = mapping.map('porto_1')
    berlin = mapping.map('berlin')
    torino = mapping.map('torino')
    redbull_ring = mapping.map('redbull_ring')

    fig, axs = plt.subplots(1,6)
    axs[0].imshow(ImageOps.invert(circle.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))), extent=(0,circle.map_width,0,circle.map_height), cmap="gray")
    axs[5].imshow(ImageOps.invert(redbull_ring.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(1))), extent=(0,redbull_ring.map_width,0,redbull_ring.map_height), cmap="gray")
    axs[3].imshow(ImageOps.invert(berlin.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))), extent=(0,berlin.map_width,0,berlin.map_height), cmap="gray")
    
    axs[1].imshow(ImageOps.invert(columbia.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(1))), extent=(0,columbia.map_width,0,columbia.map_height), cmap="gray")
    axs[2].imshow(ImageOps.invert(porto.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(1))), extent=(0,porto.map_width,0,porto.map_height), cmap="gray")
    axs[4].imshow(ImageOps.invert(torino.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(1))), extent=(0,torino.map_width,0,torino.map_height), cmap="gray")

    # for i in range(2):
    #     for j in range(3):
    #             axs[i,j].set(ylabel='y [m]', xlabel='x [m]')
    #             axs[i,j].axis('off')
    #             axs[i,j].set_title(names[j+i*3])
    # plt.show()

    for i in range(6):
        axs[i].set(ylabel='y [m]', xlabel='x [m]')
        axs[i].axis('off')
        axs[i].set_title(names[i])

    plt.show()

def display_map_outline():
    names = ['Circle', 'Redbull ring', 'Berlin', 'Columbia', 'Porto', 'Torino']
    circle = mapping.map('circle')
    columbia = mapping.map('columbia_1')
    porto = mapping.map('porto_1')
    berlin = mapping.map('berlin')
    torino = mapping.map('torino')
    redbull_ring = mapping.map('redbull_ring')

    
    plt.imshow(ImageOps.invert(circle.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))), extent=(0,circle.map_width,0,circle.map_height), cmap="gray")
    plt.axis('off')
    plt.show()
    plt.imshow(ImageOps.invert(redbull_ring.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(1))), extent=(0,redbull_ring.map_width,0,redbull_ring.map_height), cmap="gray")
    plt.axis('off')
    plt.show()
    plt.imshow(ImageOps.invert(berlin.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))), extent=(0,berlin.map_width,0,berlin.map_height), cmap="gray")
    plt.axis('off')
    plt.show()
    plt.imshow(ImageOps.invert(columbia.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(1))), extent=(0,columbia.map_width,0,columbia.map_height), cmap="gray")
    plt.axis('off')
    plt.show()
    plt.imshow(ImageOps.invert(porto.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(1))), extent=(0,porto.map_width,0,porto.map_height), cmap="gray")
    plt.axis('off')
    plt.show()
    plt.imshow(ImageOps.invert(torino.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(1))), extent=(0,torino.map_width,0,torino.map_height), cmap="gray")
    plt.axis('off')
    plt.show()
    
    # for i in range(2):
    #     for j in range(3):
    #             axs[i,j].set(ylabel='y [m]', xlabel='x [m]')
    #             axs[i,j].axis('off')
    #             axs[i,j].set_title(names[j+i*3])
    # plt.show()

    #for i in range(6):
    #    axs[i].set(ylabel='y [m]', xlabel='x [m]')
    #    axs[i].axis('off')
    #    axs[i].set_title(names[i])
    
    plt.show()


#display_all_maps_outline()
display_map_outline()




#display_maps()



    

