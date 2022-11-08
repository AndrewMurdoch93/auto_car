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
import matplotlib
matplotlib.use('pgf')
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






def learning_curve_lap_time_average(agent_names, legend, legend_title, ns, filname):

    legend_coll = legend.copy()
    legend_coll.append('Min and max')
    window = 500
    steps = [[] for _ in range(len(agent_names))]
    steps_x_axis = [[] for _ in range(len(agent_names))]
    n_actions_x_axis  = [[] for _ in range(len(agent_names))]
    steps_no_coll = [[] for _ in range(len(agent_names))]
    avg_steps_no_coll = [[] for _ in range(len(agent_names))]
    std_steps_no_coll = [[] for _ in range(len(agent_names))]
    upper_fill_steps_no_coll = [[] for _ in range(len(agent_names))]
    lower_fill_steps_no_coll = [[] for _ in range(len(agent_names))]
    
    max_steps_no_coll = [[] for _ in range(len(agent_names))]
    collisions = [[] for _ in range(len(agent_names))]
    avg_time = [[] for _ in range(len(agent_names))]
    avg_coll = [[] for _ in range(len(agent_names))]
    std_coll = [[] for _ in range(len(agent_names))]
    upper_fill_coll = [[] for _ in range(len(agent_names))]
    lower_fill_coll = [[] for _ in range(len(agent_names))]
    
    n_actions = [[] for _ in range(len(agent_names))]

    avg_steps = [[] for _ in range(len(agent_names))]
    avg_n_actions = [[] for _ in range(len(agent_names))]
    avg_collisions = [[] for _ in range(len(agent_names))]

    steps_avg_x_axis = [[] for _ in range(len(agent_names))]
    times = [[] for _ in range(len(agent_names))]

    for i in range(len(agent_names)):
        agent_name = agent_names[i]
        train_results_file_name = 'train_results/' + agent_name
        infile = open(train_results_file_name, 'rb')
        
        _ = pickle.load(infile)
        _ = pickle.load(infile)
        times[i] = pickle.load(infile)
        steps[i] = pickle.load(infile)
        collisions[i] = pickle.load(infile)
        n_actions[i] = pickle.load(infile)
        
        infile.close()
        

        avg_steps[i] = np.average(steps[i],axis=0)
        avg_n_actions[i] = np.average(n_actions[i],axis=0)
        avg_collisions[i] = np.average(collisions[i],axis=0)
        steps_avg_x_axis[i] = np.average(steps[i],axis=0)

        for j in range(len(collisions[i][ns[i]])):
            if j <= window:
                x = 0
            else:
                x = j-window 
            avg_coll[i].append(np.mean(avg_collisions[i][x:j+1]))
            avg_time[i].append(np.mean(steps[i][ns[i]][x:j+1]))
            std_coll[i].append(np.std(avg_collisions[i][x:j+1]))

        upper_fill_coll[i].append(np.array(avg_coll[i])+np.array(std_coll[i]))
        lower_fill_coll[i].append(np.array(avg_coll[i])-np.array(std_coll[i]))

        steps_x_axis[i] = np.cumsum(steps[i][ns[i]])[np.logical_not(collisions[i][ns[i]])]
        n_actions_x_axis[i] = np.cumsum(n_actions[i][ns[i]])[np.logical_not(collisions[i][ns[i]])]
        steps_no_coll[i] = steps[i][ns[i]][np.logical_not(collisions[i][ns[i]])]

        for j in range(len(steps_x_axis[i])):
            if j <= window:
                x = 0
            else:
                x = j-window 
            avg_steps_no_coll[i].append(np.mean(steps_no_coll[i][x:j+1]))
            std_steps_no_coll[i].append(np.std(steps_no_coll[i][x:j+1]))
            #max_steps_no_coll[i].append(np.max(steps_no_coll[i][x:j+1]))
        
        upper_fill_steps_no_coll[i].append(np.array(avg_steps_no_coll[i]) + np.array(std_steps_no_coll[i])) 
        lower_fill_steps_no_coll[i].append(np.array(avg_steps_no_coll[i]) - np.array(std_steps_no_coll[i])) 
    
    # end_episodes = np.zeros(len(agent_names), int)
    # for i in range(len(agent_names)):
    #     end_episodes[i] =  np.where(steps[i][ns[i]]==0)[0][0]

    end_episodes = np.zeros((np.size(np.array(steps),axis=0), np.size(np.array(steps),axis=1)), int)
    for i in range(np.size(end_episodes, axis=0)):
        for n in range(np.size(end_episodes, axis=1)):
            end_episodes[i,n] =  np.where(steps[i][n]==0)[0][0]
            end_ep = end_episodes
    end_episodes = np.min(end_episodes, axis=1)
    
    steps_y = steps.copy()
    steps_y_avg_smoothed = [[] for _ in range(len(agent_names))]
    steps_y_std = [[] for _ in range(len(agent_names))]
    upper_fill = [[] for _ in range(len(agent_names))]
    lower_fill = [[] for _ in range(len(agent_names))]

    for i in range(np.size(end_ep, axis=0)):
        for n in range(np.size(end_ep, axis=1)):
            steps_y[i][n][collisions[i][n]==1]=np.nan
    steps_y_avg = np.array(steps_y)
    steps_y_avg = np.nanmean(steps_y_avg, axis=1)

    for i in range(np.size(end_ep, axis=0)):
        for j in range(len(steps_y_avg[i])):
                if j <= window:
                    x = 0
                else:
                    x = j-window 
                steps_y_avg_smoothed[i].append(np.nanmean(steps_y_avg[i][x:j+1]))
                steps_y_std[i].append(np.nanstd(steps_y_avg[i][x:j+1]))
        
        upper_fill[i].append(np.array(steps_y_avg_smoothed[i])+np.array(steps_y_std[i]))
        lower_fill[i].append(np.array(steps_y_avg_smoothed[i])-np.array(steps_y_std[i]))

    #font = {'family' : 'normal',


    # plt.figure(1, figsize=(5,4))
    # plt.rc('axes',edgecolor='gray')
    # #plt.rc('font', **font)

    # for i in range(len(agent_names)):
    #     end_episode = end_episodes[i] 
    #     plt.plot(np.cumsum(steps_avg_x_axis[i])[0:end_episode],  avg_coll[i][0:end_episode])
    #     plt.fill_between(x=np.cumsum(steps_avg_x_axis[i])[0:end_episode], y1=upper_fill_coll[i][0][0:end_episode], y2=lower_fill_coll[i][0][0:end_episode], alpha=0.3, label='_nolegend_')
    
    # plt.hlines(y=1, xmin=0, xmax=np.cumsum(steps_avg_x_axis[i])[np.max(end_episodes)], colors='black', linestyle='dashed')
    # plt.hlines(y=0, xmin=0, xmax=np.cumsum(steps_avg_x_axis[i])[np.max(end_episodes)], colors='black', linestyle='dashed')
    # plt.xlabel('Simulation steps')
    # #plt.title('Collision rate')
    # plt.ylabel('Collision rate')
    # plt.legend(legend_coll, title=legend_title, loc='upper right')
    # #plt.xlim([0,6000])
    # plt.ylim([-0.05, 1.05])
    # plt.grid(True)
    # plt.rc('axes',edgecolor='gray')
    # plt.tick_params(axis=u'both', which=u'both',length=0)

    # plt.savefig('collision_rate.pgf', format='pgf')

    # plt.figure(2, figsize=(5,4))

    # for i in range(len(agent_names)):
    #     end_episode = end_episodes[i] 
    #     #plt.plot(np.cumsum(steps[0]), steps[0], 'x')
    #     #plt.plot(steps[0][np.where(np.logical_not(collisions[0]))], 'x')
    #     plt.plot(np.cumsum(steps_avg_x_axis[i])[0:end_episode],   np.array(steps_y_avg_smoothed[i][0:end_episode])*0.01)
    #     plt.fill_between(x=np.cumsum(steps_avg_x_axis[i])[0:end_episode], y1=upper_fill[i][0][0:end_episode]*0.01, y2=lower_fill[i][0][0:end_episode]*0.01, alpha=0.3, label='_nolegend_')
    
    # plt.xlabel('Simulation steps')
    # #plt.title('Lap time')
    # plt.ylabel('Lap time [s]')
    # plt.legend(legend, title=legend_title, loc='upper right')
    # #plt.xlim([0,6000])
    # plt.grid(True)
    # plt.rc('axes',edgecolor='gray')
    # plt.tick_params(axis=u'both', which=u'both',length=0)

    plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False,     # don't setup fonts from rc parameters
    "font.size": 12
    })

    plt.rc('axes',edgecolor='gray')
    fig, ax = plt.subplots(1, 2, figsize=(5.5,3.5))

    #plt.figure(3, figsize=(5,4))
    for i in range(len(agent_names)):
        end_episode = end_episodes[i] 
        #ax[0].plot(avg_coll[i][0:end_episode])
        ax[0].plot(np.arange(0,end_episode,100), np.array(avg_coll[i][0:end_episode])[np.arange(0,end_episode,100)])
        ax[0].fill_between(x=np.arange(end_episode)[np.arange(0,end_episode,100)], y1=np.array(upper_fill_coll[i][0])[np.arange(0,end_episode,100)], y2=np.array(lower_fill_coll[i][0])[np.arange(0,end_episode,100)], alpha=0.15, label='_nolegend_')
    
    ax[0].hlines(y=1, xmin=0, xmax=np.max(end_episodes), colors='black', linestyle='dashed')
    ax[0].hlines(y=0, xmin=0, xmax=np.max(end_episodes), colors='black', linestyle='dashed')
    ax[0].set_ylim([-0.05, 1.05])
    #ax[0].set_xlim([0, np.max(end_episodes)])
    ax[0].set_xlabel('Episodes')
    #plt.title('Collision rate')
    ax[0].set_ylabel('Collision rate')
    ax[0].tick_params('both', length=0)
    ax[0].grid(True)
    #ax[0].set_xlim([0,4000])
    #plt.rc('axes',edgecolor='gray')
    #plt.tick_params(axis=u'both', which=u'both',length=0)



    for i in range(np.size(end_ep, axis=0)):
        end_episode = end_episodes[i] 
        #plt.plot(np.cumsum(steps[0]), steps[0], 'x')
        #plt.plot(steps[0][np.where(np.logical_not(collisions[0]))], 'x')
        ax[1].plot(np.arange(0,end_episode,100), (np.array(steps_y_avg_smoothed[i])*0.01)[np.arange(0,end_episode,100)])
        ax[1].fill_between(x=np.arange(0,end_episode,100), y1=np.array(upper_fill[i][0])[np.arange(0,end_episode,100)]*0.01, y2=np.array(lower_fill[i][0])[np.arange(0,end_episode,100)]*0.01, alpha=0.15, label='_nolegend_')

        #np.arange(len(steps[i][ns[i]]))[np.logical_not(collisions[i][ns[i]])][0:end_episodes[i]]
        #plt.plot(np.array(max_steps_no_coll[i][0:end_episode_no_coll])*0.01 )
    ax[1].set_xlabel('Episodes')
    #ax[1].set_xlim([0, np.max(end_episodes)])
    #plt.title('Average time per episode without collisions')
    ax[1].set_ylabel('Lap time [s]')
    #ax[1].legend(legend, title=legend_title, loc='upper right')
    ax[1].grid(True)
    ax[1].tick_params('both', length=0)
    #ax[1].set_xlim([0,4000])
    #plt.tick_params(axis=u'both', which=u'both',length=0)
    #plt.xlim([0,6000])


    # plt.figure(5, figsize=(5,4))
    # for i in range(len(agent_names)):
    #     end_episode = end_episodes[i] 
    #     plt.plot(np.cumsum(avg_n_actions[i])[0:end_episode],  avg_coll[i][0:end_episode])
    #     plt.fill_between(x=np.cumsum(avg_n_actions[i])[0:end_episode], y1=upper_fill_coll[i][0][0:end_episode], y2=lower_fill_coll[i][0][0:end_episode], alpha=0.3, label='_nolegend_')
    
    # plt.hlines(y=1, xmin=0, xmax=np.cumsum(avg_n_actions[i])[np.max(end_episodes)], colors='black', linestyle='dashed')
    # plt.hlines(y=0, xmin=0, xmax=np.cumsum(avg_n_actions[i])[np.max(end_episodes)], colors='black', linestyle='dashed')
    # plt.xlabel('Steps')
    # #plt.title('Collision rate')
    # plt.ylabel('Collision rate')
    # plt.legend(legend_coll, title=legend_title, loc='upper right')
    # #plt.xlim([0,6000])
    # plt.ylim([-0.05, 1.05])
    # plt.grid(True)
    # plt.rc('axes',edgecolor='gray')
    # plt.tick_params(axis=u'both', which=u'both',length=0)

    # plt.figure(6, figsize=(5,4))
    # for i in range(len(agent_names)):
    #     end_episode = end_episodes[i]
    #     plt.plot(np.cumsum(avg_n_actions[i][0:end_episode]), np.array(steps_y_avg_smoothed[i][0:end_episode])*0.01)
    #     plt.fill_between(x=np.cumsum(avg_n_actions[i][0:end_episode]), y1=upper_fill[i][0][0:end_episode]*0.01, y2=lower_fill[i][0][0:end_episode]*0.01, alpha=0.3, label='_nolegend_')
    
    
    # plt.xlabel('Steps')
    # #plt.title('Lap time')
    # plt.ylabel('Lap time [s]')
    # plt.legend(legend, title=legend_title, loc='upper right')
    # #plt.xlim([0,6000])
    # plt.grid(True)
    # plt.rc('axes',edgecolor='gray')
    # plt.tick_params(axis=u'both', which=u'both',length=0)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.4) 
    plt.figlegend(legend, title=legend_title, loc = 'lower center', ncol=3)
    
    plt.savefig('results/'+filename+'.pgf', format='pgf')
    #plt.show()




def learning_curve_reward_average(agent_names, legend, legend_title):
    
    legend_new = legend.copy()
    legend_new.append('Min and max')
    window = 500
    
    steps = [[] for _ in range(len(agent_names))]
    avg_steps = [[] for _ in range(len(agent_names))]
    steps_x_axis = [[] for _ in range(len(agent_names))]    
    scores = [[] for _ in range(len(agent_names))]    
    avg_score = [[] for _ in range(len(agent_names))]
    std_score = [[] for _ in range(len(agent_names))]
    upper_fill = [[] for _ in range(len(agent_names))]
    lower_fill = [[] for _ in range(len(agent_names))]
    n_actions = [[] for _ in range(len(agent_names))]
    avg_n_actions = [[] for _ in range(len(agent_names))]


    for i in range(len(agent_names)):
        agent_name = agent_names[i]
        train_results_file_name = 'train_results/' + agent_name
        infile = open(train_results_file_name, 'rb')
        
        scores[i] = pickle.load(infile)
        _ = pickle.load(infile)
        _ = pickle.load(infile)
        steps[i] = pickle.load(infile)
        _ = pickle.load(infile)
        n_actions[i] = pickle.load(infile)
        
        infile.close()
        
        scores[i] = np.average(scores[i],axis=0)
        avg_steps[i] = np.average(steps[i],axis=0)
        avg_n_actions[i] = np.average(n_actions[i],axis=0)

        for j in range(len(scores[i])):
            if j <= window:
                x = 0
            else:
                x = j-window 
            avg_score[i].append(np.mean(scores[i][x:j+1]))
            std_score[i].append(np.std(scores[i][x:j+1]))

        upper_fill[i].append(np.array(avg_score[i])+np.array(std_score[i]))
        lower_fill[i].append(np.array(avg_score[i])-np.array(std_score[i]))
    
    end_episodes = np.zeros((np.size(np.array(steps),axis=0), np.size(np.array(steps),axis=1)), int)
    for i in range(np.size(end_episodes, axis=0)):
        for n in range(np.size(end_episodes, axis=1)):
            end_episodes[i,n] =  np.where(steps[i][n]==0)[0][0]
    end_episodes = np.min(end_episodes, axis=1)
    

    
    plt.figure(1, figsize=(5,4))
    plt.rc('axes',edgecolor='gray')
    for i in range(len(agent_names)):
        end_episode = end_episodes[i] 
        plt.plot(np.cumsum(avg_steps[i])[0:end_episode],  avg_score[i][0:end_episode])
        plt.fill_between(x=np.cumsum(avg_steps[i])[0:end_episode], y1=upper_fill[i][0][0:end_episode], y2=lower_fill[i][0][0:end_episode], alpha=0.3, label='_nolegend_')
    
    plt.xlabel('Simulation steps')
    #plt.title('Collision rate')
    plt.ylabel('Episode reward')
    plt.legend(legend_new, title=legend_title, loc='lower right')
    #plt.xlim([0,6000])
    plt.grid(True)
    plt.tick_params(axis=u'both', which=u'both',length=0)



    plt.figure(2, figsize=(5,4))
    plt.rc('axes',edgecolor='gray')
    for i in range(len(agent_names)):
        end_episode = end_episodes[i] 
        plt.plot(np.cumsum(avg_n_actions[i])[0:end_episode],  avg_score[i][0:end_episode])
        plt.fill_between(x=np.cumsum(avg_n_actions[i])[0:end_episode], y1=upper_fill[i][0][0:end_episode], y2=lower_fill[i][0][0:end_episode], alpha=0.3, label='_nolegend_')
    
    plt.xlabel('Steps')
    plt.ylabel('Episode reward')
    plt.legend(legend_new, title=legend_title, loc='lower right')
    plt.xlim([0,6000])
    plt.grid(True)
    plt.tick_params(axis=u'both', which=u'both',length=0)



    plt.figure(3, figsize=(5,4))
    plt.rc('axes',edgecolor='gray')
    for i in range(len(agent_names)):
        end_episode = end_episodes[i] 
        plt.plot(np.arange(end_episode),  avg_score[i][0:end_episode])
        plt.fill_between(x=np.arange(end_episode), y1=upper_fill[i][0][0:end_episode], y2=lower_fill[i][0][0:end_episode], alpha=0.3, label='_nolegend_')
    
    plt.xlabel('Episode')
    plt.ylabel('Episode reward')
    plt.legend(legend_new, title=legend_title, loc='lower right')
    plt.xlim([0,5000])
    plt.grid(True)
    plt.tick_params(axis=u'both', which=u'both',length=0)

    #plt.show()
    plt.savefig('reward_dist.pgf', format='pgf')





# agent_names = ['porto_ete_r_1', 'porto_ete_r_2', 'porto_ete_r_3', 'porto_ete_LiDAR_10', 'porto_ete_r_4' ]
# legend = ['$r_{\mathrm{dist}} = 1$', '$r_{\mathrm{dist}} = 0.7$', '$r_{\mathrm{dist}} = 0.5$', '$r_{\mathrm{dist}} = 0.3$', '$r_{\mathrm{dist}} = 0.1$']
# legend_title = 'Distance reward'
# ns=[0, 0, 0, 0, 0]
# filename='reward_dist'


agent_names = ['porto_ete_r_7', 'porto_ete_r_5', 'porto_ete_LiDAR_10', 'porto_ete_r_6']
legend = ['$r_{\mathrm{collision}} = 0$', '$r_{\mathrm{collision}} = -0.5$', '$r_{\mathrm{collision}} = -1$', '$r_{\mathrm{collision}} = -2$']
legend_title = 'Collision penalty'
ns=[0, 0, 0, 0]
filename = 'reward_collision'

# agent_names = ['porto_ete_r_0', 'porto_ete_r_1']
# legend = ['$r_{\mathrm{t}} = 0', '$r_{\mathrm{t}} = -0.01$']
# legend_title = 'Time step penalty'
# ns=[0, 0]
# filename = 'reward_time'

# agent_names = ['porto_ete_LiDAR_3', 'porto_ete_LiDAR_10', 'porto_ete_LiDAR_20', 'porto_ete_LiDAR_50']
# legend = ['3', '10', '20', '50']
# legend_title = 'Number of LiDAR beams'
# ns=[0, 0, 0, 0]
# filename = observation_n_beams


# agent_names = ['porto_ete_only_LiDAR', 'porto_ete_LiDAR_20', 'porto_ete_no_LiDAR']
# legend = ['Only LiDAR', 'LiDAR and pose', 'Only pose']
# legend_title = 'Observation space'
# ns=[0, 0, 0]
# filename = 'observation_space'


# agent_names = ['porto_ete_cs_1', 'porto_ete_cs_5', 'porto_ete_cs_10', 'porto_ete_cs_15', 'porto_ete_LiDAR_10', 'porto_ete_cs_25']
# legend = ['1', '5', '10', '15', '20', '25']
# legend_title = 'Simulation steps per action'
# ns=[0, 0, 0, 0, 0, 0]
# filename = 'control_steps'


# agent_names = ['porto_ete_v_5', 'porto_ete_LiDAR_10', 'porto_ete_v_7', 'porto_ete_v_8']
# legend = ['5', '6', '7', '8']
# legend_title = 'Maximum velocity [m/s]'
# ns=[0, 0, 0, 0]
# filename = 'maximum_velocity'



learning_curve_lap_time_average(agent_names, legend, legend_title, ns, filename)