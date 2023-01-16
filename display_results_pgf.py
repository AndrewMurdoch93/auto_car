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






def learning_curve_lap_time_average(agent_names, legend, legend_title, ns, filename, xlim, xspace):

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
    
    avg_success = [[] for _ in range(len(agent_names))]
    std_success = [[] for _ in range(len(agent_names))]
    upper_fill_success = [[] for _ in range(len(agent_names))]
    lower_fill_success = [[] for _ in range(len(agent_names))]

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
        #avg_success[i] = np.average(np.logical_not(collisions[i]),axis=0)*100

        for j in range(len(collisions[i][ns[i]])):
            if j <= window:
                x = 0
            else:
                x = j-window 
            avg_coll[i].append(np.mean(avg_collisions[i][x:j+1]))
            avg_time[i].append(np.mean(steps[i][ns[i]][x:j+1]))
            std_coll[i].append(np.std(avg_collisions[i][x:j+1]))
            #std_success[i].append(np.std(avg_collisions[i][x:j+1]))

        upper_fill_coll[i].append(np.array(avg_coll[i])+np.array(std_coll[i]))
        lower_fill_coll[i].append(np.array(avg_coll[i])-np.array(std_coll[i]))

        #upper_fill_success[i].append(np.array(avg_success[i])+np.array(std_success[i]))
        #lower_fill_success[i].append(np.array(avg_success[i])-np.array(std_success[i]))

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
    fig, ax = plt.subplots(1, 2, figsize=(5.5,2.8))


    #ax[0].ticklabel_format(style='scientific', axis='x', scilimits=(0,0), useMathText=True, useOffset=True)
    ax[1].ticklabel_format(style='scientific', axis='x', scilimits=(0,0), useMathText=True)
    #ax[2].ticklabel_format(style='scientific', axis='x', scilimits=(0,0), useMathText=True, useOffset=True)
    #plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
 
    xax1000 = np.arange(0,xlim+1000,xspace)
    xax1 = (xax1000/1000).astype(int)

    ax[0].set_xticks(ticks=xax1000, labels=xax1)
    ax[1].set_xticks(ticks=xax1000, labels=xax1)

    #ax[0].set_yticks(ticks=[0,25,50,75,100], labels=[0,25,50,75,100])

    #plt.figure(3, figsize=(5,4))
    for i in range(len(agent_names)):
        end_episode = end_episodes[i] 
        #ax[0].plot(avg_coll[i][0:end_episode])
        ax[0].plot(np.arange(0,end_episode,100), np.array(avg_coll[i][0:end_episode])[np.arange(0,end_episode,100)]*100)
        ax[0].fill_between(x=np.arange(end_episode)[np.arange(0,end_episode,100)], y1=np.array(upper_fill_coll[i][0])[np.arange(0,end_episode,100)]*100, y2=np.array(lower_fill_coll[i][0])[np.arange(0,end_episode,100)]*100, alpha=0.15, label='_nolegend_')
    
    ax[0].hlines(y=100, xmin=0, xmax=np.max(end_episodes), colors='black', linestyle='dashed')
    ax[0].hlines(y=0, xmin=0, xmax=np.max(end_episodes), colors='black', linestyle='dashed')
    ax[0].set_ylim([-5, 105])
    #ax[0].set_xlim([0, np.max(end_episodes)])
    ax[0].set_xlabel(r'Episodes $\times 10^3$')
    #plt.title('Collision rate')
    ax[0].set_ylabel('Failure rate [%]')
    ax[0].tick_params('both', length=0)
    ax[0].grid(True)
    ax[0].set_xlim([0,xlim])
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
    ax[1].set_xlabel(r'Episodes $\times 10^3$')
    #ax[1].set_xlim([0, np.max(end_episodes)])
    #plt.title('Average time per episode without collisions')
    ax[1].set_ylabel('Lap time [s]')
    #ax[1].legend(legend, title=legend_title, loc='upper right')
    ax[1].grid(True)
    ax[1].tick_params('both', length=0)
    ax[1].set_xlim([0,xlim])
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
    fig.subplots_adjust(bottom=0.33) 
    plt.figlegend(legend, title=legend_title, loc = 'lower center', ncol=5)
    
    plt.savefig('results/'+filename+'.pgf', format='pgf')

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
    
    # plt.rcParams.update({
    # "font.family": "serif",  # use serif/main font for text elements
    # "text.usetex": True,     # use inline math for ticks
    # "pgf.rcfonts": False,     # don't setup fonts from rc parameters
    # "font.size": 12
    # })

    fig, ax = plt.subplots(1, figsize=(5,2.5))
    plt.rc('axes',edgecolor='gray')
    
    
    color='grey'
    ax.spines['bottom'].set_color(color)
    ax.spines['top'].set_color(color) 
    ax.spines['right'].set_color(color)
    ax.spines['left'].set_color(color)

    # [np.arange(0,end_episode,100)]
    for i in range(len(agent_names)):
        end_episode = end_episodes[i] 
        ax.plot(np.arange(0,end_episode,100),  np.array(avg_score[i])[np.arange(0,end_episode,100)])
        ax.fill_between(x=np.arange(0,end_episode,100), y1=upper_fill[i][0][np.arange(0,end_episode,100)], y2=lower_fill[i][0][np.arange(0,end_episode,100)], alpha=0.15, label='_nolegend_')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode\nreward')
    ax.legend(legend, title=legend_title, loc='lower right')
    #plt.xlim([0,5000])
    ax.grid(True)
    ax.tick_params(axis=u'both', which=u'both',length=0)
    
    fig.tight_layout()
    #fig.subplots_adjust(right=0.75) 
    #plt.figlegend(legend, title=legend_title, loc='center right', ncol=1)


    # plt.show()
    plt.savefig('results/'+filename+'.pgf', format='pgf')

def learning_curve_all(agent_names, legend, legend_title, ns, filename, xlim, xspace):

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
    
    avg_success = [[] for _ in range(len(agent_names))]
    std_success = [[] for _ in range(len(agent_names))]
    upper_fill_success = [[] for _ in range(len(agent_names))]
    lower_fill_success = [[] for _ in range(len(agent_names))]

    scores = [[] for _ in range(len(agent_names))]
    avg_scores = [[] for _ in range(len(agent_names))]
    avg_score = [[] for _ in range(len(agent_names))]
    std_score = [[] for _ in range(len(agent_names))]
    score_upper_fill = [[] for _ in range(len(agent_names))]
    score_lower_fill = [[] for _ in range(len(agent_names))]


    for i in range(len(agent_names)):
        agent_name = agent_names[i]
        train_results_file_name = 'train_results/' + agent_name
        infile = open(train_results_file_name, 'rb')
        
        scores[i] = pickle.load(infile)
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
        avg_scores[i] = np.average(scores[i],axis=0)
        #avg_success[i] = np.average(np.logical_not(collisions[i]),axis=0)*100

        for j in range(len(collisions[i][ns[i]])):
            if j <= window:
                x = 0
            else:
                x = j-window 
            avg_coll[i].append(np.mean(avg_collisions[i][x:j+1]))
            avg_time[i].append(np.mean(steps[i][ns[i]][x:j+1]))
            std_coll[i].append(np.std(avg_collisions[i][x:j+1]))
            avg_score[i].append(np.mean(avg_scores[i][x:j+1]))
            std_score[i].append(np.std(avg_scores[i][x:j+1]))
            #std_success[i].append(np.std(avg_collisions[i][x:j+1]))

        upper_fill_coll[i].append(np.array(avg_coll[i])+np.array(std_coll[i]))
        lower_fill_coll[i].append(np.array(avg_coll[i])-np.array(std_coll[i]))

        score_upper_fill[i].append(np.array(avg_score[i])+np.array(std_score[i]))
        score_lower_fill[i].append(np.array(avg_score[i])-np.array(std_score[i]))
        #upper_fill_success[i].append(np.array(avg_success[i])+np.array(std_success[i]))
        #lower_fill_success[i].append(np.array(avg_success[i])-np.array(std_success[i]))

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



    plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False,     # don't setup fonts from rc parameters
    "font.size": 12
    })

    plt.rc('axes',edgecolor='gray')
    #fig, ax = plt.subplots(1, 3, figsize=(5.5,3))
    fig, ax = plt.subplots(1, 3, figsize=(5.5,2.4))
    
    #ax[0].ticklabel_format(style='scientific', axis='x', scilimits=(0,0), useMathText=True, useOffset=True)
    ax[1].ticklabel_format(style='scientific', axis='x', scilimits=(0,0), useMathText=True)
    #ax[2].ticklabel_format(style='scientific', axis='x', scilimits=(0,0), useMathText=True, useOffset=True)
    #plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    xax1000 = np.arange(0,xlim+1000,xspace)
    xax1 = (xax1000/1000).astype(int)

    ax[0].set_xticks(ticks=xax1000, labels=xax1)
    ax[1].set_xticks(ticks=xax1000, labels=xax1)
    ax[2].set_xticks(xax1000, labels=xax1)
    
    ax[0].set_yticks(ticks=[0,25,50,75,100], labels=[0,25,50,75,100])
    
    ax[1].set_yticks(ticks=[5,6,7,8,9,10], labels=[5,6,7,8,9,10])
    ax[2].set_yticks(ticks=[-10, -5, 0, 5], labels=[-10, -5, 0, 5])

    #ax[1].ticklabel_format(style='scientific', axis='x', scilimits=(0,0), useMathText=True)

    #plt.figure(3, figsize=(5,4))
    for i in range(len(agent_names)):
        end_episode = end_episodes[i] 
        #ax[0].plot(avg_coll[i][0:end_episode])
        ax[0].plot(np.arange(0,end_episode,100), np.array(avg_coll[i][0:end_episode])[np.arange(0,end_episode,100)]*100)
        ax[0].fill_between(x=np.arange(end_episode)[np.arange(0,end_episode,100)], y1=np.array(upper_fill_coll[i][0])[np.arange(0,end_episode,100)]*100, y2=np.array(lower_fill_coll[i][0])[np.arange(0,end_episode,100)]*100, alpha=0.15, label='_nolegend_')
    
    ax[0].hlines(y=100, xmin=0, xmax=np.max(end_episodes), colors='black', linestyle='dashed')
    ax[0].hlines(y=0, xmin=0, xmax=np.max(end_episodes), colors='black', linestyle='dashed')
    ax[0].set_ylim([-5, 105])
    #ax[0].set_xlim([0, np.max(end_episodes)])
    #ax[0].set_xlabel('Episodes')
    #plt.title('Collision rate')
    #ax[0].set_ylabel('Failure rate [%]')
    ax[0].set_title('Failure rate [%]', fontdict={'fontsize': 12})
    ax[0].tick_params('both', length=0)
    ax[0].grid(True)
    ax[0].set_xlim([0,xlim])
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
    ax[1].set_xlabel(r'Episodes $\times 10^3$')
    #ax[1].set_xlim([0, np.max(end_episodes)])
    #plt.title('Average time per episode without collisions')
    #ax[1].set_ylabel('Lap time [s]')
    ax[1].set_title('Lap time [s]', fontdict={'fontsize': 12})
    #ax[1].legend(legend, title=legend_title, loc='upper right')
    ax[1].grid(True)
    ax[1].tick_params('both', length=0)
    #ax[1].set_xlim([0,4000])
    #plt.tick_params(axis=u'both', which=u'both',length=0)
    ax[1].set_xlim([0,xlim])

    for i in range(np.size(end_ep, axis=0)):
        end_episode = end_episodes[i] 
        #plt.plot(np.cumsum(steps[0]), steps[0], 'x')
        #plt.plot(steps[0][np.where(np.logical_not(collisions[0]))], 'x')
        ax[2].plot(np.arange(0,end_episode,100), np.array(avg_score[i])[np.arange(0,end_episode,100)])
        ax[2].fill_between(x=np.arange(0,end_episode,100), y1=np.array(score_upper_fill[i][0])[np.arange(0,end_episode,100)], y2=np.array(score_lower_fill[i][0])[np.arange(0,end_episode,100)], alpha=0.15, label='_nolegend_')

        #np.arange(len(steps[i][ns[i]]))[np.logical_not(collisions[i][ns[i]])][0:end_episodes[i]]
        #plt.plot(np.array(max_steps_no_coll[i][0:end_episode_no_coll])*0.01 )
    #ax[2].set_xlabel('Episodes')
    #ax[1].set_xlim([0, np.max(end_episodes)])
    #plt.title('Average time per episode without collisions')
    #ax[2].set_ylabel('Lap time [s]')
    ax[2].set_title('Reward', fontdict={'fontsize': 12})
    #ax[1].legend(legend, title=legend_title, loc='upper right')
    ax[2].grid(True)
    ax[2].tick_params('both', length=0)
    ax[2].set_xlim([0,xlim])
    #plt.tick_params(axis=u'both', which=u'both',length=0)
    #plt.xlim([0,6000])


    fig.tight_layout()
    fig.subplots_adjust(bottom=0.34) 
    plt.figlegend(legend, title=legend_title, loc = 'lower center', ncol=5)
    # plt.show()
    plt.savefig('results/'+filename+'.pgf', format='pgf')

def display_velocity_slip(agent_names, ns, legend_title, legend, mismatch_parameters, frac_vary, start_condition, filename):
    
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
        if start_condition:
            env.reset(save_history=True, start_condition=start_condition, car_params=env_dict['car_params'])
        else:
            env.reset(save_history=True, start_condition=[], car_params=env_dict['car_params'])

        infile = open('agents/' + agent_name + '/' + agent_name + '_params', 'rb')
        agent_dict = pickle.load(infile)
        infile.close()

        infile = open('train_parameters/' + agent_name, 'rb')
        main_dict = pickle.load(infile)
        infile.close()
          
        if i==0 and not start_condition:
            infile = open('test_initial_condition/' + env_dict['map_name'], 'rb')
            start_conditions = pickle.load(infile)
            infile.close()
            start_condition = random.choice(start_conditions)

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
        env.reset(save_history=True, start_condition=start_condition, car_params=env_dict['car_params'])
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
        
        
        
    

    xlims = [0,100]

    legend_new = legend.copy()
    legend_new.insert(0, 'Min and max')

    legend_racetrack = legend.copy()
    legend_racetrack.insert(0, 'Track centerline')

    plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False,     # don't setup fonts from rc parameters
    "font.size": 12
    })

    fig, ax = plt.subplots(2, figsize=(5,2.5))
    plt.rc('axes', edgecolor='lightgray')

    

    #plt.figure(1, figsize=figure_size)
    #ax = plt.subplot(111)

    #plt.rc('axes',edgecolor='lightgrey')
    color='gray'
    for i in range(2):
        #ax[i].tick_params(axis='both', colors='lightgrey')
        ax[i].spines['bottom'].set_color(color)
        ax[i].spines['top'].set_color(color) 
        ax[i].spines['right'].set_color(color)
        ax[i].spines['left'].set_color(color)

    # ax.tick_params(axis=u'both', which=u'both',length=0)
    
    # track = mapping.map(env.map_name)
    # ax.imshow(ImageOps.invert(track.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(1))), extent=(0,track.map_width,0,track.map_height), cmap="gray")
    # ax.plot(env.rx, env.ry, color='gray', linestyle='dashed')
    
    # for i in range(len(agent_names)):
    #     ax.plot(np.array(pose_history[i])[:,0], np.array(pose_history[i])[:,1], linewidth=1.5)   
  
    # prog = np.array([0, 0.2, 0.4, 0.6, 0.8])
    # idx =  np.zeros(len(prog), int)
    # text = ['Start', '20%', '40%', '60%', '80%']

    # for i in range(len(idx)):
    #     idx[i] = np.mod(env.start_point+np.round(prog[i]*len(env.rx)), len(env.rx))
    # idx.astype(int)
    
    # for i in range(len(idx)):
    #     plt.text(x=env.rx[idx[i]], y=env.ry[idx[i]], s=text[i], fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))


    # ax.set_xlabel('x coordinate [m]',**myfont) 
    # ax.set_ylabel('y coordinate [m]',**myfont)
    # #ax.set_tick_params(axis=u'both', which=u'both',length=0)
    
    # # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # # Put a legend to the right of the current axis
    # ax.legend(legend_racetrack, loc='center left',  bbox_to_anchor=(1, 0.5))
    
    # #plt.legend(legend_new, title=legend_title, loc='lower right')





    

    #ax.hlines(y=env_dict['action_space_dict']['vel_select'][0], xmin=0, xmax=100, colors='black', linestyle='dashed')
    #ax.hlines(y=env_dict['action_space_dict']['vel_select'][1], xmin=0, xmax=100, colors='black', linestyle='dashed', label='_nolegend_')
    for i in range(len(agent_names)):
        ax[0].plot(np.array(progress_history[i])*100, np.array(pose_history[i])[:,4], linewidth=1.5)

    #ax[0].set_xlabel('progress along centerline [%]')
    ax[0].set_ylabel('Longitudinal \nvelocity [m/s]')
    
    #box = ax[0].get_position()
    #ax[0].set_position([box.x0, box.y0, box.width * 0.7, box.height])

    #ax.legend(legend, title=legend_title, bbox_to_anchor=(1.04, 0.5), loc="center left")
    #plt.legend(legend_new, title=legend_title, loc='lower right')
    ax[0].set_xlim(xlims)
    ax[0].set_ylim([env_dict['action_space_dict']['vel_select'][0]-0.2, env_dict['action_space_dict']['vel_select'][1]+0.2])
    ax[0].grid(True, color='lightgrey')

    ax[0].tick_params('both', length=0)
    
    #plt.show()

    # plt.figure(3, figsize=figure_size)
    # plt.rc('axes',edgecolor='lightgrey')

    # plt.hlines(y=env_dict['car_params']['s_min'], xmin=0, xmax=100, colors='black', linestyle='dashed')
    # plt.hlines(y=env_dict['car_params']['s_max'], xmin=0, xmax=100, colors='black', linestyle='dashed', label='_nolegend_')
    # for i in range(len(agent_names)):
    #     plt.plot(np.array(progress_history[i])*100, np.array(pose_history[i])[:,3], linewidth=1.5)

    # plt.xlabel('progress along centerline [%]',**myfont)
    # plt.ylabel('steering angle [rads]',**myfont)
    # plt.legend(legend_new, title=legend_title, loc='lower right')
    # plt.xlim(xlims)
    # plt.ylim([env_dict['car_params']['s_min']-0.05, env_dict['car_params']['s_max']+0.05])
    # plt.grid(True, color='lightgrey')
    # plt.tick_params(axis=u'both', which=u'both',length=0)



    #plt.figure(4, figsize=figure_size)
    #plt.rc('axes',edgecolor='lightgrey')
    for i in range(len(agent_names)):
        ax[1].plot(np.array(progress_history[i])*100, np.array(state_history[i])[:,6], linewidth=1.5)
      
    #box = ax[1].get_position()
    #ax[1].set_position([box.x0, box.y0, box.width * 0.7, box.height])

    ax[1].set_xlabel('progress along centerline [%]')
    ax[1].set_ylabel('Slip angle [rads]')
    #plt.legend(legend, title=legend_title, loc='lower right')
    ax[1].set_xlim(xlims)
    ax[1].set_ylim([-1,1])
    ax[1].grid(True)
    ax[1].tick_params(axis=u'both', which=u'both',length=0)
    ax[1].set_yticks(np.arange(-1, 1.1, 0.5))
    
    #fig.tight_layout()
    fig.subplots_adjust(right=0.75) 
    plt.figlegend(legend, title=legend_title, loc='center right', ncol=1)
    # plt.show() 
    plt.savefig('results/'+filename+'.pgf', format='pgf')

def display_path_multiple(agent_names, ns, legend_title, legend, mismatch_parameters, frac_vary, start_condition, filename):
    
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
        if start_condition:
            env.reset(save_history=True, start_condition=start_condition, car_params=env_dict['car_params'])
        else:
            env.reset(save_history=True, start_condition=[], car_params=env_dict['car_params'])

        infile = open('agents/' + agent_name + '/' + agent_name + '_params', 'rb')
        agent_dict = pickle.load(infile)
        infile.close()

        infile = open('train_parameters/' + agent_name, 'rb')
        main_dict = pickle.load(infile)
        infile.close()
          
        if i==0 and not start_condition:
            infile = open('test_initial_condition/' + env_dict['map_name'], 'rb')
            start_conditions = pickle.load(infile)
            infile.close()
            start_condition = random.choice(start_conditions)

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
        env.reset(save_history=True, start_condition=start_condition, car_params=env_dict['car_params'])
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
        
        
        
    

    xlims = [0,100]

    legend_new = legend.copy()
    legend_new.insert(0, 'Min and max')

    legend_racetrack = legend.copy()
    legend_racetrack.insert(0, 'Track centerline')

    plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False,     # don't setup fonts from rc parameters
    "font.size": 12
    })

    fig, ax = plt.subplots(4, figsize=(5,7))

    color='gray'
    for i in range(4):
        #ax[i].tick_params(axis='both', colors='lightgrey')
        ax[i].spines['bottom'].set_color(color)
        ax[i].spines['top'].set_color(color) 
        ax[i].spines['right'].set_color(color)
        ax[i].spines['left'].set_color(color)

    ax[0].tick_params(axis=u'both', which=u'both',length=0)
    
    track = mapping.map(env.map_name)
    ax[0].imshow(ImageOps.invert(track.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(1))), extent=(0,track.map_width,0,track.map_height), cmap="gray")
    ax[0].plot(env.rx, env.ry, color='gray', linestyle='dashed')
    
    for i in range(len(agent_names)):
        ax[0].plot(np.array(pose_history[i])[:,0], np.array(pose_history[i])[:,1], linewidth=1.5)   

    prog = np.array([0, 0.2, 0.4, 0.6, 0.8])
    idx =  np.zeros(len(prog), int)
    text = ['', '20%', '40%', '60%', '80%']

    for i in range(len(idx)):
        idx[i] = np.mod(env.start_point+np.round(prog[i]*len(env.rx)), len(env.rx))
    idx.astype(int)
    
    for i in range(len(idx)):
        ax[0].text(x=env.rx[idx[i]], y=env.ry[idx[i]], s=text[i], fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))
    
    ax[0].vlines(x=env.rx[idx[0]], ymin=env.ry[idx[0]]-1, ymax=env.ry[idx[0]]+1, linestyles='dotted', color='red')
    ax[0].text(x=env.rx[idx[0]]-1.2, y=env.ry[idx[0]]+1.3, s='Start/finish', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))

    ax[0].axis('off')
    # # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot



    ax[1].hlines(y=env_dict['action_space_dict']['vel_select'][0], xmin=0, xmax=100, colors='black', linestyle='dashed')
    ax[1].hlines(y=env_dict['action_space_dict']['vel_select'][1], xmin=0, xmax=100, colors='black', linestyle='dashed', label='_nolegend_')
    for i in range(len(agent_names)):
        ax[1].plot(np.array(progress_history[i])*100, np.array(pose_history[i])[:,4], linewidth=1.5)
    ax[1].set_ylabel('Longitudinal\nvelocity [m/s]')
    ax[1].set_xlim(xlims)
    ax[1].set_ylim([env_dict['action_space_dict']['vel_select'][0]-0.2, env_dict['action_space_dict']['vel_select'][1]+0.2])
    ax[1].grid(True, color='lightgrey')
    ax[1].tick_params('both', length=0)
    ax[1].set_xticklabels([])

    ax[2].hlines(y=env_dict['car_params']['s_min'], xmin=0, xmax=100, colors='black', linestyle='dashed')
    ax[2].hlines(y=env_dict['car_params']['s_max'], xmin=0, xmax=100, colors='black', linestyle='dashed', label='_nolegend_')
    for i in range(len(agent_names)):
        ax[2].plot(np.array(progress_history[i])*100, np.array(pose_history[i])[:,3], linewidth=1.5)

    ax[2].set_ylabel('steering\nangle [rads]')
    ax[2].set_xlim(xlims)
    ax[2].set_ylim([env_dict['car_params']['s_min']-0.05, env_dict['car_params']['s_max']+0.05])
    ax[2].grid(True, color='lightgrey')
    ax[2].tick_params(axis=u'both', which=u'both',length=0)
    ax[2].set_xticklabels([])


    for i in range(len(agent_names)):
        ax[3].plot(np.array(progress_history[i])*100, np.array(state_history[i])[:,6], linewidth=1.5)
    
    ax[3].set_xlabel('Progress along centerline [%]')
    ax[3].set_ylabel('Slip\nangle [rads]')
    ax[3].set_xlim(xlims)
    ax[3].set_ylim([-1,1])
    ax[3].grid(True, color='lightgrey')
    ax[3].tick_params(axis=u'both', which=u'both',length=0)
    ax[3].set_yticks(np.arange(-1, 1.1, 0.5))


    legend.insert(0, 'Track centerline')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12) 
    plt.figlegend(legend, title=legend_title, loc='lower center', ncol=2)
    
    # plt.show() 
    plt.savefig('results/'+filename+'.pgf', format='pgf')

def display_path_steer_multiple(agent_names, ns, legend_title, legend, mismatch_parameters, frac_vary, start_condition, filename):
    
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
        if start_condition:
            env.reset(save_history=True, start_condition=start_condition, car_params=env_dict['car_params'])
        else:
            env.reset(save_history=True, start_condition=[], car_params=env_dict['car_params'])

        infile = open('agents/' + agent_name + '/' + agent_name + '_params', 'rb')
        agent_dict = pickle.load(infile)
        infile.close()

        infile = open('train_parameters/' + agent_name, 'rb')
        main_dict = pickle.load(infile)
        infile.close()
          
        if i==0 and not start_condition:
            infile = open('test_initial_condition/' + env_dict['map_name'], 'rb')
            start_conditions = pickle.load(infile)
            infile.close()
            start_condition = random.choice(start_conditions)

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
        env.reset(save_history=True, start_condition=start_condition, car_params=env_dict['car_params'])
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
        

    xlims = [0,100]

    legend_new = legend.copy()
    legend_new.insert(0, 'Min and max')

    legend_racetrack = legend.copy()
    legend_racetrack.insert(0, 'Track centerline')

    plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False,     # don't setup fonts from rc parameters
    "font.size": 12
    })

    fig, ax = plt.subplots(2, figsize=(5,4))
    alpha=0.8
    color='gray'
    for i in range(2):
        #ax[i].tick_params(axis='both', colors='lightgrey')
        ax[i].spines['bottom'].set_color(color)
        ax[i].spines['top'].set_color(color) 
        ax[i].spines['right'].set_color(color)
        ax[i].spines['left'].set_color(color)

    ax[0].tick_params(axis=u'both', which=u'both',length=0)
    
    track = mapping.map(env.map_name)
    ax[0].imshow(ImageOps.invert(track.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(1))), extent=(0,track.map_width,0,track.map_height), cmap="gray")
    ax[0].plot(env.rx, env.ry, color='gray', linestyle='dashed')
    
    for i in range(len(agent_names)):
        ax[0].plot(np.array(pose_history[i])[:,0], np.array(pose_history[i])[:,1], linewidth=1.5, alpha=alpha)   

    prog = np.array([0, 0.2, 0.4, 0.6, 0.8])
    idx =  np.zeros(len(prog), int)
    text = ['', '20%', '40%', '60%', '80%']

    for i in range(len(idx)):
        idx[i] = np.mod(env.start_point+np.round(prog[i]*len(env.rx)), len(env.rx))
    idx.astype(int)
    
    for i in range(len(idx)):
        ax[0].text(x=env.rx[idx[i]], y=env.ry[idx[i]], s=text[i], fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))
    
    ax[0].vlines(x=env.rx[idx[0]], ymin=env.ry[idx[0]]-1, ymax=env.ry[idx[0]]+1, linestyles='dotted', color='red')
    ax[0].text(x=env.rx[idx[0]]-1.2, y=env.ry[idx[0]]+1.3, s='Start/finish', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))

    ax[0].axis('off')
    # # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot


    ax[1].hlines(y=env_dict['car_params']['s_min'], xmin=0, xmax=100, colors='black', linestyle='dashed')
    ax[1].hlines(y=env_dict['car_params']['s_max'], xmin=0, xmax=100, colors='black', linestyle='dashed', label='_nolegend_')
    for i in range(len(agent_names)):
        ax[1].plot(np.array(progress_history[i])*100, np.array(pose_history[i])[:,3], linewidth=1, alpha=alpha)

    ax[1].set_ylabel('steering\nangle [rads]')
    ax[1].set_xlim(xlims)
    ax[1].set_ylim([env_dict['car_params']['s_min']-0.05, env_dict['car_params']['s_max']+0.05])
    ax[1].grid(True, color='lightgrey')
    ax[1].tick_params(axis=u'both', which=u'both',length=0)
    #ax[1].set_xticklabels([])
    ax[1].set_xlabel('Progress along centerline [%]')

    legend.insert(0, 'Track centerline')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28) 
    plt.figlegend(legend, title=legend_title, loc='lower center', ncol=2)
    
    # plt.show() 
    plt.savefig('results/'+filename+'.pgf', format='pgf')

def display_only_path_multiple(agent_names, ns, legend_title, legend, mismatch_parameters, frac_vary, start_condition, filename):
    
    pose_history = []
    progress_history = []
    state_history = []
    local_path_history = []


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
        if start_condition:
            env.reset(save_history=True, start_condition=start_condition, car_params=env_dict['car_params'])
        else:
            env.reset(save_history=True, start_condition=[], car_params=env_dict['car_params'])

        infile = open('agents/' + agent_name + '/' + agent_name + '_params', 'rb')
        agent_dict = pickle.load(infile)
        infile.close()

        infile = open('train_parameters/' + agent_name, 'rb')
        main_dict = pickle.load(infile)
        infile.close()
          
        if i==0 and not start_condition:
            infile = open('test_initial_condition/' + env_dict['map_name'], 'rb')
            start_conditions = pickle.load(infile)
            infile.close()
            start_condition = random.choice(start_conditions)

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
        env.reset(save_history=True, start_condition=start_condition, car_params=env_dict['car_params'])
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
        local_path_history.append(env.local_path_history)
        
        
    



    legend_new = legend.copy()
    legend_new.insert(0, 'Min and max')

    legend_racetrack = legend.copy()
    legend_racetrack.insert(0, 'Track centerline')

    # plt.rcParams.update({
    # "font.family": "serif",  # use serif/main font for text elements
    # "text.usetex": True,     # use inline math for ticks
    # "pgf.rcfonts": False,     # don't setup fonts from rc parameters
    # "font.size": 12
    # })

    fig, ax = plt.subplots(1, figsize=(5,2.5))

    color='gray'
    #ax[i].tick_params(axis='both', colors='lightgrey')
    ax.spines['bottom'].set_color(color)
    ax.spines['top'].set_color(color) 
    ax.spines['right'].set_color(color)
    ax.spines['left'].set_color(color)

    ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    track = mapping.map(env.map_name)
    ax.imshow(ImageOps.invert(track.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(1))), extent=(0,track.map_width,0,track.map_height), cmap="gray")
    ax.plot(env.rx, env.ry, color='gray', linestyle='dashed', label='Track centerline')
    
    for i in range(len(agent_names)):
        ax.plot(np.array(pose_history[i])[:,0], np.array(pose_history[i])[:,1], linewidth=1.5, label='Vehicle path history')   


    #prog = np.array([0, 0.2, 0.4, 0.6, 0.8])
    #idx =  np.zeros(len(prog), int)
    #text = ['', '20%', '40%', '60%', '80%']
    # for i in range(len(idx)):
    #     idx[i] = np.mod(env.start_point+np.round(prog[i]*len(env.rx)), len(env.rx))
    # idx.astype(int)
    
    # for i in range(len(idx)):
    #     ax.text(x=env.rx[idx[i]], y=env.ry[idx[i]], s=text[i], fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))
    
    #ax.vlines(x=env.rx[idx[0]], ymin=env.ry[idx[0]]-1, ymax=env.ry[idx[0]]+1, linestyles='dotted', color='red')
    #ax.text(x=env.rx[idx[0]]-1.2, y=env.ry[idx[0]]+1.3, s='Start/finish', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))

    # ax.axis('off')
    # # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot


    legend.insert(0, 'Track centerline')
    fig.tight_layout()
    #fig.subplots_adjust(right=0.6) 
    plt.figlegend(loc='center right', ncol=1, labelspacing=0.7)
    
    plt.show() 
    # plt.savefig('results/'+filename+'.pgf', format='pgf')
    pass

def display_path_two_multiple(agent_names, ns, legend_title, legend, mismatch_parameters, frac_vary, start_condition, filename):
    
    pose_history = []
    progress_history = []
    state_history = []
    local_path_history = []


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
        if start_condition:
            env.reset(save_history=True, start_condition=start_condition, car_params=env_dict['car_params'])
        else:
            env.reset(save_history=True, start_condition=[], car_params=env_dict['car_params'])

        infile = open('agents/' + agent_name + '/' + agent_name + '_params', 'rb')
        agent_dict = pickle.load(infile)
        infile.close()

        infile = open('train_parameters/' + agent_name, 'rb')
        main_dict = pickle.load(infile)
        infile.close()
          
        if i==0 and not start_condition:
            infile = open('test_initial_condition/' + env_dict['map_name'], 'rb')
            start_conditions = pickle.load(infile)
            infile.close()
            start_condition = random.choice(start_conditions)

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
        env.reset(save_history=True, start_condition=start_condition, car_params=env_dict['car_params'])
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
        local_path_history.append(env.local_path_history)
        
        
    



    legend_new = legend.copy()
    legend_new.insert(0, 'Min and max')

    legend_racetrack = legend.copy()
    legend_racetrack.insert(0, 'Track centerline')

    plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False,     # don't setup fonts from rc parameters
    "font.size": 12
    })

    fig, ax = plt.subplots(2, figsize=(5,2))

    for i in range(2):
        ax[i].axis('off')

    track = mapping.map(env.map_name)
    for i in range(2):
        ax[i].imshow(ImageOps.invert(track.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(1))), extent=(0,track.map_width,0,track.map_height), cmap="gray")
        #ax[i].plot(env.rx, env.ry, color='gray', linestyle='dashed', label='Track centerline')
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    for i in [0,1]:
        ax[0].plot(np.array(pose_history[i])[:,0], np.array(pose_history[i])[:,1], linewidth=1.5, color=colors[i])   

    for i in [2,3]:
        ax[1].plot(np.array(pose_history[i])[:,0], np.array(pose_history[i])[:,1], linewidth=1.5, color=colors[i])   


    #prog = np.array([0, 0.2, 0.4, 0.6, 0.8])
    #idx =  np.zeros(len(prog), int)
    #text = ['', '20%', '40%', '60%', '80%']
    # for i in range(len(idx)):
    #     idx[i] = np.mod(env.start_point+np.round(prog[i]*len(env.rx)), len(env.rx))
    # idx.astype(int)
    
    # for i in range(len(idx)):
    #     ax.text(x=env.rx[idx[i]], y=env.ry[idx[i]], s=text[i], fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))
    
    #ax.vlines(x=env.rx[idx[0]], ymin=env.ry[idx[0]]-1, ymax=env.ry[idx[0]]+1, linestyles='dotted', color='red')
    #ax.text(x=env.rx[idx[0]]-1.2, y=env.ry[idx[0]]+1.3, s='Start/finish', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))

    # ax.axis('off')
    # # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot


    #legend.insert(0, 'Track centerline')
    fig.tight_layout()
    fig.subplots_adjust(right=0.5) 
    plt.figlegend(legend, title=legend_title, loc='center right', ncol=1, labelspacing=0.7)
    
    # plt.show() 
    plt.savefig('results/'+filename+'.pgf', format='pgf')
    pass



def display_path_velocity_multiple(agent_names, ns, legend_title, legend, mismatch_parameters, frac_vary, start_condition, filename):
    
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
        if start_condition:
            env.reset(save_history=True, start_condition=start_condition, car_params=env_dict['car_params'])
        else:
            env.reset(save_history=True, start_condition=[], car_params=env_dict['car_params'])

        infile = open('agents/' + agent_name + '/' + agent_name + '_params', 'rb')
        agent_dict = pickle.load(infile)
        infile.close()

        infile = open('train_parameters/' + agent_name, 'rb')
        main_dict = pickle.load(infile)
        infile.close()
          
        if i==0 and not start_condition:
            infile = open('test_initial_condition/' + env_dict['map_name'], 'rb')
            start_conditions = pickle.load(infile)
            infile.close()
            start_condition = random.choice(start_conditions)

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
        env.reset(save_history=True, start_condition=start_condition, car_params=env_dict['car_params'])
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
        
        
        
    

    xlims = [0,100]

    legend_new = legend.copy()
    legend_new.insert(0, 'Min and max')

    legend_racetrack = legend.copy()
    legend_racetrack.insert(0, 'Track centerline')

    plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False,     # don't setup fonts from rc parameters
    "font.size": 12
    })

    fig, ax = plt.subplots(4, figsize=(5,7))

    color='gray'
    for i in range(4):
        #ax[i].tick_params(axis='both', colors='lightgrey')
        ax[i].spines['bottom'].set_color(color)
        ax[i].spines['top'].set_color(color) 
        ax[i].spines['right'].set_color(color)
        ax[i].spines['left'].set_color(color)

    ax[0].tick_params(axis=u'both', which=u'both',length=0)
    
    track = mapping.map(env.map_name)
    ax[0].imshow(ImageOps.invert(track.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(1))), extent=(0,track.map_width,0,track.map_height), cmap="gray")
    ax[0].plot(env.rx, env.ry, color='gray', linestyle='dashed')
    
    for i in range(len(agent_names)):
        ax[0].plot(np.array(pose_history[i])[:,0], np.array(pose_history[i])[:,1], linewidth=1.5)   

    prog = np.array([0, 0.2, 0.4, 0.6, 0.8])
    idx =  np.zeros(len(prog), int)
    text = ['', '20%', '40%', '60%', '80%']

    for i in range(len(idx)):
        idx[i] = np.mod(env.start_point+np.round(prog[i]*len(env.rx)), len(env.rx))
    idx.astype(int)
    
    for i in range(len(idx)):
        ax[0].text(x=env.rx[idx[i]], y=env.ry[idx[i]], s=text[i], fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))
    
    ax[0].vlines(x=env.rx[idx[0]], ymin=env.ry[idx[0]]-1, ymax=env.ry[idx[0]]+1, linestyles='dotted', color='red')
    ax[0].text(x=env.rx[idx[0]]-1.2, y=env.ry[idx[0]]+1.3, s='Start/finish', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))

    ax[0].axis('off')
    # # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot



    ax[1].hlines(y=env_dict['action_space_dict']['vel_select'][0], xmin=0, xmax=100, colors='black', linestyle='dashed')
    ax[1].hlines(y=env_dict['action_space_dict']['vel_select'][1], xmin=0, xmax=100, colors='black', linestyle='dashed', label='_nolegend_')
    for i in range(len(agent_names)):
        ax[1].plot(np.array(progress_history[i])*100, np.array(pose_history[i])[:,4], linewidth=1.5)
    ax[1].set_ylabel('Longitudinal\nvelocity [m/s]')
    ax[1].set_xlim(xlims)
    ax[1].set_ylim([env_dict['action_space_dict']['vel_select'][0]-0.2, env_dict['action_space_dict']['vel_select'][1]+0.2])
    ax[1].grid(True, color='lightgrey')
    ax[1].tick_params('both', length=0)
    ax[1].set_xticklabels([])

    ax[2].hlines(y=env_dict['car_params']['s_min'], xmin=0, xmax=100, colors='black', linestyle='dashed')
    ax[2].hlines(y=env_dict['car_params']['s_max'], xmin=0, xmax=100, colors='black', linestyle='dashed', label='_nolegend_')
    for i in range(len(agent_names)):
        ax[2].plot(np.array(progress_history[i])*100, np.array(pose_history[i])[:,3], linewidth=1.5)

    ax[2].set_ylabel('steering\nangle [rads]')
    ax[2].set_xlim(xlims)
    ax[2].set_ylim([env_dict['car_params']['s_min']-0.05, env_dict['car_params']['s_max']+0.05])
    ax[2].grid(True, color='lightgrey')
    ax[2].tick_params(axis=u'both', which=u'both',length=0)
    ax[2].set_xticklabels([])


    for i in range(len(agent_names)):
        ax[3].plot(np.array(progress_history[i])*100, np.array(state_history[i])[:,6], linewidth=1.5)
    
    ax[3].set_xlabel('Progress along centerline [%]')
    ax[3].set_ylabel('Slip\nangle [rads]')
    ax[3].set_xlim(xlims)
    ax[3].set_ylim([-1,1])
    ax[3].grid(True, color='lightgrey')
    ax[3].tick_params(axis=u'both', which=u'both',length=0)
    ax[3].set_yticks(np.arange(-1, 1.1, 0.5))


    legend.insert(0, 'Track centerline')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12) 
    plt.figlegend(legend, title=legend_title, loc='lower center', ncol=2)
    
    # plt.show() 
    plt.savefig('results/'+filename+'.pgf', format='pgf')


def sensitivity_analysis_noise(agent_name, n, start_condition, filename):
    
    n_acts = 50
    lidar_noise = np.arange(0,1,0.01)
    action_history = np.zeros((len(lidar_noise), n_acts))
    action_long_history = np.zeros((len(lidar_noise), n_acts))
    init_noise_dict = {'xy':0, 'theta':0, 'v':0, 'lidar':0}

    infile = open('environments/' + agent_name, 'rb')
    env_dict = pickle.load(infile)
    infile.close()
    env = environment(env_dict)
    
    infile = open('agents/' + agent_name + '/' + agent_name + '_params', 'rb')
    agent_dict = pickle.load(infile)
    infile.close()

    infile = open('train_parameters/' + agent_name, 'rb')
    main_dict = pickle.load(infile)
    infile.close()
        
    a = agent_td3.agent(agent_dict)
    a.load_weights(agent_name, n)

    env.reset(save_history=True, start_condition=start_condition, car_params=env_dict['car_params'], noise=init_noise_dict)
    obs = env.observation
    
    for idx, l_n in enumerate(lidar_noise):
        noise_dict=init_noise_dict.copy()
        noise_dict['lidar'] = l_n
        for i in range(n_acts):
            env.reset(save_history=True, start_condition=start_condition, car_params=env_dict['car_params'], noise=noise_dict)
            obs = env.observation
            action = a.choose_greedy_action(obs)
            action_history[idx,i] = action[0]
            action_long_history[idx,i] = action[1]

    avg = np.average(action_history, axis=1)
    std_dev = np.std(action_history, axis=1)

    avg_long = np.average(action_long_history, axis=1)
    std_dev_long = np.std(action_long_history, axis=1)

    avg_filter = functions.savitzky_golay(avg, 9, 2)
    std_dev_filter = functions.savitzky_golay(std_dev, 9, 2)

    avg_filter_long = functions.savitzky_golay(avg_long, 9, 2)
    std_dev_filter_long = functions.savitzky_golay(std_dev_long, 9, 2)
    
    plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False,     # don't setup fonts from rc parameters
    "font.size": 12
    })

    fig, ax = plt.subplots(1, figsize=(5.5,2.3))

    ax.spines['bottom'].set_color('lightgrey')
    ax.spines['top'].set_color('lightgrey') 
    ax.spines['right'].set_color('lightgrey')
    ax.spines['left'].set_color('lightgrey')
    ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.grid(True)
    ax.set_title('')
    ax.set_xlabel('LiDAR noise standard deviation [m]')
    ax.set_ylabel('Control action value')
    # ax.plot(lidar_noise, avg)
    # ax.fill_between(x=lidar_noise, y1=avg-std_dev, y2=avg+std_dev,alpha=0.5)

    ax.plot(lidar_noise, avg_filter)
    ax.fill_between(x=lidar_noise, y1=avg_filter-std_dev_filter, y2=avg_filter+std_dev_filter,alpha=0.3, label='_nolegend_')
    
    ax.plot(lidar_noise, avg_filter_long)
    ax.fill_between(x=lidar_noise, y1=avg_filter_long-std_dev_filter_long, y2=avg_filter_long+std_dev_filter_long,alpha=0.3, label='_nolegend_')
    
    ax.hlines(y=1,xmin=0,xmax=lidar_noise[-1], color='k', linestyles='--')
    ax.hlines(y=-1,xmin=0,xmax=lidar_noise[-1], color='k', linestyles='--')

    ax.set_xlim([0,0.3])
    fig.tight_layout()
    fig.subplots_adjust(right=0.7)
    plt.figlegend(['Steering', 'Acceleration', 'Action limits'], loc='center right', ncol=1)
    plt.savefig('results/'+filename+'.pgf', format='pgf')
    # plt.show()
    pass


agent_name = 'porto_ete_v5_r_collision_5' 
n=0
start_condition = {'x':4, 'y':4.8, 'v':5, 'theta':np.pi, 'delta':0, 'goal':0}
filename='action_noise'
sensitivity_analysis_noise(agent_name=agent_name, n=n, start_condition=start_condition, filename=filename)



def display_path_actions_multiple(agent_names, ns, legend_title, legend, mismatch_parameters, frac_vary, start_condition, filename):
    
    pose_history = []
    progress_history = []
    state_history = []
    action_step_history = []

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
        if start_condition:
            env.reset(save_history=True, start_condition=start_condition, car_params=env_dict['car_params'])
        else:
            env.reset(save_history=True, start_condition=[], car_params=env_dict['car_params'])

        infile = open('agents/' + agent_name + '/' + agent_name + '_params', 'rb')
        agent_dict = pickle.load(infile)
        infile.close()

        infile = open('train_parameters/' + agent_name, 'rb')
        main_dict = pickle.load(infile)
        infile.close()
          
        if i==0 and not start_condition:
            infile = open('test_initial_condition/' + env_dict['map_name'], 'rb')
            start_conditions = pickle.load(infile)
            infile.close()
            start_condition = random.choice(start_conditions)

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
        env.reset(save_history=True, start_condition=start_condition, car_params=env_dict['car_params'])
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
        action_step_history.append(env.action_step_history)
        
        
    

    xlims = [0,100]

    legend_new = legend.copy()
    legend_new.insert(0, 'Min and max')

    legend_racetrack = legend.copy()
    legend_racetrack.insert(0, 'Track centerline')

    plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False,     # don't setup fonts from rc parameters
    "font.size": 12
    })

    fig, ax = plt.subplots(3, figsize=(5,5))
    alpha = 0.8
    color='gray'
    for i in range(3):
        #ax[i].tick_params(axis='both', colors='lightgrey')
        ax[i].spines['bottom'].set_color(color)
        ax[i].spines['top'].set_color(color) 
        ax[i].spines['right'].set_color(color)
        ax[i].spines['left'].set_color(color)

    ax[0].tick_params(axis=u'both', which=u'both',length=0)
    
    track = mapping.map(env.map_name)
    ax[0].imshow(ImageOps.invert(track.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(1))), extent=(0,track.map_width,0,track.map_height), cmap="gray")
    # ax[0].plot(env.rx, env.ry, color='gray', linestyle='dashed')
    
    for i in range(len(agent_names)):
        ax[0].plot(np.array(pose_history[i])[:,0], np.array(pose_history[i])[:,1], linewidth=1.5)   

    prog = np.array([0, 0.2, 0.4, 0.6, 0.8])
    idx =  np.zeros(len(prog), int)
    text = ['', '20%', '40%', '60%', '80%']

    for i in range(len(idx)):
        idx[i] = np.mod(env.start_point+np.round(prog[i]*len(env.rx)), len(env.rx))
    idx.astype(int)
    
    for i in range(len(idx)):
        ax[0].text(x=env.rx[idx[i]], y=env.ry[idx[i]], s=text[i], fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))
    
    ax[0].vlines(x=env.rx[idx[0]], ymin=env.ry[idx[0]]-1, ymax=env.ry[idx[0]]+1, linestyles='dotted', color='red')
    ax[0].text(x=env.rx[idx[0]]-1.2, y=env.ry[idx[0]]+1.3, s='Start/finish', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))

    ax[0].axis('off')
    # # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot



    ax[1].hlines(y=env_dict['action_space_dict']['vel_select'][0], xmin=0, xmax=100, colors='black', linestyle='dashed')
    ax[1].hlines(y=env_dict['action_space_dict']['vel_select'][1], xmin=0, xmax=100, colors='black', linestyle='dashed', label='_nolegend_')
    for i in range(len(agent_names)):
        ax[1].plot(np.array(progress_history[i])*100, np.array(pose_history[i])[:,4], linewidth=1.5)
    ax[1].set_ylabel('Longitudinal\nvelocity [m/s]')
    ax[1].set_xlim(xlims)
    ax[1].set_ylim([env_dict['action_space_dict']['vel_select'][0]-0.2, env_dict['action_space_dict']['vel_select'][1]+0.2])
    ax[1].grid(True, color='lightgrey')
    ax[1].tick_params('both', length=0)
    ax[1].set_xticklabels([])


    ax[2].hlines(y=1, xmin=0, xmax=100, colors='black', linestyle='dashed')
    ax[2].hlines(y=-1, xmin=0, xmax=100, colors='black', linestyle='dashed', label='_nolegend_')
    for i in range(len(agent_names)):
        plt.plot(np.array(progress_history[i])[0:len(np.array(action_step_history[i])[:,1])]*100, np.array(action_step_history[i])[:,1], linewidth=1.5, alpha=alpha)
    ax[2].set_ylabel('Longitudinal\naction')
    ax[2].set_xlim(xlims)
    # ax[2].set_ylim([env_dict['action_space_dict']['vel_select'][0]-0.2, env_dict['action_space_dict']['vel_select'][1]+0.2])
    ax[2].grid(True, color='lightgrey')
    ax[2].tick_params('both', length=0)
    #ax[2].set_xticklabels([])
    ax[2].set_xlabel('Progress along centerline [%]')

   


    # legend.insert(0, 'Track centerline')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.17) 
    plt.figlegend(legend, title=legend_title, loc='lower center', ncol=5)
    
    # plt.show() 
    plt.savefig('results/'+filename+'.pgf', format='pgf')


def display_velocity_profile(agent_names, ns, legend_title, legend, mismatch_parameters, frac_vary, start_condition, filename):
    
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
        if start_condition:
            env.reset(save_history=True, start_condition=start_condition, car_params=env_dict['car_params'])
        else:
            env.reset(save_history=True, start_condition=[], car_params=env_dict['car_params'])

        infile = open('agents/' + agent_name + '/' + agent_name + '_params', 'rb')
        agent_dict = pickle.load(infile)
        infile.close()

        infile = open('train_parameters/' + agent_name, 'rb')
        main_dict = pickle.load(infile)
        infile.close()
          
        if i==0 and not start_condition:
            infile = open('test_initial_condition/' + env_dict['map_name'], 'rb')
            start_conditions = pickle.load(infile)
            infile.close()
            start_condition = random.choice(start_conditions)

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
        env.reset(save_history=True, start_condition=start_condition, car_params=env_dict['car_params'])
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
        
        
        
    

    xlims = [0,100]

    legend_new = legend.copy()
    legend_new.insert(0, 'Min and max')

    legend_racetrack = legend.copy()
    legend_racetrack.insert(0, 'Track centerline')

    plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False,     # don't setup fonts from rc parameters
    "font.size": 12
    })

    fig, ax = plt.subplots(1, figsize=(5,2))

    color='gray'
    ax.spines['bottom'].set_color(color)
    ax.spines['top'].set_color(color) 
    ax.spines['right'].set_color(color)
    ax.spines['left'].set_color(color)

  


    ax.hlines(y=env_dict['action_space_dict']['vel_select'][0], xmin=0, xmax=100, colors='black', linestyle='dashed', label='_nolegend_')
    ax.hlines(y=env_dict['action_space_dict']['vel_select'][1], xmin=0, xmax=100, colors='black', linestyle='dashed', label='_nolegend_')
    for i in range(len(agent_names)):
        ax.plot(np.array(progress_history[i])*100, np.array(pose_history[i])[:,4], linewidth=1.5)
    ax.set_ylabel('Longitudinal\nvelocity [m/s]')
    ax.set_xlim(xlims)
    ax.set_ylim([env_dict['action_space_dict']['vel_select'][0]-0.2, env_dict['action_space_dict']['vel_select'][1]+0.2])
    ax.grid(True, color='lightgrey')


    ax.set_xlabel('Progress along centerline [%]')
    ax.grid(True, color='lightgrey')
    ax.tick_params(axis=u'both', which=u'both',length=0)
    # ax[0].set_yticks(np.arange(-1, 1.1, 0.5))


    # legend.insert(0, 'Track centerline')
    fig.tight_layout()
    fig.subplots_adjust(right=0.8) 
    plt.figlegend(legend, title=legend_title, loc='center right', ncol=1)
    
    # plt.show() 
    plt.savefig('results/'+filename+'.pgf', format='pgf')


def display_lap_noise_results_multiple(agent_names, noise_params, legend_title, legend, filename):
    
    plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False,     # don't setup fonts from rc parameters
    "font.size": 12
    })
    
    fig, axs = plt.subplots(len(noise_params), 2, figsize=(5.5,8))
    numbering = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']

    for j, parameter in enumerate(noise_params):
        
        color='gray'
        axs[j,0].spines['bottom'].set_color(color)
        axs[j,0].spines['top'].set_color(color) 
        axs[j,0].spines['right'].set_color(color)
        axs[j,0].spines['left'].set_color(color)

        axs[j,1].spines['bottom'].set_color(color)
        axs[j,1].spines['top'].set_color(color) 
        axs[j,1].spines['right'].set_color(color)
        axs[j,1].spines['left'].set_color(color)

        # axs[j,0].set_yticks(ticks=[50,75,100], labels=[50,75,100])
        # axs[j,1].set_yticks(ticks=[5,6,7], labels=[5,6,7])
        

        for agent in agent_names:
            
            #infile = open('lap_results_mismatch/' + agent + '_new/' + parameter, 'rb')
            infile = open('lap_results_noise/' + agent + '/' + parameter, 'rb')
            results_dict = pickle.load(infile)
            infile.close() 

            n_episodes = len(results_dict['collision_results'][0,0,:])
            n_param = len(results_dict['collision_results'][0,:,0])
            n_runs = len(results_dict['collision_results'][:,0,0])

            avg_col = np.zeros(n_param)
            avg_time = np.zeros(n_param)
            dev = np.zeros(n_param)



            # for i in range(n_param):
            #     avg_col[i] = np.sum(np.logical_not(results_dict['collision_results'][:,i,:]))/(n_episodes*n_runs)
            #     avg_time[i] =  np.ma.mean((results_dict['times_results'][:,i,:]))
            #     # failures = np.count_nonzero(results_dict['collision_results'][:,0,:].flatten())
            #     # successes = n_episodes - failures
            #     # dev[i] = np.sqrt(n_episodes*(successes/n_episodes)*((failures)/n_episodes))/(n_episodes*n_runs)
            
            avg_cols = np.mean(np.mean(np.logical_not(results_dict['collision_results']),axis=2),axis=0)*100
            avg_times = np.mean(np.ma.array(results_dict['times_results'], mask=results_dict['collision_results'].astype(bool)).mean(axis=2),axis=0)
            
            # kernel_size = 2
            # kernel = np.ones(kernel_size) / kernel_size
            # avg_cols_convolved = np.convolve(avg_cols, kernel, mode='same')
            # avg_times_convolved = np.convolve(avg_times, kernel, mode='same')

            avg_cols_filter = functions.savitzky_golay(avg_cols, 9, 2)
            avg_times_filter = functions.savitzky_golay(avg_times, 9, 2)

            axs[j,0].grid(True)
            axs[j,0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            axs[j,0].tick_params('both', length=0)
            axs[j,0].plot(results_dict['noise_std_values'], avg_cols_filter)
            # axs[j].fill_between(results_dict['noise_std_values'], avg-dev, avg+dev, alpha=0.25)
            axs[j,0].set(ylabel='Successful\nlaps [%]')
            # axs.yaxis.set_major_formatter(plt.ticker.FormatStrFormatter('%.2f'))
            

            axs[j,1].grid(True)
            # axs[j,1].set_ylim([5,7])
            axs[j,1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            axs[j,1].tick_params('both', length=0)
            axs[j,1].plot(results_dict['noise_std_values'], avg_times_filter)
            # axs[j].fill_between(results_dict['noise_std_values'], avg-dev, avg+dev, alpha=0.25)
            axs[j,1].set(ylabel='Lap time [s]')

        #axs[j].set_title('(' + numbering[j] + ') ' + plot_titles[j])
    # axs[j].set(xlabel='standard deviation')
    # axs[j].legend(legend, title=legend_title, loc='lower right')
    space = '                                               '
    axs[0,0].set_title('                                               $x$ and $y$ coordinates')
    axs[1,0].set_title('                                               Vehicle heading')
    axs[2,0].set_title('                                               Velocity')
    axs[3,0].set_title('                                               LiDAR')


    
    axs[0,0].set_xlabel('Noise standard deviation [m]')
    axs[0,1].set_xlabel('Noise standard deviation [m]')
    axs[1,0].set_xlabel('Noise standard deviation [rads]')
    axs[1,1].set_xlabel('Noise standard deviation [rads]')
    axs[2,0].set_xlabel('Noise standard deviation [m/s]')
    axs[2,1].set_xlabel('Noise standard deviation [m/s]')
    axs[3,0].set_xlabel('Noise standard deviation [m]')
    axs[3,1].set_xlabel('Noise standard deviation [m]')

    axs[0,0].set_xlim([0,0.3])
    # axs[1,0].set_xlim([0,0.15])
    # axs[2,0].set_xlim([0,0.4])
    # axs[3,0].set_xlim([0,0.4])

    axs[0,1].set_xlim([0,0.3])
    # axs[1,1].set_xlim([0,0.4])
    # axs[2,1].set_xlim([0,0.4])
    # axs[3,1].set_xlim([0,0.4])

    axs[0,0].set_ylim([45,105])
    axs[1,0].set_ylim([90,101])
    # axs[2,0].set_ylim([50,105])
    # axs[3,0].set_ylim([50,105])

    # axs[0,1].set_ylim([5,7])
    # axs[1,1].set_ylim([50,105])
    # axs[2,1].set_ylim([50,105])
    # axs[3,1].set_ylim([50,105])



    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14) 
    plt.figlegend(legend, title=legend_title, loc = 'lower center', ncol=2)
    plt.savefig('results/'+filename+'.pgf', format='pgf')
    # plt.show()

agent_names = ['porto_ete_v5_r_collision_5', 'porto_pete_s_polynomial', 'porto_pete_v_k_1_attempt_2', 'porto_pete_sv_p_r_0']    
noise_params = ['xy', 'theta', 'v', 'lidar']
legend = ['End-to-end', 'Steering control', 'Velocity control', 'Steering and velocity control']
legend_title = ''
filename = 'noise_vary'
display_lap_noise_results_multiple(agent_names, noise_params, legend_title, legend, filename)


def display_lap_unknown_mass(agent_names, legend, filename):
    
    plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False,     # don't setup fonts from rc parameters
    "font.size": 12
    })
    
    fig, axs = plt.subplots(1, figsize=(5.5,3))

  
    for agent in agent_names:
        
        #infile = open('lap_results_mismatch/' + agent + '_new/' + parameter, 'rb')
        infile = open('lap_results_mismatch/' + agent + '/' + 'unknown_mass', 'rb')
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

        avg_filter = functions.savitzky_golay(avg, 13, 2)
        # std_dev_filter = functions.savitzky_golay(std_dev, 9, 2)

        axs.plot(results_dict['distances'], avg_filter*100, alpha=0.8)


    axs.vlines(x=0,ymin=90,ymax=100,color='black',linestyle='--', label='_nolegend_', alpha=0.8)
    axs.vlines(x=0.33,ymin=90,ymax=100,color='black',linestyle='--', alpha=0.8)

    axs.text(x=0.355, y=90.5, s='Rear axle', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))
    axs.text(x=0.025, y=90.5, s='Front axle', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))

    axs.set_ylim([90,100.5])
    axs.grid()
    axs.set(ylabel='Successful laps [%]')   
    axs.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # axs.set_title('Lap success rate with unknown mass')
    axs.set(xlabel='Distance of payload mass from front axle [m]')
    # axs.legend(legend, title=legend_title, loc='lower right')
    plt.gca().invert_xaxis()
    plt.grid(True)
    axs.spines['bottom'].set_color('grey')
    axs.spines['top'].set_color('grey') 
    axs.spines['right'].set_color('grey')
    axs.spines['left'].set_color('grey')
    
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35) 
    plt.figlegend(legend,loc='lower center', ncol=2)
    # plt.show()
    plt.savefig('results/'+filename+'.pgf', format='pgf')

def display_lap_unknown_mass_time(agent_names, legend, filename):
    
    plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False,     # don't setup fonts from rc parameters
    "font.size": 12
    })

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(5.5,2.8))

  
    for agent in agent_names:
        
        #infile = open('lap_results_mismatch/' + agent + '_new/' + parameter, 'rb')
        infile = open('lap_results_mismatch/' + agent + '/' + 'unknown_mass', 'rb')
        results_dict = pickle.load(infile)
        infile.close() 

        n_episodes = len(results_dict['collision_results'][0,0,:])
        n_param = len(results_dict['collision_results'][0,:,0])
        n_runs = len(results_dict['collision_results'][:,0,0])

        avg_col = np.zeros(n_param)
        dev_col = np.zeros(n_param)

        avg_time = np.zeros(n_param)
        dev_time = np.zeros(n_param)

        for i in range(n_param):
            avg_col[i] = np.round(np.sum(np.logical_not(results_dict['collision_results'][:,i,:]))/(n_episodes*n_runs), 2)
            failures = np.count_nonzero(results_dict['collision_results'][:,0,:].flatten())
            successes = n_episodes - failures
            dev_col[i] = np.sqrt(n_episodes*(successes/n_episodes)*((failures)/n_episodes))/(n_episodes*n_runs)
            

            # for i in range(np.size(end_ep, axis=0)):
            #     for n in range(np.size(end_ep, axis=1)):
            #         steps_y[i][n][collisions[i][n]==1]=np.nan
            # steps_y_avg = np.array(steps_y)
            # steps_y_avg = np.nanmean(steps_y_avg, axis=1)
            # avg_time[i] = np.nanmean()
            
            cols = results_dict['collision_results']
            times = results_dict['times_results']
            times[cols==1]=np.nan
            avg_times = np.mean(np.nanmean(times,axis=0),axis=1)
            dev_times= np.std(np.nanstd(times,axis=0),axis=1)

        avg_col_filter = functions.savitzky_golay(avg_col, 13, 2)
        std_dev_col_filter = functions.savitzky_golay(dev_col, 13, 2)

        avg_times_filter = functions.savitzky_golay(avg_times, 13, 2)
        dev_times_filter = functions.savitzky_golay(dev_times, 13, 2)

        axs[0].plot(results_dict['distances'], avg_col_filter*100, alpha=0.8)
        axs[0].fill_between(results_dict['distances'], (avg_col_filter+std_dev_col_filter)*100, (avg_col_filter-std_dev_col_filter)*100, alpha=0.2, label='_nolegend_')

        axs[1].plot(results_dict['distances'], avg_times_filter, alpha=0.8)
        axs[1].fill_between(results_dict['distances'], (avg_times_filter+dev_times_filter), (avg_times_filter-dev_times_filter), alpha=0.2, label='_nolegend_')

        # axs.plot(results_dict['distances'], avg_times*100, alpha=0.8)
        # axs.fill_between(results_dict['distances'], (avg_times+dev_times)*100, (avg_times-dev_times)*100, alpha=0.2, label='_nolegend_')
       
    
    axs[0].vlines(x=0,ymin=90,ymax=100,color='black',linestyle='--', label='_nolegend_', alpha=0.8)
    axs[0].vlines(x=0.3,ymin=90,ymax=100,color='black',linestyle='--', alpha=0.8)
    axs[0].text(x=0.37, y=90.5, s='Rear axle', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))
    axs[0].text(x=0.1, y=90.5, s='Front axle', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))

    axs[1].vlines(x=0,ymin=5.8,ymax=6.3,color='black',linestyle='--', label='_nolegend_', alpha=0.8)
    axs[1].vlines(x=0.33,ymin=5.8,ymax=6.3,color='black',linestyle='--', alpha=0.8)
    axs[1].text(x=0.37, y=6.25, s='Rear axle', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))
    axs[1].text(x=0.1, y=6.25, s='Front axle', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))



    axs[0].set(ylabel='Successful laps [%]') 
    axs[1].set(ylabel='Lap time [s]') 
    
    for i in range(2):
        axs[i].grid(True)  
        # axs.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        # axs.set_title('Lap success rate with unknown mass')
        axs[i].set(xlabel='Distance of payload mass\nfrom front axle [m]')
        # axs.legend(legend, title=legend_title, loc='lower right')
        # axs[i].gca().invert_xaxis()
        axs[i].spines['bottom'].set_color('grey')
        axs[i].spines['top'].set_color('grey') 
        axs[i].spines['right'].set_color('grey')
        axs[i].spines['left'].set_color('grey')
        axs[i].invert_xaxis()

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.48) 
    plt.figlegend(legend,loc='lower center', ncol=2)
    # plt.show()
    plt.savefig('results/'+filename+'.pgf', format='pgf')


agent_names = ['porto_ete_v5_r_collision_5', 'porto_pete_s_polynomial', 'porto_pete_v_k_1_attempt_2', 'porto_pete_sv_p_r_0']    
legend = ['End-to-end', 'Steering control', 'Velocity control', 'Steering and velocity control']
filename = 'unknown_mass'
# display_lap_unknown_mass_time(agent_names, legend, filename)


def display_path_mismatch_multiple(agent_names, ns, legend_title, legend, mismatch_parameters, frac_vary, noise_dicts, start_condition, filename):
    
    pose_history = []
    progress_history = []
    state_history = []
    local_path_history = []
    action_step_history = []
    

    for agent_name, n, i in zip(agent_names, ns, range(len(agent_names))):

        infile = open('environments/' + agent_name, 'rb')
        env_dict = pickle.load(infile)
        infile.close()
        # Compensate for changes to reward structure
        env_dict['reward_signal']['max_progress'] = 0
        
        # Model mismatches

        for mis_idx in range(2):
            car_params = env_dict['car_params'].copy()
    
            if mis_idx == 1:
                for par, var in zip(mismatch_parameters, frac_vary):
                    if par == 'unknown_mass':
                        mass=car_params['m']*0.1
                        m_new = car_params['m'] + mass
                        lf_new = (car_params['m']*car_params['lf']+mass*var) / (m_new)
                        I_new = car_params['I'] + car_params['m']*abs(lf_new-car_params['lf'])**2 + mass*abs(lf_new-var)**2
                        car_params['m'] = m_new
                        car_params['lf'] = lf_new
                        car_params['I'] = I_new
                    elif par == 'C_S':
                        car_params['C_Sf'] *= 1+var
                        car_params['C_Sr'] *= 1+var
                    elif par == 'l_f':
                        axle_length = car_params['lf']+car_params['lr']
                        car_params['lf'] *= 1+var
                        car_params['lr'] =  axle_length - car_params['lf']
                    elif par == 'sv':
                        car_params['sv_max'] *= 1+var
                        car_params['sv_min'] *= 1+var    
                    else:
                        car_params[par] *= 1+var

            
            noise_dict = noise_dicts[0]

            env = environment(env_dict)
            if start_condition:
                env.reset(save_history=True, start_condition=start_condition, car_params=car_params, noise=noise_dict)
            else:
                env.reset(save_history=True, start_condition=[], car_params=car_params, noise=noise_dict)

            infile = open('agents/' + agent_name + '/' + agent_name + '_params', 'rb')
            agent_dict = pickle.load(infile)
            infile.close()

            infile = open('train_parameters/' + agent_name, 'rb')
            main_dict = pickle.load(infile)
            infile.close()
            
            if i==0 and not start_condition:
                infile = open('test_initial_condition/' + env_dict['map_name'], 'rb')
                start_conditions = pickle.load(infile)
                infile.close()
                start_condition = random.choice(start_conditions)

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
            env.reset(save_history=True, start_condition=start_condition, car_params=car_params, noise=noise_dict)
            obs = env.observation
            done = False
            score = 0

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
            print('Collision = ', env.collision)

            state_history.append(env.state_history)
            pose_history.append(env.pose_history)
            progress_history.append(env.progress_history)
            local_path_history.append(env.local_path_history)
            action_step_history.append(env.action_step_history)
        
        


    plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False,     # don't setup fonts from rc parameters
    "font.size": 12
    })

    fig, ax =   plt.subplots(nrows=2, ncols=2, figsize=(5.5,2.8))
    plt_idx=0
    for graph in [[0,0], [0,1], [1,0], [1,1]]:
        y=graph[0]
        x=graph[1]

        ax[y,x].axis('off')
        
        track = mapping.map(env.map_name)
        ax[y,x].imshow(ImageOps.invert(track.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(1))), extent=(0,track.map_width,0,track.map_height), cmap="gray")
        # ax.plot(env.rx, env.ry, color='gray', linestyle='dashed')
        alpha=0.7

        for _ in range(2):
            ax[y,x].plot(np.array(state_history[plt_idx])[:,0], np.array(state_history[plt_idx])[:,1], linewidth=1.5, alpha=alpha)  
            plt_idx+=1
        
    ax[0,0].set_title('End-to-end', fontsize=12)
    ax[0,1].set_title('Steering controller', fontsize=12)
    ax[1,0].set_title('Velocity controller', fontsize=12)
    ax[1,1].set_title('Steering and velocity controllers', fontsize=12)
    
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12) 
    
    plt.figlegend(legend, loc='lower center', ncol=2, labelspacing=0.7)
    # plt.show()
    plt.savefig('results/'+filename+'.pgf', format='pgf')



agent_names = ['porto_ete_v5_r_collision_5', 'porto_pete_s_polynomial', 'porto_pete_v_k_1_attempt_2', 'porto_pete_sv_p_r_0']    
legend = ['No model error', 'Mass placed on front axle']
legend_title = ''
ns=[0,0,0,0]
mismatch_parameters = ['unknown_mass']
frac_vary = [0]
noise_dicts = [{'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}]
start_condition = {'x':10, 'y':4.5, 'v':3, 'theta':np.pi, 'delta':0, 'goal':0}
filename='unknown_mass_path'
# display_path_mismatch_multiple(agent_names=agent_names, ns=ns, legend_title=legend_title,          
#                                              legend=legend, mismatch_parameters=mismatch_parameters, frac_vary=frac_vary, noise_dicts=noise_dicts,
#                                              start_condition=start_condition, filename=filename)

def display_lap_mismatch_results_multiple_mu(agent_names, parameters, legend_title, legend, plot_titles, nom_value, graph, text, filename):
    
    plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False,     # don't setup fonts from rc parameters
    "font.size": 12
    })

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(5.5,3))

    for j, parameter in enumerate(parameters):
        for agent in agent_names:
            
            #infile = open('lap_results_mismatch/' + agent + '_new/' + parameter, 'rb')
            infile = open('lap_results_mismatch/' + agent + '/' + parameter, 'rb')
            results_dict = pickle.load(infile)
            infile.close() 

            n_episodes = len(results_dict['collision_results'][0,0,:])
            n_param = len(results_dict['collision_results'][0,:,0])
            n_runs = len(results_dict['collision_results'][:,0,0])

            avg_col = np.zeros(n_param)
            dev_col = np.zeros(n_param)

            avg_time = np.zeros(n_param)
            dev_time = np.zeros(n_param)

            for i in range(n_param):
                
                cols = results_dict['collision_results']
                times = results_dict['times_results']
                times[cols==1]=np.nan
                
                avg_col[i] = np.round(np.sum(np.logical_not(results_dict['collision_results'][:,i,:]))/(n_episodes*n_runs), 2)
                failures = np.count_nonzero(results_dict['collision_results'][:,i,:].flatten())
                successes = n_episodes*n_runs - failures
                dev_col[i] = np.sqrt(n_episodes*(successes/n_episodes)*((failures)/n_episodes))/(n_episodes*n_runs)
                

            avg_times = np.nanmean(times,axis=(0,2))
            dev_times = np.nanstd(times,axis=(0,2))

            avg_col_filter = functions.savitzky_golay(avg_col, 5, 2)
            dev_col_filter = functions.savitzky_golay(dev_col, 5, 2)

            avg_times_filter = functions.savitzky_golay(avg_times, 5, 2)
            dev_times_filter = functions.savitzky_golay(dev_times, 5, 2)

            
            if text==True:
                print(agent)
                print(f"{'parameter: ':11s} {parameter:13s}")
                print(f"{'Fraction:':10s}", end='')
                for i in range(n_param):
                    print(f"{results_dict['frac_variation'][i]:8.2f}", end='')
                    pass
                print('')
                print(f"{'Value:':10s}", end='')
                for i in range(n_param):
                    print(f"{nom_value[j]*(1+results_dict['frac_variation'])[i]:8.2f}", end='')
                    pass
                print('')
                print(f"{'Success:':10s}", end='')
                for i in range(n_param):
                    print(f"{avg_col[i]:8.2f}", end='')
                    pass
                print('')
                print(f"{'Lap time:':10s}", end='')
                for i in range(n_param):
                    print(f"{avg_times[i]:8.2f}", end='')
                    pass
                print('')


            # plot collisions
            if parameter=='C_S' or parameter=='sv' or parameter=='a_max':
                axs[0].plot(nom_value[j]*(1+results_dict['frac_variation']), avg_col_filter*100, alpha=0.8)
                axs[0].fill_between(nom_value[j]*(1+results_dict['frac_variation']), (avg_col_filter+dev_col_filter)*100, (avg_col_filter-dev_col_filter)*100, alpha=0.2, label='_nolegend_')
            else:
                axs[0].plot(nom_value[j]*(1+results_dict['frac_variation']), avg_col*100, alpha=0.8)
                axs[0].fill_between(nom_value[j]*(1+results_dict['frac_variation']), (avg_col+dev_col)*100, (avg_col-dev_col)*100, alpha=0.2, label='_nolegend_')
            
            # plot lap times
            axs[1].plot(nom_value[j]*(1+results_dict['frac_variation']), avg_times, alpha=0.8)
            axs[1].fill_between(nom_value[j]*(1+results_dict['frac_variation']), (avg_times+dev_times), (avg_times-dev_times), alpha=0.2, label='_nolegend_')

    axs[0].grid(True)
    axs[1].grid(True)
        
    axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs[0].tick_params('both', length=0)
    axs[1].tick_params('both', length=0)
    color='grey'
    axs[0].spines['bottom'].set_color(color)
    axs[0].spines['top'].set_color(color) 
    axs[0].spines['right'].set_color(color)
    axs[0].spines['left'].set_color(color)
    axs[1].spines['bottom'].set_color(color)
    axs[1].spines['top'].set_color(color) 
    axs[1].spines['right'].set_color(color)
    axs[1].spines['left'].set_color(color)

    # axs[0].set_title('                                                 '+parameter, fontsize=12)
    
    axs[0].set(ylabel='Successful laps [%]') 
    axs[1].set(ylabel='Lap time [s]') 
    
    xlabel = 'Friction coefficient, $\mu$'
    axs[0].set(xlabel=xlabel) 
    axs[1].set(xlabel=xlabel)
    
    axs[0].set_xlim([0.2,1.2])
    axs[1].set_xlim([0.2,1.2])
    # axs[j,0].set_xticks(ticks=, labels=)
    # axs[j,1].set_xticks(ticks=, labels=)

    axs[0].vlines(x=nom_value, ymin=-50, ymax=150, color='black', linestyle='--')
    axs[0].vlines(x=0.7, ymin=-50, ymax=150, color='black', linestyle='--')
    axs[0].vlines(x=0.5, ymin=-50, ymax=150, color='black', linestyle='--')
    axs[0].text(x=0.75, y=40, s='Nominal value', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))
    axs[0].text(x=0.6, y=25, s='Dry asphalt', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))
    axs[0].text(x=0.4, y=10, s='Wet asphalt', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))
    
    axs[1].vlines(x=nom_value, ymin=0, ymax=15, color='black', linestyle='--')
    axs[1].vlines(x=0.7, ymin=0, ymax=15, color='black', linestyle='--')
    axs[1].vlines(x=0.5, ymin=0, ymax=15, color='black', linestyle='--')
    axs[1].text(x=0.75, y=7.3, s='Nominal value', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))
    axs[1].text(x=0.6, y=7.9, s='Dry asphalt', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))
    axs[1].text(x=0.4, y=8.5, s='Wet asphalt', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))

    axs[0].set_ylim([-5,105])
    axs[1].set_ylim([5.5,9])

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35) 
    plt.figlegend(legend,loc='lower center', ncol=2)
    if graph==True:
        # plt.show()
        pass   
    
    plt.savefig('results/'+filename+'.pgf', format='pgf')

agent_names = ['porto_ete_v5_r_collision_5', 'porto_pete_s_polynomial', 'porto_pete_v_k_1_attempt_2', 'porto_pete_sv_p_r_0']    
parameters = ['mu']
nom_value = [1.0489]
legend_title = ''
legend = ['End-to-end', 'Steering control', 'Velocity control', 'Steering and velocity control']
plot_titles = parameters
graph=True
text=False
filename = 'mu'
# display_lap_mismatch_results_multiple_mu(agent_names, parameters, legend_title, legend, plot_titles, nom_value, graph, text, filename)

# agent_names = ['porto_ete_r_1', 'porto_ete_r_2', 'porto_ete_r_3', 'porto_ete_LiDAR_10', 'porto_ete_r_4' ]
# legend = ['1', '0.7', '0.5', '0.3', '0.1']
# legend_title = 'Distance reward ($r_{\mathrm{dist}}$)'
# ns=[0, 0, 0, 0, 0]
# filename='reward_dist'

# agent_names = ['porto_ete_r_7', 'porto_ete_r_5', 'porto_ete_LiDAR_10', 'porto_ete_r_6']
# legend = ['0', '-0.5', '-1', '$-2']
# legend_title = 'Collision penalty ($r_{\mathrm{collision}}$)'
# ns=[0, 0, 0, 0]
# filename = 'reward_collision'

# agent_names = ['porto_ete_r_0', 'porto_ete_r_1']
# legend = ['0', '-0.01']
# legend_title = 'Time step penalty ($r_{\mathrm{t}}$)'
# ns=[0, 0]
# filename = 'reward_time'

# agent_names = ['porto_ete_LiDAR_3', 'porto_ete_LiDAR_10', 'porto_ete_LiDAR_20', 'porto_ete_LiDAR_50']
# legend = ['3', '10', '20', '50']
# legend_title = 'Number of LiDAR beams'
# ns=[0, 0, 0, 0]
# filename = 'observation_n_beams_1'
# xlim = 5000
# xspace = 2000


# agent_names = ['porto_ete_only_LiDAR', 'porto_ete_LiDAR_20', 'porto_ete_no_LiDAR']
# legend = ['Only LiDAR', 'LiDAR and pose', 'Only pose']
# legend_title = 'Observation space'
# ns=[0, 0, 0]
# filename = 'observation_space_1'
# xlim=6000
# xspace = 2000

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


# agent_names = ['porto_ete_LiDAR_10', 'porto_ete_ddpg']
# legend = ['TD3', 'DDPG']
# legend_title = 'Learning method'
# ns=[0, 0]
# filename = 'learning_method_learning_curve_1'
# xlim=7000
# xspace = 2000

# agent_names = ['porto_ete_v_5']
# legend = ['End-to-end agent']
# legend_title = ''
# ns=[2]
# filename = 'end_to_end_agent_v5'

# agent_names = ['porto_ete_v5_r_dist_1', 'porto_ete_v_5', 'porto_ete_v5_r_dist_2', 'porto_ete_v5_r_dist_4']
# legend = ['0.1', '0.3', '0.5', '1']
# legend_title = 'Distance reward ($r_{\mathrm{dist}}$)'
# ns=[0, 0, 0, 0]
# filename = 'distance_reward_v5_1'
# xlim= 4000
# xspace = 1000

# agent_names = ['porto_ete_v5_r_collision_2', 'porto_ete_v5_r_collision_5']
# legend = ['$r_{\mathrm{collision}}=-4$', '$r_{\mathrm{collision}}=-10$']
# legend_title = ''
# ns=[0, 0]
# filename = 'path_collision_penalty'

# agent_names = ['porto_ete_v5_r_collision_5']
# legend = ['End-to-end agent']
# legend_title = ''
# ns=[0]
# filename = 'path_end_to_end_agent'

# agent_names = ['porto_ete_v5_gamma_0', 'porto_ete_v5_gamma_1', 'porto_ete_v5_r_collision_5', 'porto_ete_v5_gamma_4']
# legend = ['0.9', '0.95', '0.99', '1']
# legend_title = 'Reward discount rate'
# ns=[0, 0, 0, 0]
# filename = 'gamma_learning_curve'

# agent_names = ['porto_ete_v5_alpha_0', 'porto_ete_v5_r_collision_5', 'porto_ete_v5_alpha_1']
# legend = ['$10^{-4}$','$10^{-3}$', '$2 \cdot 10^{-3}$']
# legend_title = 'Learning rate'
# ns=[0, 0, 0]
# filename = 'alpha_learning_curve_1'
# xlim= 3400
# xspace = 1000

# agent_names = ['porto_ete_v5_r_collision_5']
# legend = ['']
# legend_title = ''
# ns=[0]
# filename = 'end_to_end_final_1'
# xlim = 3000
# xspace =1000

# agent_names = ['porto_pete_s_r_collision_0']
# legend = ['']
# legend_title = ''
# ns=[0]
# filename = 'understeer_path'
# xlim = 3000
# xspace =1000


# agent_names = ['porto_ete_v5_r_collision_5', 'porto_pete_s_r_collision_0', 'porto_pete_s_polynomial']
# ns = [0, 0, 0]
# legend = ['End-to-end', 'Circular path', 'Polynomial path']
# legend_title = ''
# filename = 'steer_learning_curve'
# xlim = 3000
# xspace = 1000

# agent_names = ['porto_pete_s_lfc_0', 'porto_pete_s_r_collision_0', 'porto_pete_s_lfc_2']
# legend = ['$L_{c}=0.5$', '$L_{c}=1$', '$L_{c}=2$']
# legend_title = ''
# ns=[0, 0, 0]
# filename = 'lfc_paths'


# agent_names = ['porto_pete_s_lfc_2', 'porto_pete_s_stanley']
# legend = ['Pure pursuit', 'Stanley']
# legend_title = ''
# ns=[0,0]
# filename = 'path_tracker_comparison_learning_curve'
# xlim = 4000
# xspace = 1000


#tune k
# agent_names = [ 'porto_pete_v_r_collision_6_attempt_2',  'porto_pete_v_k_1_attempt_2']
# legend = ['1', '2']
# legend_title = 'attempt'
# ns=[0, 0]


# agent_names = ['porto_pete_v_r_dist_03', 'porto_pete_v_r_collision_6', 'porto_pete_v_k_0',  'porto_pete_v_r_collision_2']
# legend = ['$r_{\mathrm{dist}}=0.3, r_{\mathrm{collision}}=-2$', '$r_{\mathrm{dist}}=0.3, r_{\mathrm{collision}}=-8$', '$r_{\mathrm{dist}}=0.2, r_{\mathrm{collision}}=-2$', '$r_{\mathrm{dist}}=0.2, r_{\mathrm{collision}}=-8$']
# legend_title = 'Reward signal'
# ns=[0,0,0,0]
# filename = 'reward_velocity_controller'


# agent_names = ['porto_pete_v_k_1_attempt_2', 'porto_ete_v5_r_collision_5']
# legend = ['Velocity control', 'End-to-end']
# legend_title = ''
# ns=[0,0]
# filename = 'velocity_control_learning_curves'
# xlim = 5000
# xspace = 1000


# agent_names = ['porto_pete_v_k_0', 'porto_pete_v_r_collision_6']
# legend = ['-2', '-8']
# legend_title = '$r_{\mathrm{collision}}$'
# ns=[0,0]
# filename = 'velocity_reward_collision'

agent_names = ['porto_ete_v5_r_collision_5', 'porto_pete_s_r_collision_0', 'porto_pete_s_polynomial', 
                'porto_pete_v_k_1_attempt_2', 'porto_pete_sv_c_r_8', 'porto_pete_sv_p_r_0']
noise_params = ['xy', 'theta', 'v', 'lidar']
# noise_params = ['xy']
legend_title = 'Agent architecture'
legend = ['End-to-end',
            'Steering control,\ncircular path',
            'Steering control, \npolynomial path',
            'Velocity control',
            'Steering and velocity \ncontrol, circular path',
            'Steering and velocity \ncontrol, polynomial path']
filename='noise_vary'
# display_lap_noise_results_multiple(agent_names, noise_params, legend_title, legend, filename)

# mismatch_parameters = ['C_Sf']
# frac_vary = [0]
# start_condition = {'x':10, 'y':4.5, 'v':3, 'theta':np.pi, 'delta':0, 'goal':0}
# #start_condition = []
# #filename = 'end_to_end_agent'
# display_path_multiple(agent_names=agent_names, ns=ns, legend_title=legend_title,          
#                         legend=legend, mismatch_parameters=mismatch_parameters, frac_vary=frac_vary, 
#                         start_condition=start_condition, filename=filename)

# mismatch_parameters = ['C_Sf']
# frac_vary = [0]
# start_condition = {'x':10, 'y':4.5, 'v':3, 'theta':np.pi, 'delta':0, 'goal':0}
# #start_condition = []
# display_only_path_multiple(agent_names=agent_names, ns=ns, legend_title=legend_title,          
#                         legend=legend, mismatch_parameters=mismatch_parameters, frac_vary=frac_vary, 
#                         start_condition=start_condition, filename=filename)


# mismatch_parameters = ['C_Sf']
# frac_vary = [0]
# start_condition = {'x':10, 'y':4.5, 'v':3, 'theta':np.pi, 'delta':0, 'goal':0}
# #start_condition = []
# display_path_steer_multiple(agent_names=agent_names, ns=ns, legend_title=legend_title,          
#                         legend=legend, mismatch_parameters=mismatch_parameters, frac_vary=frac_vary, 
#                         start_condition=start_condition, filename=filename)

# mismatch_parameters = ['C_Sf']
# frac_vary = [0]
# start_condition = {'x':10, 'y':4.5, 'v':3, 'theta':np.pi, 'delta':0, 'goal':0}
# #start_condition = []
# display_path_actions_multiple(agent_names=agent_names, ns=ns, legend_title=legend_title,          
#                         legend=legend, mismatch_parameters=mismatch_parameters, frac_vary=frac_vary, 
#                         start_condition=start_condition, filename=filename)


# mismatch_parameters = ['C_Sf']
# frac_vary = [0]
# start_condition = {'x':10, 'y':4.5, 'v':3, 'theta':np.pi, 'delta':0, 'goal':0}
# #start_condition = []
# display_path_two_multiple(agent_names=agent_names, ns=ns, legend_title=legend_title,          
#                         legend=legend, mismatch_parameters=mismatch_parameters, frac_vary=frac_vary, 
#                         start_condition=start_condition, filename=filename)

# mismatch_parameters = ['C_Sf']
# frac_vary = [0]
# start_condition = {'x':10, 'y':4.5, 'v':3, 'theta':np.pi, 'delta':0, 'goal':0}
# #start_condition = []
# display_velocity_profile(agent_names=agent_names, ns=ns, legend_title=legend_title,          
#                         legend=legend, mismatch_parameters=mismatch_parameters, frac_vary=frac_vary, 
#                         start_condition=start_condition, filename=filename)


# learning_curve_lap_time_average(agent_names, legend, legend_title, ns, filename, xlim, xspace)

# learning_curve_reward_average(agent_names, legend, legend_title)

# learning_curve_all(agent_names, legend, legend_title, ns, filename, xlim, xspace)

# display_velocity_slip(agent_names, ns, legend_title, legend, mismatch_parameters, frac_vary, start_condition, filename)