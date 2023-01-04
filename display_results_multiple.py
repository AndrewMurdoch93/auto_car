from audioop import avg
from configparser import BasicInterpolation
from re import S
from statistics import median
from threading import local
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
#matplotlib.use('pgf')
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
import os

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
        

def learning_curve_lap_time(agent_names, legend, legend_title, ns):

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

    for i in range(len(agent_names)):
        agent_name = agent_names[i]
        train_results_file_name = 'train_results/' + agent_name
        infile = open(train_results_file_name, 'rb')
        
        _ = pickle.load(infile)
        _ = pickle.load(infile)
        _ = pickle.load(infile)
        steps[i] = pickle.load(infile)
        collisions[i] = pickle.load(infile)
        n_actions[i] = pickle.load(infile)
        
        infile.close()
        
        for j in range(len(collisions[i][ns[i]])):
            if j <= window:
                x = 0
            else:
                x = j-window 
            avg_coll[i].append(np.mean(collisions[i][ns[i]][x:j+1]))
            avg_time[i].append(np.mean(steps[i][ns[i]][x:j+1]))
            std_coll[i].append(np.std(collisions[i][ns[i]][x:j+1]))

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
    
    end_episodes = np.zeros(len(agent_names), int)
    for i in range(len(agent_names)):
        end_episodes[i] =  np.where(steps[i][ns[i]]==0)[0][0]

    plt.figure(1, figsize=(5,4))
    for i in range(len(agent_names)):
        end_episode = end_episodes[i] 
        plt.plot(np.cumsum(steps[i][ns[i]])[0:end_episode],  avg_coll[i][0:end_episode])
        plt.fill_between(x=np.cumsum(steps[i][ns[i]])[0:end_episode], y1=upper_fill_coll[i][0][0:end_episode], y2=lower_fill_coll[i][0][0:end_episode], alpha=0.3, label='_nolegend_')
    
    plt.hlines(y=1, xmin=0, xmax=np.cumsum(steps[0])[np.max(end_episodes)], colors='black', linestyle='dashed')
    plt.hlines(y=0, xmin=0, xmax=np.cumsum(steps[0])[np.max(end_episodes)], colors='black', linestyle='dashed')
    plt.xlabel('Simulation steps')
    #plt.title('Collision rate')
    plt.ylabel('Collision rate')
    plt.legend(legend_coll, title=legend_title, loc='upper right')
    #plt.xlim([0,6000])
    plt.ylim([-0.05, 1.05])
    plt.grid(True)
    plt.rc('axes',edgecolor='gray')
    plt.tick_params(axis=u'both', which=u'both',length=0)

    plt.figure(2, figsize=(5,4))
    for i in range(len(agent_names)):
        end_episode_no_coll = np.where(steps_no_coll[i]==0)[0][0]
        #plt.plot(np.cumsum(steps[0]), steps[0], 'x')
        #plt.plot(steps[0][np.where(np.logical_not(collisions[0]))], 'x')
        plt.plot(steps_x_axis[i][0:end_episode_no_coll],   np.array(avg_steps_no_coll[i][0:end_episode_no_coll])*0.01 )
        plt.fill_between(x=steps_x_axis[i][0:end_episode_no_coll], y1=upper_fill_steps_no_coll[i][0][0:end_episode_no_coll]*0.01 , y2=lower_fill_steps_no_coll[i][0][0:end_episode_no_coll]*0.01, alpha=0.3, label='_nolegend_')
    
    
    plt.xlabel('Simulation steps')
    #plt.title('Lap time')
    plt.ylabel('Lap time [s]')
    plt.legend(legend, title=legend_title, loc='upper right')
    #plt.xlim([0,6000])
    plt.grid(True)
    plt.rc('axes',edgecolor='gray')
    plt.tick_params(axis=u'both', which=u'both',length=0)

    
    plt.figure(3, figsize=(5,4))
    for i in range(len(agent_names)):
        end_episode = end_episodes[i] 
        #plt.plot(np.cumsum(steps[0]), steps[0], 'x')
        #plt.plot(steps[0][np.where(np.logical_not(collisions[0]))], 'x')
        plt.plot(avg_coll[i][0:end_episode])
        #plt.plot(var_coll[i][0:end_episode])
        plt.fill_between(x=np.arange(end_episode), y1=upper_fill_coll[i][0][0:end_episode], y2=lower_fill_coll[i][0][0:end_episode], alpha=0.3, label='_nolegend_')
    
    plt.hlines(y=1, xmin=0, xmax=np.max(end_episodes), colors='black', linestyle='dashed')
    plt.hlines(y=0, xmin=0, xmax=np.max(end_episodes), colors='black', linestyle='dashed')
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Episodes')
    #plt.title('Collision rate')
    plt.ylabel('Collision rate')
    plt.legend(legend_coll, title=legend_title, loc='upper right')
    plt.grid(True)
    plt.rc('axes',edgecolor='gray')
    plt.tick_params(axis=u'both', which=u'both',length=0)
    #plt.xlim([0,6000])

    plt.figure(4, figsize=(5,4))
    for i in range(len(agent_names)):
        end_episode_no_coll = np.where(steps_no_coll[i]==0)[0][0]
        #plt.plot(np.cumsum(steps[0]), steps[0], 'x')
        #plt.plot(steps[0][np.where(np.logical_not(collisions[0]))], 'x')
        plt.plot(np.arange(end_episodes[i])[np.logical_not(collisions[i][ns[i]])[0:end_episodes[i]]], np.array(avg_steps_no_coll[i][0:end_episode_no_coll])*0.01)
        plt.fill_between(x=np.arange(end_episodes[i])[np.logical_not(collisions[i][ns[i]])[0:end_episodes[i]]], y1=upper_fill_steps_no_coll[i][0][0:end_episode_no_coll]*0.01 , y2=lower_fill_steps_no_coll[i][0][0:end_episode_no_coll]*0.01, alpha=0.3, label='_nolegend_')

        np.arange(len(steps[i][ns[i]]))[np.logical_not(collisions[i][ns[i]])][0:end_episodes[i]]
        #plt.plot(np.array(max_steps_no_coll[i][0:end_episode_no_coll])*0.01 )
    plt.xlabel('Episodes')
    #plt.title('Average time per episode without collisions')
    plt.ylabel('Lap time [s]')
    plt.legend(legend, title=legend_title, loc='upper right')
    plt.grid(True)
    plt.rc('axes',edgecolor='gray')
    plt.tick_params(axis=u'both', which=u'both',length=0)
        #plt.xlim([0,6000])


    plt.figure(5, figsize=(5,4))
    for i in range(len(agent_names)):
        end_episode = end_episodes[i] 
        plt.plot(np.cumsum(n_actions[i][ns[i]])[0:end_episode],  avg_coll[i][0:end_episode])
        plt.fill_between(x=np.cumsum(n_actions[i][ns[i]])[0:end_episode], y1=upper_fill_coll[i][0][0:end_episode], y2=lower_fill_coll[i][0][0:end_episode], alpha=0.3, label='_nolegend_')
    
    plt.hlines(y=1, xmin=0, xmax=np.cumsum(n_actions[i][0])[np.max(end_episodes)], colors='black', linestyle='dashed')
    plt.hlines(y=0, xmin=0, xmax=np.cumsum(n_actions[i][0])[np.max(end_episodes)], colors='black', linestyle='dashed')
    plt.xlabel('Steps')
    #plt.title('Collision rate')
    plt.ylabel('Collision rate')
    plt.legend(legend_coll, title=legend_title, loc='upper right')
    #plt.xlim([0,6000])
    plt.ylim([-0.05, 1.05])
    plt.grid(True)
    plt.rc('axes',edgecolor='gray')
    plt.tick_params(axis=u'both', which=u'both',length=0)

    plt.figure(6, figsize=(5,4))
    for i in range(len(agent_names)):
        end_episode_no_coll = np.where(steps_no_coll[i]==0)[0][0]
        plt.plot(n_actions_x_axis[i][0:end_episode_no_coll],   np.array(avg_steps_no_coll[i][0:end_episode_no_coll])*0.01 )
        plt.fill_between(x=n_actions_x_axis[i][0:end_episode_no_coll], y1=upper_fill_steps_no_coll[i][0][0:end_episode_no_coll]*0.01 , y2=lower_fill_steps_no_coll[i][0][0:end_episode_no_coll]*0.01, alpha=0.3, label='_nolegend_')
    
    
    plt.xlabel('Steps')
    #plt.title('Lap time')
    plt.ylabel('Lap time [s]')
    plt.legend(legend, title=legend_title, loc='upper right')
    #plt.xlim([0,6000])
    plt.grid(True)
    plt.rc('axes',edgecolor='gray')
    plt.tick_params(axis=u'both', which=u'both',length=0)

    plt.show()


def learning_curve_lap_time_average(agent_names, legend, legend_title, ns):

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


    plt.figure(1, figsize=(5,4))
    plt.rc('axes',edgecolor='gray')
    #plt.rc('font', **font)

    for i in range(len(agent_names)):
        end_episode = end_episodes[i] 
        plt.plot(np.cumsum(steps_avg_x_axis[i])[0:end_episode],  avg_coll[i][0:end_episode])
        plt.fill_between(x=np.cumsum(steps_avg_x_axis[i])[0:end_episode], y1=upper_fill_coll[i][0][0:end_episode], y2=lower_fill_coll[i][0][0:end_episode], alpha=0.3, label='_nolegend_')
    
    plt.hlines(y=1, xmin=0, xmax=np.cumsum(steps_avg_x_axis[i])[np.max(end_episodes)], colors='black', linestyle='dashed')
    plt.hlines(y=0, xmin=0, xmax=np.cumsum(steps_avg_x_axis[i])[np.max(end_episodes)], colors='black', linestyle='dashed')
    plt.xlabel('Simulation steps')
    #plt.title('Collision rate')
    plt.ylabel('Collision rate')
    plt.legend(legend_coll, title=legend_title, loc='upper right')
    #plt.xlim([0,6000])
    plt.ylim([-0.05, 1.05])
    plt.grid(True)
    plt.rc('axes',edgecolor='gray')
    plt.tick_params(axis=u'both', which=u'both',length=0)

    plt.savefig('collision_rate.pgf', format='pgf')

    plt.figure(2, figsize=(5,4))

    for i in range(len(agent_names)):
        end_episode = end_episodes[i] 
        #plt.plot(np.cumsum(steps[0]), steps[0], 'x')
        #plt.plot(steps[0][np.where(np.logical_not(collisions[0]))], 'x')
        plt.plot(np.cumsum(steps_avg_x_axis[i])[0:end_episode],   np.array(steps_y_avg_smoothed[i][0:end_episode])*0.01)
        plt.fill_between(x=np.cumsum(steps_avg_x_axis[i])[0:end_episode], y1=upper_fill[i][0][0:end_episode]*0.01, y2=lower_fill[i][0][0:end_episode]*0.01, alpha=0.3, label='_nolegend_')
    
    plt.xlabel('Simulation steps')
    #plt.title('Lap time')
    plt.ylabel('Lap time [s]')
    plt.legend(legend, title=legend_title, loc='upper right')
    #plt.xlim([0,6000])
    plt.grid(True)
    plt.rc('axes',edgecolor='gray')
    plt.tick_params(axis=u'both', which=u'both',length=0)

    

    plt.figure(3, figsize=(5,4))
    for i in range(len(agent_names)):
        end_episode = end_episodes[i] 
        plt.plot(avg_coll[i][0:end_episode])
        plt.fill_between(x=np.arange(end_episode), y1=upper_fill_coll[i][0][0:end_episode], y2=lower_fill_coll[i][0][0:end_episode], alpha=0.3, label='_nolegend_')
    
    plt.hlines(y=1, xmin=0, xmax=np.max(end_episodes), colors='black', linestyle='dashed')
    plt.hlines(y=0, xmin=0, xmax=np.max(end_episodes), colors='black', linestyle='dashed')
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Episodes')
    #plt.title('Collision rate')
    plt.ylabel('Collision rate')
    plt.legend(legend_coll, title=legend_title, loc='upper right')
    plt.grid(True)
    plt.rc('axes',edgecolor='gray')
    plt.tick_params(axis=u'both', which=u'both',length=0)



    plt.figure(4, figsize=(5,4))

    for i in range(np.size(end_ep, axis=0)):
        end_episode = end_episodes[i] 
        #plt.plot(np.cumsum(steps[0]), steps[0], 'x')
        #plt.plot(steps[0][np.where(np.logical_not(collisions[0]))], 'x')
        plt.plot(np.arange(end_episode), np.array(steps_y_avg_smoothed[i][0:end_episode])*0.01)
        plt.fill_between(x=np.arange(end_episode), y1=upper_fill[i][0][0:end_episode]*0.01, y2=lower_fill[i][0][0:end_episode]*0.01, alpha=0.3, label='_nolegend_')

        #np.arange(len(steps[i][ns[i]]))[np.logical_not(collisions[i][ns[i]])][0:end_episodes[i]]
        #plt.plot(np.array(max_steps_no_coll[i][0:end_episode_no_coll])*0.01 )
    plt.xlabel('Episodes')
    #plt.title('Average time per episode without collisions')
    plt.ylabel('Lap time [s]')
    plt.legend(legend, title=legend_title, loc='upper right')
    plt.grid(True)
    plt.rc('axes',edgecolor='gray')
    plt.tick_params(axis=u'both', which=u'both',length=0)
        #plt.xlim([0,6000])


    plt.figure(5, figsize=(5,4))
    for i in range(len(agent_names)):
        end_episode = end_episodes[i] 
        plt.plot(np.cumsum(avg_n_actions[i])[0:end_episode],  avg_coll[i][0:end_episode])
        plt.fill_between(x=np.cumsum(avg_n_actions[i])[0:end_episode], y1=upper_fill_coll[i][0][0:end_episode], y2=lower_fill_coll[i][0][0:end_episode], alpha=0.3, label='_nolegend_')
    
    plt.hlines(y=1, xmin=0, xmax=np.cumsum(avg_n_actions[i])[np.max(end_episodes)], colors='black', linestyle='dashed')
    plt.hlines(y=0, xmin=0, xmax=np.cumsum(avg_n_actions[i])[np.max(end_episodes)], colors='black', linestyle='dashed')
    plt.xlabel('Steps')
    #plt.title('Collision rate')
    plt.ylabel('Collision rate')
    plt.legend(legend_coll, title=legend_title, loc='upper right')
    #plt.xlim([0,6000])
    plt.ylim([-0.05, 1.05])
    plt.grid(True)
    plt.rc('axes',edgecolor='gray')
    plt.tick_params(axis=u'both', which=u'both',length=0)

    plt.figure(6, figsize=(5,4))
    for i in range(len(agent_names)):
        end_episode = end_episodes[i]
        plt.plot(np.cumsum(avg_n_actions[i][0:end_episode]), np.array(steps_y_avg_smoothed[i][0:end_episode])*0.01)
        plt.fill_between(x=np.cumsum(avg_n_actions[i][0:end_episode]), y1=upper_fill[i][0][0:end_episode]*0.01, y2=lower_fill[i][0][0:end_episode]*0.01, alpha=0.3, label='_nolegend_')
    
    
    plt.xlabel('Steps')
    #plt.title('Lap time')
    plt.ylabel('Lap time [s]')
    plt.legend(legend, title=legend_title, loc='upper right')
    #plt.xlim([0,6000])
    plt.grid(True)
    plt.rc('axes',edgecolor='gray')
    plt.tick_params(axis=u'both', which=u'both',length=0)

    plt.show()


def learning_curve_reward(agent_names, legend, legend_title, ns):
    
    legend_new = legend.copy()
    legend_new.append('Min and max')
    window = 500
    
    steps = [[] for _ in range(len(agent_names))]
    steps_x_axis = [[] for _ in range(len(agent_names))]    
    scores = [[] for _ in range(len(agent_names))]    
    avg_score = [[] for _ in range(len(agent_names))]
    std_score = [[] for _ in range(len(agent_names))]
    upper_fill = [[] for _ in range(len(agent_names))]
    lower_fill = [[] for _ in range(len(agent_names))]
    n_actions = [[] for _ in range(len(agent_names))]

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
        
        for j in range(len(scores[i][ns[i]])):
            if j <= window:
                x = 0
            else:
                x = j-window 
            avg_score[i].append(np.mean(scores[i][ns[i]][x:j+1]))
            std_score[i].append(np.std(scores[i][ns[i]][x:j+1]))

        upper_fill[i].append(np.array(avg_score[i])+np.array(std_score[i]))
        lower_fill[i].append(np.array(avg_score[i])-np.array(std_score[i]))
    
    end_episodes = np.zeros(len(agent_names), int)
    for i in range(len(agent_names)):
        end_episodes[i] =  np.where(steps[i][ns[i]]==0)[0][0]

    
    plt.figure(1, figsize=(5,4))
    plt.rc('axes',edgecolor='gray')
    for i in range(len(agent_names)):
        end_episode = end_episodes[i] 
        plt.plot(np.cumsum(steps[i][ns])[0:end_episode],  avg_score[i][0:end_episode])
        plt.fill_between(x=np.cumsum(steps[i][ns])[0:end_episode], y1=upper_fill[i][0][0:end_episode], y2=lower_fill[i][0][0:end_episode], alpha=0.3, label='_nolegend_')
    
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
        plt.plot(np.cumsum(n_actions[i][ns])[0:end_episode],  avg_score[i][0:end_episode])
        plt.fill_between(x=np.cumsum(n_actions[i][ns])[0:end_episode], y1=upper_fill[i][0][0:end_episode], y2=lower_fill[i][0][0:end_episode], alpha=0.3, label='_nolegend_')
    
    plt.xlabel('Steps')
    plt.ylabel('Episode reward')
    plt.legend(legend_new, title=legend_title, loc='upper right')
    #plt.xlim([0,6000])
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
    plt.legend(legend_new, title=legend_title, loc='upper right')
    #plt.xlim([0,6000])
    plt.grid(True)
    plt.tick_params(axis=u'both', which=u'both',length=0)

    plt.show()


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
    #plt.xlim([0,6000])
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
    #plt.xlim([0,6000])
    plt.grid(True)
    plt.tick_params(axis=u'both', which=u'both',length=0)

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

    path = 'lap_results/' + agent_name
    if os.path.exists(path):
        infile = open(path, 'rb')
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
        perc_failure = 1-perc_success
        std_dev = np.std(times[np.logical_not(np.array(collisions))])

        
        print('\nLap results over all n, without noise:')
        print('Average lap time: ', ave_times)
        print('Lap time std deviation:', std_dev)
        print('Success %: ', perc_success*100)
        print('Failure %: ', perc_failure*100)
        
        print("\nAgent lap statistics for individual runs without noise:")

        print(f"{'n':3s}{'| fraction success':20s}{'|avg lap time':12s}")
        for n in range(len(times[:,0])):
            avg_time = np.average(times[n, np.logical_not(np.array(collisions[n]))])
            frac_succ =  np.sum(np.logical_not(np.array(collisions[n])))/len(np.array(collisions[n]))
            print(f"{n:3d}", end='')
            print(f"{frac_succ:20.3f}{avg_time:12.3f}")
        
    
    path = 'lap_results_with_noise/' + agent_name
    if os.path.exists(path):
        infile = open(path, 'rb')
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
        perc_failure = 1-perc_success
        std_dev = np.std(times[np.logical_not(np.array(collisions))])

        
        print('\nLap results over all n, with noise:')
        print('Average lap time: ', ave_times)
        print('Lap time std deviation:', std_dev)
        print('Success %: ', perc_success*100)
        print('Failure %: ', perc_failure*100)
        
        print("\nAgent lap statistics for individual runs with noise:")

        print(f"{'n':3s}{'| fraction success':20s}{'|avg lap time':12s}")
        for n in range(len(times[:,0])):
            avg_time = np.average(times[n, np.logical_not(np.array(collisions[n]))])
            frac_succ =  np.sum(np.logical_not(np.array(collisions[n])))/len(np.array(collisions[n]))
            print(f"{n:3d}", end='')
            print(f"{frac_succ:20.3f}{avg_time:12.3f}")





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
    # plt.show()

    # p = sns.barplot(x='map', y='success', hue='architecture', capsize=.2, data=df_success)
    # p.set_xlabel('% variation from original ' + param + ' value')
    # p.set_ylabel('fraction successful laps')
    # handles, _ = p.get_legend_handles_labels()
    # p.legend(handles, legend ,title=legend_title, loc='lower left')
    # plt.title('Lap success rate for ' + param + ' mismatch')
    # plt.show()

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

# graph_lap_results(agent_names)


#graphs mistmatch results for all tracks
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


# graph_lap_results_mismatch(agent_names, 'C_Sf', title='Front tyre stiffness coefficient 20% higher than expected')




def display_lap_mismatch_results_multiple(agent_names, parameters, legend_title, legend, plot_titles):
    
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
            # axs[j].set_ylim([0.9,1])
            axs[j].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            axs[j].plot(results_dict['frac_variation']*100, avg)
            #plt.fill_between(results_dict['frac_variation']*100, avg-dev, avg+dev, alpha=0.25)
            axs[j].set(ylabel='fraction successful laps')
            #axs.yaxis.set_major_formatter(plt.ticker.FormatStrFormatter('%.2f'))

        axs[j].set_title('(' + numbering[j] + ') ' + plot_titles[j])
    axs[j].set(xlabel='% variation from original value')
    axs[j].legend(legend, title=legend_title, loc='lower right')
    plt.show()


   

# agent_names = ['porto_pete_sv', 'porto_pete_s', 'porto_pete_v', 'porto_ete']
# agent_names = ['porto_ete_v5_r_collision_5', 'porto_pete_s_r_collision_0', 'porto_pete_s_polynomial', 'porto_pete_v_k_1_attempt_2', 'porto_pete_sv_c_r_8', 'porto_pete_sv_p_r_0']    
agent_names = ['porto_ete_v5_r_collision_5', 'porto_pete_s_polynomial', 'porto_pete_v_k_1_attempt_2', 'porto_pete_sv_p_r_0']    

# agent_names = ['porto_pete_s_r_collision_0']
# agent_names = ['porto_pete_s_polynomial']   
# agent_names = ['porto_pete_v_k_1_attempt_2']
# agent_names = ['porto_pete_sv_c_r_8']
# agent_names = ['porto_pete_sv_p_r_0']
# parameters = ['mu', 'C_S']
# parameters = ['lf', 'h', 'm', 'I']
parameters = ['sv', 'a_max']
legend_title = ''
legend = agent_names
plot_titles = parameters
# display_lap_mismatch_results_multiple(agent_names, parameters, legend_title, legend, plot_titles)


def display_lap_mismatch_results_multiple_1(agent_names, parameters, legend_title, legend, plot_titles, nom_value, graph, text):
    
    
    fig, axs = plt.subplots(nrows=len(parameters), ncols=2)

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
                axs[j,0].plot(nom_value[j]*(1+results_dict['frac_variation']), avg_col_filter*100, alpha=0.8)
                axs[j,0].fill_between(nom_value[j]*(1+results_dict['frac_variation']), (avg_col_filter+dev_col_filter)*100, (avg_col_filter-dev_col_filter)*100, alpha=0.2, label='_nolegend_')
            else:
                axs[j,0].plot(nom_value[j]*(1+results_dict['frac_variation']), avg_col*100, alpha=0.8)
                axs[j,0].fill_between(nom_value[j]*(1+results_dict['frac_variation']), (avg_col+dev_col)*100, (avg_col-dev_col)*100, alpha=0.2, label='_nolegend_')
            
            # plot lap times
            axs[j,1].plot(nom_value[j]*(1+results_dict['frac_variation']), avg_times, alpha=0.8)
            axs[j,1].fill_between(nom_value[j]*(1+results_dict['frac_variation']), (avg_times+dev_times), (avg_times-dev_times), alpha=0.2, label='_nolegend_')

        axs[j,0].grid(True)
        axs[j,1].grid(True)
            
        axs[j,0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        axs[j,1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axs[j,0].tick_params('both', length=0)
        axs[j,1].tick_params('both', length=0)
        color='grey'
        axs[j,0].spines['bottom'].set_color(color)
        axs[j,0].spines['top'].set_color(color) 
        axs[j,0].spines['right'].set_color(color)
        axs[j,0].spines['left'].set_color(color)
        axs[j,1].spines['bottom'].set_color(color)
        axs[j,1].spines['top'].set_color(color) 
        axs[j,1].spines['right'].set_color(color)
        axs[j,1].spines['left'].set_color(color)


        axs[j,0].set_title('                                                           '+parameter, fontsize=12)
        
        axs[j,0].set(ylabel='Successful laps [%]') 
        axs[j,1].set(ylabel='Times [s]') 
        
        xlabels = ['Deviation from mean', 'Deviation from mean']
        axs[j,0].set(xlabel=xlabels[j]) 
        axs[j,1].set(xlabel=xlabels[j])
        
        
        # axs[j,0].set_xticks(ticks=, labels=)
        # axs[j,1].set_xticks(ticks=, labels=)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22) 
    plt.figlegend(legend,loc='lower center', ncol=2)
    if graph==True:
        plt.show()
    
    
def display_lap_mismatch_results_multiple_2(agent_names, parameters, legend_title, legend, plot_titles, nom_value, graph, text):
    
    
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
        axs[1].set(ylabel='Times [s]') 
        
        xlabels = ['Deviation from mean', 'Deviation from mean']
        axs[0].set(xlabel=xlabels[j]) 
        axs[1].set(xlabel=xlabels[j])
        
        axs[0].set_xlim([0.2,1.2])
        axs[1].set_xlim([0.2,1.2])
        # axs[j,0].set_xticks(ticks=, labels=)
        # axs[j,1].set_xticks(ticks=, labels=)

    axs[0].vlines(x=nom_value, ymin=0, ymax=100, color='black', linestyle='--')
    axs[0].vlines(x=0.7, ymin=0, ymax=100, color='black', linestyle='--')
    axs[0].vlines(x=0.5, ymin=0, ymax=100, color='black', linestyle='--')
    axs[0].text(x=0.75, y=40, s='Nominal value', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))
    axs[0].text(x=0.6, y=25, s='Dry asphalt', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))
    axs[0].text(x=0.4, y=10, s='Wet asphalt', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))
    
    axs[1].vlines(x=nom_value, ymin=5.5, ymax=9, color='black', linestyle='--')
    axs[1].vlines(x=0.7, ymin=5.5, ymax=9, color='black', linestyle='--')
    axs[1].vlines(x=0.5, ymin=5.5, ymax=9, color='black', linestyle='--')
    axs[1].text(x=0.75, y=7.3, s='Nominal value', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))
    axs[1].text(x=0.6, y=7.9, s='Dry asphalt', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))
    axs[1].text(x=0.4, y=8.5, s='Wet asphalt', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))


    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35) 
    plt.figlegend(legend,loc='lower center', ncol=2)
    if graph==True:
        plt.show()   


   

agent_names = ['porto_ete_v5_r_collision_5', 'porto_pete_s_polynomial', 'porto_pete_v_k_1_attempt_2', 'porto_pete_sv_p_r_0']    
# parameters = ['mu', 'C_S']
# nom_value = [1.0489, 1]
parameters = ['C_Sf', 'C_Sr']
nom_value = [4.718, 5.4562]
# parameters = ['lf', 'h', 'm', 'I']
# parameters = ['sv', 'a_max']
# nom_value = [3.2, 9.51]
# parameters = ['mu']
# nom_value = [1.0489]

legend_title = ''
legend = ['End-to-end', 'Steering control', 'Velocity control', 'Steering and velocity control']
plot_titles = parameters
graph=True
text=False
# display_lap_mismatch_results_multiple_1(agent_names, parameters, legend_title, legend, plot_titles, nom_value, graph, text)
#mu
# display_lap_mismatch_results_multiple_2(agent_names, parameters, legend_title, legend, plot_titles, nom_value, graph, text)




def display_lap_unknown_mass(agent_names, legend):
    
    # plt.rcParams.update({
    # "font.family": "serif",  # use serif/main font for text elements
    # "text.usetex": True,     # use inline math for ticks
    # "pgf.rcfonts": False,     # don't setup fonts from rc parameters
    # "font.size": 12
    # })

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
    plt.show()



def display_lap_unknown_mass_time(agent_names, legend):
    
    # plt.rcParams.update({
    # "font.family": "serif",  # use serif/main font for text elements
    # "text.usetex": True,     # use inline math for ticks
    # "pgf.rcfonts": False,     # don't setup fonts from rc parameters
    # "font.size": 12
    # })

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(5.5,3))

  
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
    axs[0].text(x=0.08, y=90.5, s='Front axle', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))

    axs[1].vlines(x=0,ymin=5.8,ymax=6.3,color='black',linestyle='--', label='_nolegend_', alpha=0.8)
    axs[1].vlines(x=0.33,ymin=5.8,ymax=6.3,color='black',linestyle='--', alpha=0.8)
    axs[1].text(x=0.37, y=6.25, s='Rear axle', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))
    axs[1].text(x=0.08, y=6.25, s='Front axle', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))



    axs[0].set(ylabel='Successful laps [%]') 
    axs[1].set(ylabel='Times [s]') 
    
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
    fig.subplots_adjust(bottom=0.42) 
    plt.figlegend(legend,loc='lower center', ncol=2)
    plt.show()





# agent_names = ['porto_ete_v5_r_collision_5', 'porto_pete_s_r_collision_0', 'porto_pete_s_polynomial', 'porto_pete_v_k_1_attempt_2', 'porto_pete_sv_c_r_8', 'porto_pete_sv_p_r_0']    
agent_names = ['porto_ete_v5_r_collision_5', 'porto_pete_s_polynomial', 'porto_pete_v_k_1_attempt_2', 'porto_pete_sv_p_r_0']    

# agent_names = ['porto_pete_s_r_collision_0']
# agent_names = ['porto_pete_s_polynomial']   
# agent_names = ['porto_pete_v_k_1_attempt_2']
# agent_names = ['porto_pete_sv_c_r_8']
# agent_names = ['porto_ete_v5_r_collision_5', 'porto_pete_sv_p_r_0']
legend = ['End-to-end', 'Steering control', 'Velocity control', 'Steering and velocity control']
# display_lap_unknown_mass_time(agent_names, legend)



def display_lap_noise_results_multiple(agent_names, noise_params, legend_title, legend):
    
    
    fig, axs = plt.subplots(len(noise_params), 2, figsize=(5.5,8))
    numbering = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']

    for j, parameter in enumerate(noise_params):
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
            axs[j,0].set_ylim([0,101])
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
    
    axs[0,0].set_title('$x$ and $y$ coordinates')
    axs[1,0].set_title('Vehicle heading')
    axs[2,0].set_title('Velocity')
    axs[3,0].set_title('LiDAR')

    
    axs[0,0].set_xlabel('m')
    axs[0,1].set_xlabel('m')
    axs[1,0].set_xlabel('rads')
    axs[1,1].set_xlabel('rads')
    axs[2,0].set_xlabel('m/s')
    axs[2,1].set_xlabel('m/s')
    axs[3,0].set_xlabel('m')
    axs[3,1].set_xlabel('m')

    axs[0,0].set_xlim([0,0.3])
    axs[1,0].set_xlim([0,0.4])
    axs[2,0].set_xlim([0,1])
    axs[3,0].set_xlim([0,0.4])

    axs[0,1].set_xlim([0,0.3])
    axs[1,1].set_xlim([0,0.4])
    axs[2,1].set_xlim([0,1])
    axs[3,1].set_xlim([0,0.4])

    axs[0,0].set_ylim([50,105])
    axs[1,0].set_ylim([50,105])
    axs[2,0].set_ylim([50,105])
    axs[3,0].set_ylim([50,105])

    axs[0,1].set_ylim([5,7])
    # axs[1,1].set_ylim([50,105])
    # axs[2,1].set_ylim([50,105])
    # axs[3,1].set_ylim([50,105])



    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22) 
    plt.figlegend(legend, title=legend_title, loc = 'lower center', ncol=2)
    plt.show()


def display_lap_noise_results_single(agent_names, noise_param, legend_title, legend):
    
    
    fig, axs = plt.subplots(1, 2, figsize=(5.5,4))
    numbering = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']

    parameter=noise_param[0]
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

        axs[0].grid(True)
        axs[0].set_ylim([0,101])
        axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        axs[0].tick_params('both', length=0)
        axs[0].plot(results_dict['noise_std_values'], avg_cols_filter)
        # axs[j].fill_between(results_dict['noise_std_values'], avg-dev, avg+dev, alpha=0.25)
        axs[0].set(ylabel='Successful\nlaps [%]')
        # axs.yaxis.set_major_formatter(plt.ticker.FormatStrFormatter('%.2f'))
        

        axs[1].grid(True)
        # axs[j,1].set_ylim([5,7])
        axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axs[1].tick_params('both', length=0)
        axs[1].plot(results_dict['noise_std_values'], avg_times_filter)
        # axs[j].fill_between(results_dict['noise_std_values'], avg-dev, avg+dev, alpha=0.25)
        axs[1].set(ylabel='Lap time [s]')

    #axs[j].set_title('(' + numbering[j] + ') ' + plot_titles[j])
    # axs[j].set(xlabel='standard deviation')
    # axs[j].legend(legend, title=legend_title, loc='lower right')

    axs[0].set_title('$x$ and $y$ coordinates')
    axs[0].set_xlabel('m')
    axs[1].set_xlabel('m')
    
    axs[0].set_xlim([0,0.3])
    axs[1].set_xlim([0,0.3])

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.4) 
    plt.figlegend(legend, title=legend_title, loc = 'lower center', ncol=2)
    plt.show()



def sensitivity_analysis(agent_names, ns, legend_title, legend, mismatch_parameters, frac_vary, noise_dicts, start_condition):
    
    pose_history = []
    progress_history = []
    state_history = []
    local_path_history = []
    action_step_history = []
    
    init_noise_dict = {'xy':0, 'theta':0, 'v':0, 'lidar':0}

    for agent_name, n, i in zip(agent_names, ns, range(len(agent_names))):

        infile = open('environments/' + agent_name, 'rb')
        env_dict = pickle.load(infile)
        infile.close()
        # Compensate for changes to reward structure
        env_dict['reward_signal']['max_progress'] = 0
        
        # Model mismatches
        if mismatch_parameters:
            for par, var in zip(mismatch_parameters[i], frac_vary[i]):
                env_dict['car_params'][par] *= 1+var 

        noise_dict = noise_dicts[i]

        env = environment(env_dict)
        if start_condition:
            env.reset(save_history=True, start_condition=start_condition, car_params=env_dict['car_params'], noise=init_noise_dict)
        else:
            env.reset(save_history=True, start_condition=[], car_params=env_dict['car_params'], noise=init_noise_dict)

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
        env.reset(save_history=True, start_condition=start_condition, car_params=env_dict['car_params'], noise=init_noise_dict)
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
        

    myfont = {'fontname':'serif'}
    figure_size = (5.5,3)
    xlims = [0,100]

    legend_new = legend.copy()
    legend_new.insert(0, 'Min and max')

    legend_racetrack = legend.copy()
    legend_racetrack.insert(0, 'Track centerline')



    plt.figure(1, figsize=figure_size)
    ax = plt.subplot(111)
    ax.axis('off')
    #plt.rc('axes',edgecolor='lightgrey')
    
    #ax.tick_params(axis='both', colors='lightgrey')
    ax.spines['bottom'].set_color('lightgrey')
    ax.spines['top'].set_color('lightgrey') 
    ax.spines['right'].set_color('lightgrey')
    ax.spines['left'].set_color('lightgrey')

    ax.tick_params(axis=u'both', which=u'both',length=0)
    
    track = mapping.map(env.map_name)
    ax.imshow(ImageOps.invert(track.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(1))), extent=(0,track.map_width,0,track.map_height), cmap="gray")
    # ax.plot(env.rx, env.ry, color='gray', linestyle='dashed')
    alpha=0.7

    for i in range(len(agent_names)):
   
        if env_dict['steer_control_dict']['steering_control']:
            for j in np.array(local_path_history[i])[np.arange(0,len(local_path_history[i]),20)]:
                ax.plot(j[0], j[1], alpha=0.5, linestyle='dashdot', color='red')
                ax.plot(j[0][0], j[1][0], alpha=0.5, color='red', marker='s')

        ax.plot(np.array(state_history[i])[:,0], np.array(state_history[i])[:,1], linewidth=1.5, alpha=alpha, label='Path without noise')   
        
        #sample points
        idx = np.arange(0,len(state_history[i]),40)
        idx = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
        sample_states = np.array(state_history[i])[idx]
        sample_actions = np.array(action_step_history[i])[idx]
        
        ax.plot(sample_states[:,0], sample_states[:,1], 'x', color='orange', alpha=alpha, label='_nolabel_')  
        
        arrow_length = 0.1
        arrow_size=0.3
        c1=0
        c2=0
        for state,action in zip(sample_states,sample_actions):
            if c1==0:
                plt.arrow(state[0], state[1], arrow_length*math.cos(state[4]), arrow_length*math.sin(state[4]), head_length=arrow_size, head_width=arrow_size, shape='full', ec='None', fc='blue', alpha=0.5, label='Vehicle heading')
                plt.arrow(state[0], state[1], arrow_length*math.cos(state[4]+action[0]*0.4), arrow_length*math.sin(state[4]+action[0]*0.4), head_length=arrow_size, head_width=arrow_size, shape='full',ec='None', fc='red', alpha=0.5, label='Desired steering\nangle command')
                c1+=1
            if c1>0:
                plt.arrow(state[0], state[1], arrow_length*math.cos(state[4]), arrow_length*math.sin(state[4]), head_length=arrow_size, head_width=arrow_size, shape='full', ec='None', fc='blue', alpha=0.5, label='_nolegend_')
                plt.arrow(state[0], state[1], arrow_length*math.cos(state[4]+action[0]*0.4), arrow_length*math.sin(state[4]+action[0]*0.4), head_length=arrow_size, head_width=arrow_size, shape='full',ec='None', fc='red', alpha=0.5, label='_nolegend_')
                

            for i in range(5):
                noisy_x = state[0]+np.random.normal(loc=0, scale=noise_dict['xy'])
                noisy_y = state[1]+np.random.normal(loc=0, scale=noise_dict['xy'])
                noisy_theta = state[4]%(2*np.pi)+np.random.normal(loc=0, scale=noise_dict['theta'])
                noisy_v = state[3]+np.random.normal(loc=0, scale=noise_dict['v'])
                lidar_dists, lidar_coords = env.lidar.get_scan(noisy_x, noisy_y, noisy_theta)
                x_norm = (noisy_x)/env.map_width
                y_norm = (noisy_y)/env.map_height
                theta_norm = (noisy_theta)/(2*math.pi)
                v_norm = (noisy_v)/env.params['v_max']
                lidar_norm = np.array(lidar_dists+np.random.normal(loc=0, scale=noise_dict['lidar'], size=env.lidar_dict['n_beams']))/env.lidar_dict['max_range']
                observation = [x_norm, y_norm, theta_norm, v_norm]
                for n in lidar_norm:
                    observation.append(n)
                action=a.choose_greedy_action(observation)
                if c2==0:
                    ax.plot(noisy_x, noisy_y, 'x', color='orange', alpha=alpha, label='Vehicle position after\nnoise is added')
                    c2+=1
                if c2>0:
                    ax.plot(noisy_x, noisy_y, 'x', color='orange', alpha=alpha, label='_nolabel_')
                
                plt.arrow(noisy_x, noisy_y, arrow_length*math.cos(noisy_theta), arrow_length*math.sin(noisy_theta), head_length=arrow_size, head_width=arrow_size, shape='full', ec='None', fc='blue', alpha=0.5, label='_nolabel_')
                plt.arrow(noisy_x, noisy_y, arrow_length*math.cos(noisy_theta+action[0]*0.4), arrow_length*math.sin(noisy_theta+action[0]*0.4), head_length=arrow_size, head_width=arrow_size, shape='full',ec='None', fc='red', alpha=0.5, label='_nolabel_')
  
                 
                

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
    #ax.set_tick_params(axis=u'both', which=u'both',length=0)
    
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    plt.figlegend(title=legend_title, loc='lower center', ncol=2)
    plt.show()



# agent_names = ['porto_ete_v5_r_collision_5']    
# legend = ['']
# legend_title = ''
# ns=[0]
# start_condition = {'x':10, 'y':4.5, 'v':3, 'theta':np.pi, 'delta':0, 'goal':0}
# mismatch_parameters = [['C_Sf']]
# frac_vary = [[0]]
# noise_dicts = [{'xy':0.25, 'theta':0.1, 'v':0, 'lidar':0}]
# sensitivity_analysis(agent_names=agent_names, ns=ns, legend_title=legend_title,          
#                                              legend=legend, mismatch_parameters=mismatch_parameters, frac_vary=frac_vary, noise_dicts=noise_dicts,
#                                              start_condition=start_condition)


def sensitivity_analysis_line(agent_names, ns, legend_title, legend, mismatch_parameters, frac_vary, noise_dicts, start_condition):
    
    pose_history = []
    progress_history = []
    state_history = []
    local_path_history = []
    action_step_history = []
    
    init_noise_dict = {'xy':0, 'theta':0, 'v':0, 'lidar':0}

    for agent_name, n, i in zip(agent_names, ns, range(len(agent_names))):

        infile = open('environments/' + agent_name, 'rb')
        env_dict = pickle.load(infile)
        infile.close()
        # Compensate for changes to reward structure
        env_dict['reward_signal']['max_progress'] = 0
        
        # Model mismatches
        if mismatch_parameters:
            for par, var in zip(mismatch_parameters[i], frac_vary[i]):
                env_dict['car_params'][par] *= 1+var 

        noise_dict = noise_dicts[i]

        env = environment(env_dict)
        if start_condition:
            env.reset(save_history=True, start_condition=start_condition, car_params=env_dict['car_params'], noise=init_noise_dict)
        else:
            env.reset(save_history=True, start_condition=[], car_params=env_dict['car_params'], noise=init_noise_dict)

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
        env.reset(save_history=True, start_condition=start_condition, car_params=env_dict['car_params'], noise=init_noise_dict)
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
        

    myfont = {'fontname':'serif'}
    figure_size = (5.5,3)
    xlims = [0,100]

    legend_new = legend.copy()
    legend_new.insert(0, 'Min and max')

    legend_racetrack = legend.copy()
    legend_racetrack.insert(0, 'Track centerline')



    plt.figure(1, figsize=figure_size)
    ax = plt.subplot(111)
    ax.axis('off')
    #plt.rc('axes',edgecolor='lightgrey')
    
    #ax.tick_params(axis='both', colors='lightgrey')
    ax.spines['bottom'].set_color('lightgrey')
    ax.spines['top'].set_color('lightgrey') 
    ax.spines['right'].set_color('lightgrey')
    ax.spines['left'].set_color('lightgrey')

    ax.tick_params(axis=u'both', which=u'both',length=0)
    
    track = mapping.map(env.map_name)
    ax.imshow(ImageOps.invert(track.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(1))), extent=(0,track.map_width,0,track.map_height), cmap="gray")
    # ax.plot(env.rx, env.ry, color='gray', linestyle='dashed')
    alpha=0.7

    for i in range(len(agent_names)):
   
        if env_dict['steer_control_dict']['steering_control']:
            for j in np.array(local_path_history[i])[np.arange(0,len(local_path_history[i]),20)]:
                ax.plot(j[0], j[1], alpha=0.5, linestyle='dashdot', color='red')
                ax.plot(j[0][0], j[1][0], alpha=0.5, color='red', marker='s')

        ax.plot(np.array(state_history[i])[:,0], np.array(state_history[i])[:,1], linewidth=1.5, alpha=alpha, label='Path without noise')   
        
        #sample points
        idx = np.arange(0,len(state_history[i]),40)
        idx = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
        sample_states = np.array(state_history[i])[idx]
        sample_actions = np.array(action_step_history[i])[idx]
        
        ax.plot(sample_states[:,0], sample_states[:,1], 'x', color='orange', alpha=alpha, label='_nolabel_')  
        
        arrow_length = 0.1
        arrow_size=0.3
        c1=0
        c2=0
        for state,action in zip(sample_states,sample_actions):
            if c1==0:
                plt.arrow(state[0], state[1], arrow_length*math.cos(state[4]), arrow_length*math.sin(state[4]), head_length=arrow_size, head_width=arrow_size, shape='full', ec='None', fc='blue', alpha=0.5, label='Vehicle heading')
                plt.arrow(state[0], state[1], arrow_length*math.cos(state[4]+action[0]*0.4), arrow_length*math.sin(state[4]+action[0]*0.4), head_length=arrow_size, head_width=arrow_size, shape='full',ec='None', fc='red', alpha=0.5, label='Desired steering\nangle command')
                c1+=1
            if c1>0:
                plt.arrow(state[0], state[1], arrow_length*math.cos(state[4]), arrow_length*math.sin(state[4]), head_length=arrow_size, head_width=arrow_size, shape='full', ec='None', fc='blue', alpha=0.5, label='_nolegend_')
                plt.arrow(state[0], state[1], arrow_length*math.cos(state[4]+action[0]*0.4), arrow_length*math.sin(state[4]+action[0]*0.4), head_length=arrow_size, head_width=arrow_size, shape='full',ec='None', fc='red', alpha=0.5, label='_nolegend_')
                

            for i in range(5):
                noisy_x = state[0]+np.random.normal(loc=0, scale=noise_dict['xy'])
                noisy_y = state[1]+np.random.normal(loc=0, scale=noise_dict['xy'])
                noisy_theta = state[4]%(2*np.pi)+np.random.normal(loc=0, scale=noise_dict['theta'])
                noisy_v = state[3]+np.random.normal(loc=0, scale=noise_dict['v'])
                lidar_dists, lidar_coords = env.lidar.get_scan(noisy_x, noisy_y, noisy_theta)
                x_norm = (noisy_x)/env.map_width
                y_norm = (noisy_y)/env.map_height
                theta_norm = (noisy_theta)/(2*math.pi)
                v_norm = (noisy_v)/env.params['v_max']
                lidar_norm = np.array(lidar_dists+np.random.normal(loc=0, scale=noise_dict['lidar'], size=env.lidar_dict['n_beams']))/env.lidar_dict['max_range']
                observation = [x_norm, y_norm, theta_norm, v_norm]
                for n in lidar_norm:
                    observation.append(n)
                action=a.choose_greedy_action(observation)
                if c2==0:
                    ax.plot(noisy_x, noisy_y, 'x', color='orange', alpha=alpha, label='Vehicle position after\nnoise is added')
                    c2+=1
                if c2>0:
                    ax.plot(noisy_x, noisy_y, 'x', color='orange', alpha=alpha, label='_nolabel_')
                
                plt.arrow(noisy_x, noisy_y, arrow_length*math.cos(noisy_theta), arrow_length*math.sin(noisy_theta), head_length=arrow_size, head_width=arrow_size, shape='full', ec='None', fc='blue', alpha=0.5, label='_nolabel_')
                plt.arrow(noisy_x, noisy_y, arrow_length*math.cos(noisy_theta+action[0]*0.4), arrow_length*math.sin(noisy_theta+action[0]*0.4), head_length=arrow_size, head_width=arrow_size, shape='full',ec='None', fc='red', alpha=0.5, label='_nolabel_')
  
                 
                

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
    #ax.set_tick_params(axis=u'both', which=u'both',length=0)
    
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    plt.figlegend(title=legend_title, loc='lower center', ncol=2)
    plt.show()



# agent_names = ['porto_ete_v5_r_collision_5']    
# legend = ['']
# legend_title = ''
# ns=[0]
# start_condition = {'x':10, 'y':4.5, 'v':3, 'theta':np.pi, 'delta':0, 'goal':0}
# mismatch_parameters = [['C_Sf']]
# frac_vary = [[0]]
# noise_dicts = [{'xy':0.25, 'theta':0.1, 'v':0, 'lidar':0}]
# sensitivity_analysis_line(agent_names=agent_names, ns=ns, legend_title=legend_title,          
#                                              legend=legend, mismatch_parameters=mismatch_parameters, frac_vary=frac_vary, noise_dicts=noise_dicts,
#                                              start_condition=start_condition)



def sensitivity_analysis_noise(agent_name, n, start_condition):
    
    n_acts = 50
    lidar_noise = np.arange(0,1,0.01)
    action_history = np.zeros((len(lidar_noise), n_acts))
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
        
    avg = np.average(action_history, axis=1)
    std_dev = np.std(action_history, axis=1)

    avg_filter = functions.savitzky_golay(avg, 9, 2)
    std_dev_filter = functions.savitzky_golay(std_dev, 9, 2)
    
    fig, ax = plt.subplots(1, figsize=(5.5,3))

    ax.spines['bottom'].set_color('lightgrey')
    ax.spines['top'].set_color('lightgrey') 
    ax.spines['right'].set_color('lightgrey')
    ax.spines['left'].set_color('lightgrey')
    ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.grid(True)
    ax.set_title('')
    ax.set_xlabel('LiDAR noise standard deviation [m]')
    ax.set_ylabel('Steering control action')
    # ax.plot(lidar_noise, avg)
    # ax.fill_between(x=lidar_noise, y1=avg-std_dev, y2=avg+std_dev,alpha=0.5)

    ax.plot(lidar_noise, avg_filter)
    ax.fill_between(x=lidar_noise, y1=avg_filter-std_dev_filter, y2=avg_filter+std_dev_filter,alpha=0.3)
    ax.hlines(y=1,xmin=0,xmax=lidar_noise[-1], color='k', linestyles='--')
    ax.hlines(y=-1,xmin=0,xmax=lidar_noise[-1], color='k', linestyles='--')

    fig.tight_layout()
    fig.subplots_adjust(right=0.75)
    plt.figlegend(['Average', 'Standard\ndeviation'], loc='center right', ncol=1)
    plt.show()


# agent_name = 'porto_ete_v5_r_collision_5' 
# legend = ['']
# legend_title = ''
# n=0
# start_condition = {'x':4, 'y':4.8, 'v':5, 'theta':np.pi, 'delta':0, 'goal':0}
# sensitivity_analysis_noise(agent_name=agent_name, n=n, start_condition=start_condition)



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

    terminal_poses_file_name = 'terminal_poses/' + agent_name

    infile = open('environments/' + agent_name, 'rb')
    env_dict = pickle.load(infile)
    infile.close()

    infile = open(terminal_poses_file_name, 'rb')
    terminal_poses = pickle.load(infile)
    infile.close()

    end_episode = np.where(terminal_poses[0]==0)[0][0]
    print('End episode = ', end_episode)
    env = environment(env_dict)
    
    figure_size = (4,2)
    plt.figure(1, figsize=figure_size)
    ax = plt.subplot(111)

    track = mapping.map(env.map_name)
    ax.imshow(ImageOps.invert(track.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(1))), extent=(0,track.map_width,0,track.map_height), cmap="gray")
    ax.plot(env.rx, env.ry, color='gray', linestyle='dashed')
    ax.plot(np.array(terminal_poses)[0,0:end_episode,0], np.array(terminal_poses)[0,0:end_episode,1], 'rx')
    #sns.jointplot(x=np.array(terminal_poses)[:,0],y=np.array(terminal_poses)[:,1], kind="hex", alpha=0.5)
    ax.axis('off')
    plt.show()

# display_collision_distribution('collision_distribution_LiDAR')
# display_collision_distribution('collision_distribution_no_LiDAR')
# display_collision_distribution('collision_distribution_LiDAR_pose')

def display_path_multiple(agent_names, ns, legend_title, legend, mismatch_parameters, frac_vary, noise_dicts, start_condition):
    
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
        
        car_params = env_dict['car_params'].copy()
        # Model mismatches
        
        mass=car_params['m']*0.1

        if mismatch_parameters:
            for par, var in zip(mismatch_parameters[i], frac_vary[i]):
                if par == 'unknown_mass':
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
        
        
        noise_dict = noise_dicts[i]

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
        
        
    myfont = {'fontname':'serif'}
    figure_size = (10,4)
    xlims = [0,100]

    legend_new = legend.copy()
    legend_new.insert(0, 'Min and max')

    legend_racetrack = legend.copy()
    # legend_racetrack.insert(0, 'Track centerline')



    plt.figure(1, figsize=figure_size)
    ax = plt.subplot(111)
    
    #plt.rc('axes',edgecolor='lightgrey')
    
    #ax.tick_params(axis='both', colors='lightgrey')
    ax.spines['bottom'].set_color('lightgrey')
    ax.spines['top'].set_color('lightgrey') 
    ax.spines['right'].set_color('lightgrey')
    ax.spines['left'].set_color('lightgrey')

    ax.tick_params(axis=u'both', which=u'both',length=0)
    
    track = mapping.map(env.map_name)
    ax.imshow(ImageOps.invert(track.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(1))), extent=(0,track.map_width,0,track.map_height), cmap="gray")
    # ax.plot(env.rx, env.ry, color='gray', linestyle='dashed')
    alpha=0.7

    for i in range(len(agent_names)):
   
        # if env_dict['steer_control_dict']['steering_control']:
        #     for j in np.array(local_path_history[i])[np.arange(0,len(local_path_history[i]),20)]:
        #         ax.plot(j[0], j[1], alpha=0.5, linestyle='dashdot', color='red')
        #         ax.plot(j[0][0], j[1][0], alpha=0.5, color='red', marker='s')

        ax.plot(np.array(state_history[i])[:,0], np.array(state_history[i])[:,1], linewidth=1.5, alpha=alpha)   
        # ax.plot(np.array(pose_history[i])[:,0][np.arange(0,len(local_path_history[i]),40)], np.array(pose_history[i])[:,1][np.arange(0,len(local_path_history[i]),40)], 'x')
        
        #ax.plot(np.array(pose_history[i])[:,0], np.array(pose_history[i])[:,1], linewidth=1.5, alpha=alpha)    

    prog = np.array([0, 0.2, 0.4, 0.6, 0.8])
    idx =  np.zeros(len(prog), int)
    text = ['Start', '20%', '40%', '60%', '80%']

    for i in range(len(idx)):
        idx[i] = np.mod(env.start_point+np.round(prog[i]*len(env.rx)), len(env.rx))
    idx.astype(int)
    
    for i in range(len(idx)):
        plt.text(x=env.rx[idx[i]], y=env.ry[idx[i]], s=text[i], fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))


    ax.set_xlabel('x coordinate [m]',**myfont) 
    ax.set_ylabel('y coordinate [m]',**myfont)
    #ax.set_tick_params(axis=u'both', which=u'both',length=0)
    
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    # ax.legend(legend, title=legend_title, loc='center left',  bbox_to_anchor=(1, 0.5))
    ax.legend(legend, loc='center left',  bbox_to_anchor=(1, 0.5))
    #plt.legend(legend_new, title=legend_title, loc='lower right')





    plt.figure(2, figsize=figure_size)
    plt.rc('axes', edgecolor='lightgrey')

    plt.hlines(y=env_dict['action_space_dict']['vel_select'][0], xmin=0, xmax=100, colors='black', linestyle='dashed')
    plt.hlines(y=env_dict['action_space_dict']['vel_select'][1], xmin=0, xmax=100, colors='black', linestyle='dashed', label='_nolegend_')
    for i in range(len(agent_names)):
        plt.plot(np.array(progress_history[i])*100, np.array(state_history[i])[:,3], linewidth=1.5, alpha=alpha)
        # plt.plot(np.array(progress_history[i])*100, np.array(pose_history[i])[:,4], linewidth=1.5, alpha=alpha)

    plt.xlabel('progress along centerline [%]',**myfont)
    plt.ylabel('Longitudinal velocity [m/s]',**myfont)
    plt.legend(legend_new, title=legend_title, loc='lower right')
    plt.xlim(xlims)
    #plt.ylim([env_dict['action_space_dict']['vel_select'][0]-0.2, env_dict['action_space_dict']['vel_select'][1]+0.2])
    plt.grid(True, color='lightgrey')

    plt.tick_params(axis=u'both', which=u'both',length=0)



    plt.figure(3, figsize=figure_size)
    plt.rc('axes',edgecolor='lightgrey')

    plt.hlines(y=env_dict['car_params']['s_min'], xmin=0, xmax=100, colors='black', linestyle='dashed')
    plt.hlines(y=env_dict['car_params']['s_max'], xmin=0, xmax=100, colors='black', linestyle='dashed', label='_nolegend_')
    for i in range(len(agent_names)):
        plt.plot(np.array(progress_history[i])*100, np.array(state_history[i])[:,2], linewidth=1.5, alpha=alpha)
        # plt.plot(np.array(progress_history[i])*100, np.array(pose_history[i])[:,3], linewidth=1.5, alpha=alpha)

    plt.xlabel('progress along centerline [%]',**myfont)
    plt.ylabel('steering angle [rads]',**myfont)
    plt.legend(legend_new, title=legend_title, loc='lower right')
    plt.xlim(xlims)
    plt.ylim([env_dict['car_params']['s_min']-0.05, env_dict['car_params']['s_max']+0.05])
    plt.grid(True, color='lightgrey')
    plt.tick_params(axis=u'both', which=u'both',length=0)



    plt.figure(4, figsize=figure_size)
    plt.rc('axes',edgecolor='lightgrey')
    for i in range(len(agent_names)):
        plt.plot(np.array(progress_history[i])*100, np.array(state_history[i])[:,6], linewidth=1.5, alpha=alpha)
      
    plt.xlabel('progress along centerline [%]',**myfont)
    plt.ylabel('Slip angle [rads]',**myfont)
    plt.legend(legend, title=legend_title, loc='lower right')
    plt.xlim(xlims)
    plt.ylim([-1,1])
    plt.grid(True, color='lightgrey')
    plt.tick_params(axis=u'both', which=u'both',length=0)



    plt.figure(5, figsize=figure_size)

    max_idx = np.zeros(len(agent_names))
    for i in range(len(agent_names)):
        max_idx[i] = np.argmax(np.array(progress_history)[i])

    plt.rc('axes',edgecolor='lightgrey')
    plt.hlines(y=100, xmin=0, xmax=np.argmax(max_idx), colors='black', linestyle='dashed')
    plt.hlines(y=0, xmin=0, xmax=np.argmax(max_idx), colors='black', linestyle='dashed', label='_nolegend_')
    
    for i in range(len(agent_names)):
        
        plt.plot(np.arange(len(progress_history[i])), np.array(progress_history[i])*100, linewidth=1.5, alpha=alpha)
    
    plt.xlabel('Simulation step',**myfont)
    plt.ylabel('progress along centerline [%]',**myfont)
    plt.ylim([-5,105])
    plt.legend(legend_new, title=legend_title, loc='lower right')
    plt.grid(True, color='lightgrey')
    plt.tick_params(axis=u'both', which=u'both',length=0)



    plt.figure(6, figsize=figure_size)
    plt.rc('axes',edgecolor='lightgrey')
    plt.hlines(y=1, xmin=0, xmax=100, colors='black', linestyle='dashed')
    plt.hlines(y=-1, xmin=0, xmax=100, colors='black', linestyle='dashed', label='_nolegend_')
    
    for i in range(len(agent_names)):
        plt.plot(np.array(progress_history[i])[0:len(np.array(action_step_history[i])[:,0])]*100, np.array(action_step_history[i])[:,0], linewidth=1.5, alpha=alpha)
    
    plt.xlabel('Simulation step',**myfont)
    plt.ylabel('Latitude action',**myfont)
    plt.ylim([-1,1])
    plt.legend(legend_new, title=legend_title, loc='lower right')
    plt.grid(True, color='lightgrey')
    plt.tick_params(axis=u'both', which=u'both',length=0)



    plt.figure(7, figsize=figure_size)
    plt.rc('axes',edgecolor='lightgrey')
    plt.hlines(y=1, xmin=0, xmax=100, colors='black', linestyle='dashed')
    plt.hlines(y=-1, xmin=0, xmax=100, colors='black', linestyle='dashed', label='_nolegend_')
    
    for i in range(len(agent_names)):
        plt.plot(np.array(progress_history[i])[0:len(np.array(action_step_history[i])[:,1])]*100, np.array(action_step_history[i])[:,1], linewidth=1.5, alpha=alpha)
    
    plt.xlabel('Simulation step',**myfont)
    plt.ylabel('Longitude action',**myfont)
    plt.ylim([-1,1])
    plt.legend(legend_new, title=legend_title, loc='lower right')
    plt.grid(True, color='lightgrey')
    plt.tick_params(axis=u'both', which=u'both',length=0)
    plt.show()




def display_path_mismatch_multiple(agent_names, ns, legend_title, legend, mismatch_parameters, frac_vary, noise_dicts, start_condition):
    
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
        
        
    myfont = {'fontname':'serif'}
    figure_size = (10,4)
    xlims = [0,100]

    legend_new = legend.copy()
    legend_new.insert(0, 'Min and max')

    legend_racetrack = legend.copy()

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
        
    ax[0,0].set_title('End-to-end')
    ax[0,1].set_title('Steering controller')
    ax[1,0].set_title('Velocity controller')
    ax[1,1].set_title('Steering and velocity controllers')
    
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12) 
    
    plt.figlegend(legend, loc='lower center', ncol=2, labelspacing=0.7)
    plt.show()   


agent_names = ['porto_ete_v5_r_collision_5', 'porto_pete_s_polynomial', 'porto_pete_v_k_1_attempt_2', 'porto_pete_sv_p_r_0']    
legend = ['No model error', 'Mass placed on front axle']
legend_title = ''
ns=[0,0,0,0]
mismatch_parameters = ['unknown_mass']
frac_vary = [0]
noise_dicts = [{'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}]
start_condition = {'x':10, 'y':4.5, 'v':3, 'theta':np.pi, 'delta':0, 'goal':0}
# display_path_mismatch_multiple(agent_names=agent_names, ns=ns, legend_title=legend_title,          
#                                              legend=legend, mismatch_parameters=mismatch_parameters, frac_vary=frac_vary, noise_dicts=noise_dicts,
#                                              start_condition=start_condition)




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
#display_map_outline()




#display_maps()



    

