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



def graph_eval_time_steps(agent_names):
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rc('axes',edgecolor='gray')

    fig, axs = plt.subplots(figsize=(4.5,2.5))

    for i in range(len(agent_names)):
        
        agent_name = agent_names[i]
        
        eval_file_name = 'evaluation_results/' + agent_name
        infile = open(eval_file_name, 'rb')
        eval_steps = pickle.load(infile)
        eval_lap_times = pickle.load(infile)
        eval_collisions = pickle.load(infile)
        infile.close()



        last_row=len(eval_lap_times[0])
        for n in range(len(eval_lap_times)):
            new_last_row = np.where(np.all(eval_lap_times[n]==0,axis=1))[0][0]
            if new_last_row < last_row:
                last_row = new_last_row 


        eval_steps = eval_steps[:,0:last_row]
        eval_lap_times = eval_lap_times[:,0:last_row]
        eval_collisions = eval_collisions[:,0:last_row]
        
      
        avg_lap_times = np.zeros(len(eval_lap_times[0,:,0]))
        std_lap_times = np.zeros(len(eval_lap_times[0,:,0]))
        collisions = np.zeros(len(eval_lap_times[0,:,0]))
        std_collisions = np.zeros(len(eval_lap_times[0,:,0]))

        for n in range(len(avg_lap_times)):
            mask = np.logical_not(eval_collisions[:,n].astype(bool))
            avg_lap_times[n] = np.average(eval_lap_times[:,n][mask])
            std_lap_times[n] =  np.std(eval_lap_times[:,n][mask])
            collisions[n] = np.average(mask)*100
            # std_collisions[n] = np.std(mask)*100
        pass
        
        xaxis = np.average(eval_steps,axis=0)/(20)
        
        axs.plot(xaxis, avg_lap_times, label='Lap time')
        axs.fill_between(x=xaxis, y1=avg_lap_times-std_lap_times,y2=avg_lap_times+std_lap_times, alpha=0.2)
        axs.set_xlabel('MDP time steps')
        axs.set_ylabel('Lap time [s]')
        axs.tick_params(axis=u'both', which=u'both',length=0)
        axs.grid(True, color='lightgrey')
        axs.ticklabel_format(style='scientific', axis='x', scilimits=(0,0), useMathText=True, useOffset=True)

        axs1 = axs.twinx()  
        axs1.plot(xaxis, collisions, color='orange', label='Successful laps')
        axs1.set_ylim([80,101])
        axs1.set_ylabel('Successful laps [%]')
        axs1.tick_params(axis=u'both', which=u'both',length=0)

        
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.35) 
        plt.figlegend(loc = 'lower center', ncol=3)    
    
    plt.show()


# graph_eval_time_steps(['time_steps'])



def graph_replay_batch_size(agent_names):
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rc('axes',edgecolor='gray')

    batch_size = np.array([50,100,150,200,300,400,600,1000])
    train_times = np.zeros(len(agent_names))
    min_train_times = np.zeros(len(agent_names))
    max_train_times = np.zeros(len(agent_names))
    lap_times = np.zeros(len(agent_names))
    std_lap_times = np.zeros(len(agent_names))
    collisions = np.zeros(len(agent_names))

    for i in range(len(agent_names)):
        
        agent_name = agent_names[i]
        
        train_file_name = 'train_results/' + agent_name
        infile = open(train_file_name, 'rb')
        _ = pickle.load(infile)
        _ = pickle.load(infile)
        agent_train_times = pickle.load(infile)
        infile.close()

        test_file_name = 'lap_results_with_noise/' + agent_name
        infile = open(test_file_name, 'rb')
        agent_lap_times = pickle.load(infile)
        agent_collisions = pickle.load(infile)
        infile.close() 

        agent_lap_times[agent_collisions.astype(bool)] = np.nan
        

        train_times[i] = np.sum(agent_train_times)/(60*3)
        min_train_times[i] = np.min(np.sum(agent_train_times,axis=1))/(60)
        max_train_times[i] = np.max(np.sum(agent_train_times,axis=1))/(60)
        lap_times[i] = np.nanmean(agent_lap_times)
        std_lap_times[i] = np.nanstd(agent_lap_times)
        collisions[i] = np.average(agent_collisions)

    
    fig, axs = plt.subplots(figsize=(4.5,2.5))
    axs.plot(batch_size, train_times, label='Training time')
    axs.fill_between(x=batch_size, y1=min_train_times,y2=max_train_times, alpha=0.2)
    axs.set_xlabel('Batch size')
    axs.set_ylabel('Training time [minutes]')
    axs.set_ylim([20,40])
    axs.tick_params(axis=u'both', which=u'both',length=0)
    axs.grid(True, color='lightgrey')

    axs1 = axs.twinx()  
    axs1.plot(batch_size, lap_times, color='orange', label='Test lap time')
    axs1.fill_between(x=batch_size, y1=lap_times-std_lap_times,y2=lap_times+std_lap_times, alpha=0.2, color='orange')
    axs1.set_ylabel('Test lap time [s]')
    axs1.tick_params(axis=u'both', which=u'both',length=0)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35) 
    plt.figlegend(loc = 'lower center', ncol=3)    

    plt.show()


# graph_replay_batch_size(['batch_50', 'batch_100', 'batch_150', 'batch_200', 'batch_300', 'batch_400', 'batch_600', 'batch_1000'])


def graph_agent_sample_rate(agent_names):
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rc('axes',edgecolor='gray')

    sample_rate = np.array([3,5,10,20,50])
    train_times = np.zeros(len(agent_names))
    min_train_times = np.zeros(len(agent_names))
    max_train_times = np.zeros(len(agent_names))
    lap_times = np.zeros(len(agent_names))
    std_lap_times = np.zeros(len(agent_names))
    collisions = np.zeros(len(agent_names))

    for i in range(len(agent_names)):
        
        agent_name = agent_names[i]
        
        train_file_name = 'train_results/' + agent_name
        infile = open(train_file_name, 'rb')
        _ = pickle.load(infile)
        _ = pickle.load(infile)
        agent_train_times = pickle.load(infile)
        infile.close()

        test_file_name = 'lap_results_with_noise/' + agent_name
        infile = open(test_file_name, 'rb')
        agent_lap_times = pickle.load(infile)
        agent_collisions = pickle.load(infile)
        infile.close() 

        agent_lap_times[agent_collisions.astype(bool)] = np.nan
        
        train_times[i] = np.sum(agent_train_times)/(60*3)
        min_train_times[i] = np.min(np.sum(agent_train_times,axis=1))/(60)
        max_train_times[i] = np.max(np.sum(agent_train_times,axis=1))/(60)
        collisions[i] = np.average(agent_collisions)
        lap_times[i] = np.nanmean(agent_lap_times)
        std_lap_times[i] = np.nanstd(agent_lap_times)
    
    fig, axs = plt.subplots(figsize=(4.5,2.5))
    axs.plot(sample_rate, train_times, label='Training time')
    # axs.fill_between(x=xaxis, y1=,y2=, alpha=0.2)
    axs.set_xlabel('Agent sample rate [Hz]')
    axs.set_ylabel('Training time [minutes]')
    # axs.set_ylim([20,40])
    axs.tick_params(axis=u'both', which=u'both',length=0)
    axs.grid(True, color='lightgrey')
    
    axs1 = axs.twinx()  
    axs1.plot(sample_rate, collisions*100, color='orange', label='Failed laps')
    # axs1.fill_between(x=sample_rate, y1=lap_times-std_lap_times,y2=lap_times+std_lap_times, alpha=0.2, color='orange')
    axs1.set_ylabel('Failed laps [%]')
    axs1.tick_params(axis=u'both', which=u'both',length=0)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35) 
    plt.figlegend(loc = 'lower center', ncol=3)    

    plt.show()


agent_names = ['f_agent_3', 'f_agent_5', 'f_agent_10', 'f_agent_20', 'f_agent_50']
graph_agent_sample_rate(agent_names)