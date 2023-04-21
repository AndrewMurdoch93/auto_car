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

def graph_eval(agent_names):
    
    
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
        
        xaxis = np.average(eval_steps,axis=0)

        axs.plot(xaxis, avg_lap_times, label='Lap time [s]')
        axs.fill_between(x=xaxis, y1=avg_lap_times-std_lap_times,y2=avg_lap_times+std_lap_times, alpha=0.2)
        axs.set_xlabel('Training steps')
        axs.set_ylabel('Lap time [s]')
        axs.tick_params(axis=u'both', which=u'both',length=0)
        axs.grid(True, color='lightgrey')

        axs1 = axs.twinx()  
        axs1.plot(xaxis, collisions, color='orange', label='Successful laps [%]')
        axs1.set_ylim([80,101])
        axs1.set_ylabel('Successful laps [%]')
        axs1.tick_params(axis=u'both', which=u'both',length=0)

        
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.35) 
        plt.figlegend(loc = 'lower center', ncol=3)    
    
    plt.show()




graph_eval(['time_steps'])
# 'time_steps'
