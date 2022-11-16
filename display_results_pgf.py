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





def learning_curve_lap_time_average(agent_names, legend, legend_title, ns, filename):

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
    fig, ax = plt.subplots(1, 2, figsize=(5.5,3))

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
    ax[0].set_xlabel('Episodes')
    #plt.title('Collision rate')
    ax[0].set_ylabel('Failure rate [%]')
    ax[0].tick_params('both', length=0)
    ax[0].grid(True)
    ax[0].set_xlim([0,4000])
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
    ax[1].set_xlim([0,4000])
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
    fig.subplots_adjust(bottom=0.37) 
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
    
    plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False,     # don't setup fonts from rc parameters
    "font.size": 12
    })

    fig, ax = plt.subplots(1, figsize=(5,2))
    plt.rc('axes',edgecolor='gray')
    
    
    color='grey'
    ax.spines['bottom'].set_color(color)
    ax.spines['top'].set_color(color) 
    ax.spines['right'].set_color(color)
    ax.spines['left'].set_color(color)


    for i in range(len(agent_names)):
        end_episode = end_episodes[i] 
        ax.plot(np.arange(end_episode),  avg_score[i][0:end_episode])
        ax.fill_between(x=np.arange(end_episode), y1=upper_fill[i][0][0:end_episode], y2=lower_fill[i][0][0:end_episode], alpha=0.3, label='_nolegend_')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode\nreward')
    #ax.legend(legend_new, title=legend_title, loc='lower right')
    #plt.xlim([0,5000])
    ax.grid(True)
    ax.tick_params(axis=u'both', which=u'both',length=0)
    
    fig.tight_layout()
    fig.subplots_adjust(right=0.75) 
    plt.figlegend(legend, title=legend_title, loc='center right', ncol=1)


    #plt.show()
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
# filename = 'observation_n_beams'

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


# agent_names = ['porto_ete_LiDAR_10', 'porto_ete_ddpg']
# legend = ['TD3', 'DDPG']
# legend_title = 'Learning \nmethod'
# ns=[0, 0]
# filename = 'learning_method_reward'

# agent_names = ['porto_ete_v_5']
# legend = ['End-to-end agent']
# legend_title = ''
# ns=[2]
# filename = 'end_to_end_agent_v5'

# agent_names = ['porto_ete_v5_r_dist_1', 'porto_ete_v_5', 'porto_ete_v5_r_dist_2', 'porto_ete_v5_r_dist_4']
# legend = ['0.1', '0.3', '0.5', '1']
# legend_title = 'Distance reward ($r_{\mathrm{dist}}$)'
# ns=[0, 0, 0, 0]
# filename = 'distance_reward_v5'

# mismatch_parameters = ['C_Sf']
# frac_vary = [0]
# start_condition = {'x':10, 'y':4.5, 'v':3, 'theta':np.pi, 'delta':0, 'goal':0}
# #start_condition = []
# #filename = 'end_to_end_agent'
# display_path_multiple(agent_names=agent_names, ns=ns, legend_title=legend_title,          
#                         legend=legend, mismatch_parameters=mismatch_parameters, frac_vary=frac_vary, 
#                         start_condition=start_condition, filename=filename)


learning_curve_lap_time_average(agent_names, legend, legend_title, ns, filename)

#learning_curve_reward_average(agent_names, legend, legend_title)

#display_velocity_slip(agent_names, ns, legend_title, legend, mismatch_parameters, frac_vary, start_condition, filename)