from cProfile import label
import functions
import numpy as np
from matplotlib import  pyplot as plt
from matplotlib import image
import math
import cmath
import yaml
from argparse import Namespace
import bisect
import sys
import cubic_spline_planner
import yaml
from PIL import Image, ImageOps, ImageDraw, ImageFilter
import random
from datetime import datetime
import time
from numba import njit
from numba import int32, int64, float32, float64,bool_    
from numba.experimental import jitclass
import pickle
import mapping
import cubic_spline_planner
import display_results_multiple
from environment import environment




def plot_frenet_polynomial():
    map_name='porto_1'

    track = mapping.map(map_name)
    occupancy_grid = track.occupancy_grid
    map_height = track.map_height
    map_width = track.map_width
    map_res = track.resolution

    track.find_centerline()
    goal_x = track.centerline[:,0]
    goal_y = track.centerline[:,1]
    rx, ry, ryaw, rk, ds, csp = functions.generate_line(goal_x, goal_y)

    LiDAR = functions.lidar_scan(lidar_res=0.1, n_beams=2, max_range=10, fov=np.pi, occupancy_grid=occupancy_grid, map_res=map_res, map_height=map_height)

    LiDAR_dists = np.zeros((len(rx),2))

    for idx, (x,y,theta)  in enumerate(zip(rx,ry,ryaw)):
        lidar_dists, lidar_coords = LiDAR.get_scan(x, y, theta)
        LiDAR_dists[idx,0] = lidar_dists[0]
        LiDAR_dists[idx,1] = lidar_dists[1]
        
    
    
    x_pose = rx[150]
    y_pose = ry[150]
    theta_pose = 0

    
    s_pose, ind, n_pose = functions.convert_xy_to_sn(rx, ry, ryaw, x_pose, y_pose, 0.1)


    s_0s = np.array([s_pose,s_pose,s_pose])
    n_0s = np.array([n_pose,n_pose,s_pose])
    thetas = np.array([theta_pose,theta_pose,theta_pose])
    n_1s = np.array([0.5, 0.7, -0.7])

    s_1s = s_0s+2
    s_2s = s_1s+1
    
    s_ = [[] for _ in range(len(s_0s))]
    n_ = [[] for _ in range(len(s_0s))]


    for idx, (s_0, n_0, theta, n_1, s_1, s_2) in enumerate(zip(s_0s, n_0s, thetas, n_1s, s_1s, s_2s)): 
       
        A = np.array([[3*s_1**2, 2*s_1, 1, 0], [3*s_0**2, 2*s_0, 1, 0], [s_0**3, s_0**2, s_0, 1], [s_1**3, s_1**2, s_1, 1]])
        B = np.array([0, theta, n_0, n_1])
        x = np.linalg.solve(A, B)


        a = x[0]
        b = x[1]
        c = x[2]
        d = x[3]

        s = np.linspace(s_0, s_1)
        n = a*s**3 + b*s**2 + c*s + d
        s = np.concatenate((s, np.linspace(s_1, s_2)))
        n = np.concatenate((n, np.ones(len(np.linspace(s_1, s_2)))*n_1))
        
        s_[idx].append(s)
        n_[idx].append(n)
    


    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    fig, ax = plt.subplots(figsize=(5.5,3.5))
     
    color='grey'
    ax.tick_params(axis=u'both', which=u'both',length=0)
    ax.spines['bottom'].set_color(color)
    ax.spines['top'].set_color(color) 
    ax.spines['right'].set_color(color)
    ax.spines['left'].set_color(color)

    # ax.set_xticks(ticks=[], labels=[])
    # ax.set_yticks(ticks=[], labels=[])

    ax.plot(ds, LiDAR_dists[:,0], color='black') #top boundary
    ax.plot(ds, -LiDAR_dists[:,1], color='black', label='Track boundary') #Bottom boundary

    ax.plot(ds, np.zeros(len(ds)), color='grey', linestyle='dashed', label='Centerline') #centerline

    ax.vlines(x=0, ymin=-1, ymax=1, color='red', linestyle='dashed') #start
    ax.vlines(x=ds[-1], ymin=-1, ymax=1, color='red', linestyle='dashed') #start

    ax.text(x=0-1, y=1, s='Start', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))
    ax.text(x=ds[-1]-1, y=1, s='Finish', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))


    ax.plot(s_[0][0], n_[0][0], alpha=0.8, color='red', linestyle='dashdot', label='Sampled path')
    # ax.plot(s_[1][0], n_[1][0], label='', alpha=0.8, color='red', linestyle='dashdot')
    # ax.plot(s_[2][0], -n_[1][0], label='', alpha=0.8, color='red', linestyle='dashdot')
    ax.fill_between(x=s_[1][0], y1=n_[1][0], y2=-n_[1][0], alpha=0.1, color='red', label='Selectable paths')
    
    ax.plot(s_pose, n_pose,'o',color='Orange', label='Vehicle pose')
    
    # ax.grid(True)
    ax.set_xlabel('Distance along \ncenterline, $s$ [m]')
    ax.set_ylabel('Distance perpendicular \nto centerline, $n$ [m]')
    fig.tight_layout()
    plt.figlegend(loc = 'lower center', ncol=2)
    fig.subplots_adjust(bottom=0.4) 
    plt.show()

    
    cx, cy, cyaw, ds, c = functions.convert_sn_to_xy(s_[0][0], n_[0][0], csp)
    cx_min, cy_min, cyaw_min, ds_min, c_min = functions.convert_sn_to_xy(s_[1][0], n_[1][0], csp)
    cx_max, cy_max, cyaw_max, ds_max, c_max = functions.convert_sn_to_xy(s_[1][0], -n_[1][0], csp)

    fig, ax = plt.subplots(figsize=(5.5,3))
    ax.imshow(ImageOps.invert(track.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(1))), extent=(0,track.map_width,0,track.map_height), cmap="gray")
    ax.plot(rx, ry, color='grey', linestyle='dashed', label='Centerline')
    # ax.plot(rx[0], ry[0], 'x')
    ax.plot(cx,cy, color='red', alpha=0.8, linestyle='dashdot', label='Sampled path')
    # ax.plot(cx_max,cy_min, color='red', alpha=0.8, linestyle='dashdot')
    # ax.plot(cx_max,cy_max, color='red', alpha=0.8, linestyle='dashdot')
    ax.fill_between(x=cx_max, y1=cy_min, y2=cy_max, color='red', alpha=0.1, label='Selectable paths')
    plt.plot(x_pose,y_pose,'o',color='Orange', label='Vehicle pose')

    ax.axis('off')
    plt.figlegend(loc = 'lower center', ncol=2)
    # fig.subplots_adjust(bottom=0.2) 
    plt.show()


    fig, ax = plt.subplots(figsize=(5.5,3))
    ax.imshow(ImageOps.invert(track.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(1))), extent=(0,track.map_width,0,track.map_height), cmap="gray")
    ax.plot(rx, ry, color='grey', linestyle='dashed', label='Centerline')
    plt.plot(x_pose,y_pose,'o',color='Orange', label='Vehicle pose')

    ax.axis('off')
    plt.figlegend(loc = 'lower center', ncol=2)
    # fig.subplots_adjust(bottom=0.2) 
    plt.show()





def plot_frenet_path():
    
    map_name='f1_esp'
    agent_names = ['porto_pete_sv_p_r_0']
    ns = [0]
    mismatch_parameters = []
    frac_vary = []
    noise_dicts = [{'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}]
    start_condition = {'x':8.9, 'y':2.8, 'v':3, 'theta':0, 'delta':0, 'goal':0}
    figure_size = (5.5,3.2)

    infile = open('environments/' + agent_names[0], 'rb')
    env_dict = pickle.load(infile)
    infile.close()
    env = environment(env_dict)

    track = mapping.map(map_name)
    occupancy_grid = track.occupancy_grid
    map_height = track.map_height
    map_width = track.map_width
    map_res = track.resolution

    track.find_centerline()
    goal_x = track.centerline[:,0]
    goal_y = track.centerline[:,1]
    rx, ry, ryaw, rk, ds, csp = functions.generate_line(goal_x, goal_y)

    LiDAR = functions.lidar_scan(lidar_res=0.1, n_beams=2, max_range=10, fov=np.pi, occupancy_grid=occupancy_grid, map_res=map_res, map_height=map_height)

    LiDAR_dists = np.zeros((len(rx),2))

    for idx, (x,y,theta)  in enumerate(zip(rx,ry,ryaw)):
        lidar_dists, lidar_coords = LiDAR.get_scan(x, y, theta)
        LiDAR_dists[idx,0] = lidar_dists[0]
        LiDAR_dists[idx,1] = lidar_dists[1]
        
    
    state_history = display_results_multiple.eval_lap(agent_names, ns, mismatch_parameters, frac_vary, noise_dicts, start_condition)

    x = np.array(state_history[0])[:,0]
    y = np.array(state_history[0])[:,1]

    s = np.zeros(len(x))
    n = np.zeros(len(y))

    for i in range(len(s)):
        (s[i], ind, n[i]) = functions.convert_xy_to_sn(rx, ry, ryaw, x[i], y[i], 0.1)

    # s_ = s-s[0]
    # s_=np.mod(s_,np.max(s_))

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']



    fig1, ax = plt.subplots(1, 2, figsize=figure_size)
     
    color='grey'
    ax[1].tick_params(axis=u'both', which=u'both',length=0)
    ax[1].spines['bottom'].set_color(color)
    ax[1].spines['top'].set_color(color) 
    ax[1].spines['right'].set_color(color)
    ax[1].spines['left'].set_color(color)

    # ax.set_xticks(ticks=[], labels=[])
    # ax.set_yticks(ticks=[], labels=[])

    ax[1].plot(ds, LiDAR_dists[:,0], color='black') #top boundary
    ax[1].plot(ds, -LiDAR_dists[:,1], color='black', label='Track boundary') #Bottom boundary

    ax[1].plot(ds, np.zeros(len(ds)), color='grey', linestyle='dashed', label='Centerline') #centerline

    ax[1].vlines(x=0, ymin=-1, ymax=1, color='red', linestyle='dashed', label='Start/ finish') #start
    ax[1].vlines(x=ds[-1], ymin=-1, ymax=1, color='red', linestyle='dashed', label='_nolegend_') #start

    # ax1.text(x=0-1, y=1, s='Start', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))
    # ax1.text(x=ds[-1]-1, y=1, s='Finish', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))

    ax[1].plot(s[5:-5],n[5:-5], label='Vehicle trajectory')
    ax[1].set_xlabel('Distance along \n centerline, $n$')
    ax[1].set_ylabel('Distance perpendicular \n to centerline, $s$')
    # ax[1].set_title('(b)')
    ax[1].text(x=15,y=2,s='(b)')
    ax[0].text(x=8,y=9.5,s='(a)')


    # fig2, ax2 = plt.subplots(1, figsize=figure_size)
    ax[0].axis('off')
    # ax[0].set_title('(a)')
    # ax.tick_params(axis='both', colors='lightgrey')
    # ax.spines['bottom'].set_color('lightgrey')
    # ax.spines['top'].set_color('lightgrey') 
    # ax.spines['right'].set_color('lightgrey')
    # ax.spines['left'].set_color('lightgrey')

    ax[0].tick_params(axis=u'both', which=u'both',length=0)
    
    track = mapping.map(map_name)
    ax[0].imshow(ImageOps.invert(track.gray_im.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(1))), extent=(0,track.map_width,0,track.map_height), cmap="gray")
    # ax.plot(env.rx, env.ry, color='gray', linestyle='dashed')
    alpha=0.7

    ax[0].plot(rx, ry, linestyle='--', color='grey', label='_nolegend_')

    for i in range(len(agent_names)):
   
        # if env_dict['steer_control_dict']['steering_control']:
        #     for j in np.array(local_path_history[i])[np.arange(0,len(local_path_history[i]),20)]:
        #         ax.plot(j[0], j[1], alpha=0.5, linestyle='dashdot', color='red')
        #         ax.plot(j[0][0], j[1][0], alpha=0.5, color='red', marker='s')

        ax[0].plot(np.array(state_history[i])[:,0], np.array(state_history[i])[:,1], linewidth=1.5, alpha=alpha, label='_nolegend_')   
        # ax.plot(np.array(pose_history[i])[:,0][np.arange(0,len(local_path_history[i]),40)], np.array(pose_history[i])[:,1][np.arange(0,len(local_path_history[i]),40)], 'x')
        
        #ax.plot(np.array(pose_history[i])[:,0], np.array(pose_history[i])[:,1], linewidth=1.5, alpha=alpha)    

    # prog = np.array([0, 0.2, 0.4, 0.6, 0.8])
    # idx =  np.zeros(len(prog), int)
    # text = ['Start', '20%', '40%', '60%', '80%']

    # for i in range(len(idx)):
    #     idx[i] = np.mod(env.start_point+np.round(prog[i]*len(env.rx)), len(env.rx))
    # idx.astype(int)
    
    # # for i in range(len(idx)):
    # #     plt.text(x=env.rx[idx[i]], y=env.ry[idx[i]], s=text[i], fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))

    # ax2.vlines(x=env.rx[idx[0]], ymin=env.ry[idx[0]]-1, ymax=env.ry[idx[0]]+1, linestyles='dotted', color='red')
    # ax2.text(x=env.rx[idx[0]]-1.2, y=env.ry[idx[0]]+1.3, s='Start/finish', fontsize = 'small', bbox=dict(facecolor='white', edgecolor='black',pad=0.1,boxstyle='round'))

    ax[0].vlines(x=rx[0], ymin=ry[0]-1.2, ymax=ry[0]+1, linestyles='dashed', color='red', label='_nolegend_')


    fig1.tight_layout() 
    plt.figlegend(loc = 'lower center', ncol=2)
    fig1.subplots_adjust(bottom=0.37) 

    plt.show()





if __name__ == '__main__':
    
    # plot_frenet_polynomial()
    plot_frenet_path()