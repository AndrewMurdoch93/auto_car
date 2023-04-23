import main_multiple
import functions
from environment import environment
import display_results_multiple
import agent_dqn
import agent_reinforce
import agent_actor_critic
import agent_actor_critic_continuous
import agent_dueling_dqn
import agent_dueling_ddqn
import agent_rainbow
import agent_ddpg
import agent_td3

import pickle
import sys
import numpy as np
import random
import os




agent_name = 'eval'

main_dict = {'name':agent_name, 'max_episodes':10000, 'max_steps':3e6, 'learning_method':'td3', 'runs':3, 'comment':''}

agent_ddpg_dict = {'alpha':0.000025, 'beta':0.00025, 'tau':0.001, 'gamma':0.99, 'max_size':1000000, 'layer1_size':400, 'layer2_size':300, 'batch_size':64}

agent_td3_dict = {'alpha':0.001, 'beta':0.001, 'tau':0.005, 'gamma':0.99, 'update_actor_interval':2, 'warmup':150, 
                    'max_size':1000000, 'layer1_size':400, 'layer2_size':300, 'layer3_size':300, 'batch_size':150, 'noise':0.1}

car_params =   {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145
                , 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2
                , 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}

reward_signal = {'goal_reached':0, 'out_of_bounds':-1, 'max_steps':0, 'collision':-10, 
                    'backwards':-0.01, 'park':-1, 'time_step':-0.01, 'progress':0, 'distance':0.25, 
                    'max_progress':0}    

action_space_dict = {'action_space': 'continuous', 'vel_select':[3,5], 'R_range':[2]}

#action_space_dict = {'action_space': 'discrete', 'n_waypoints': 10, 'vel_select':[7], 'R_range':[6]}

steer_control_dict = {'steering_control': False, 'wpt_arc':np.pi/2, 'track_width':1.2}

if  steer_control_dict['steering_control'] == True:
    steer_control_dict['path_strategy'] = 'circle'  #circle or linear or polynomial or gradient
    steer_control_dict['control_strategy'] = 'pure_pursuit'  #pure_pursuit or stanley

    if steer_control_dict['control_strategy'] == 'pure_pursuit':
        steer_control_dict['track_dict'] = {'k':0.1, 'Lfc':1}
    if steer_control_dict['control_strategy'] == 'stanley':
        steer_control_dict['track_dict'] = {'l_front': car_params['lf'], 'k':5, 'max_steer':car_params['s_max']}

lidar_dict = {'is_lidar':True, 'lidar_res':0.1, 'n_beams':10, 'max_range':20, 'fov':np.pi}

# noise_dict = {'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}
noise_dict = {'xy':0, 'theta':0, 'v':0, 'lidar':0}

env_dict = {'sim_conf': functions.load_config(sys.path[0], "config")
        , 'save_history': False
        , 'map_name': 'porto_1'
        , 'max_steps': 3000
        , 'control_steps': 20
        , 'display': False
        , 'velocity_control': False
        , 'velocity_gain':1
        , 'steer_control_dict': steer_control_dict
        , 'car_params':car_params
        , 'reward_signal':reward_signal
        , 'lidar_dict':lidar_dict
        , 'only_lidar':False
        , 'action_space_dict':action_space_dict
        , 'noise_dict':noise_dict
        } 

n_test=100
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

agent_name = 'train_noise'
main_dict['name'] = agent_name
main_dict['only_lidar'] = False
lidar_dict['is_lidar'] = True
main_dict['lidar_dict'] = lidar_dict
main_dict['control_steps'] = 20
main_dict['noise_dict'] = {'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}
agent_td3_dict['batch_size'] = 150
a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
a.train()
main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})


agent_name = 'batch_50'
main_dict['name'] = agent_name
main_dict['only_lidar'] = False
lidar_dict['is_lidar'] = True
main_dict['lidar_dict'] = lidar_dict
main_dict['control_steps'] = 20
main_dict['noise_dict'] = {'xy':0, 'theta':0, 'v':0, 'lidar':0}
agent_td3_dict['batch_size'] = 50
a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
a.train()
main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

agent_name = 'batch_100'
main_dict['name'] = agent_name
main_dict['only_lidar'] = False
lidar_dict['is_lidar'] = True
main_dict['lidar_dict'] = lidar_dict
main_dict['control_steps'] = 20
agent_td3_dict['batch_size'] = 100
a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
a.train()
main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

agent_name = 'batch_150'
main_dict['name'] = agent_name
main_dict['only_lidar'] = False
lidar_dict['is_lidar'] = True
main_dict['lidar_dict'] = lidar_dict
main_dict['control_steps'] = 20
agent_td3_dict['batch_size'] = 150
a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
a.train()
main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

agent_name = 'batch_200'
main_dict['name'] = agent_name
main_dict['only_lidar'] = False
lidar_dict['is_lidar'] = True
main_dict['lidar_dict'] = lidar_dict
main_dict['control_steps'] = 20
agent_td3_dict['batch_size'] = 200
a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
a.train()
main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

# agent_name = 'f_agent_3'
# main_dict['name'] = agent_name
# main_dict['only_lidar'] = False
# lidar_dict['is_lidar'] = True
# main_dict['lidar_dict'] = lidar_dict
# main_dict['control_steps'] = 33
# agent_td3_dict['batch_size'] = 150
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

# agent_name = 'f_agent_10'
# main_dict['name'] = agent_name
# main_dict['only_lidar'] = False
# lidar_dict['is_lidar'] = True
# main_dict['lidar_dict'] = lidar_dict
# main_dict['control_steps'] = 10
# agent_td3_dict['batch_size'] = 150
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

# agent_name = 'f_agent_20'
# main_dict['name'] = agent_name
# main_dict['only_lidar'] = False
# lidar_dict['is_lidar'] = True
# main_dict['lidar_dict'] = lidar_dict
# main_dict['control_steps'] = 5
# agent_td3_dict['batch_size'] = 150
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})


# #5 beams
# agent_name = 'lidar_5'
# main_dict['name'] = agent_name
# main_dict['only_lidar'] = False
# lidar_dict['is_lidar'] = True
# lidar_dict['n_beams'] = 5
# main_dict['lidar_dict'] = lidar_dict
# main_dict['control_steps'] = 20
# agent_td3_dict['batch_size'] = 150
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})


# agent_name = 'lidar_10'
# main_dict['name'] = agent_name
# main_dict['only_lidar'] = False
# lidar_dict['is_lidar'] = True
# lidar_dict['n_beams'] = 10
# main_dict['lidar_dict'] = lidar_dict
# main_dict['control_steps'] = 20
# agent_td3_dict['batch_size'] = 150
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})


# agent_name = 'lidar_20'
# main_dict['name'] = agent_name
# main_dict['only_lidar'] = False
# lidar_dict['is_lidar'] = True
# lidar_dict['n_beams'] = 20
# main_dict['lidar_dict'] = lidar_dict
# main_dict['control_steps'] = 20
# agent_td3_dict['batch_size'] = 150
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})


# agent_name = 'lidar_50'
# main_dict['name'] = agent_name
# main_dict['only_lidar'] = False
# lidar_dict['is_lidar'] = True
# lidar_dict['n_beams'] = 50
# main_dict['lidar_dict'] = lidar_dict
# main_dict['control_steps'] = 20
# agent_td3_dict['batch_size'] = 150
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})


# #only LiDAR
# agent_name = 'only_LiDAR'
# main_dict['name'] = agent_name
# main_dict['only_lidar'] = True
# lidar_dict['is_lidar'] = True
# main_dict['lidar_dict'] = lidar_dict
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})


# #Only pose
# agent_name = 'only_pose'
# main_dict['name'] = agent_name
# main_dict['only_lidar'] = False
# lidar_dict['is_lidar'] = False
# main_dict['lidar_dict'] = lidar_dict
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})



