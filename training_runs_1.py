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




agent_name = 'redbull_2'

main_dict = {'name':agent_name, 'max_episodes':20000, 'max_steps':3e6, 'learning_method':'td3', 'runs':1, 'comment':''}

agent_ddpg_dict = {'alpha':0.000025, 'beta':0.00025, 'tau':0.001, 'gamma':0.99, 'max_size':1000000, 'layer1_size':400, 'layer2_size':300, 'batch_size':200}

agent_td3_dict = {'alpha':0.001, 'beta':0.001, 'tau':0.005, 'gamma':0.99, 'update_actor_interval':2, 'warmup':200, 
                    'max_size':1000000, 'layer1_size':400, 'layer2_size':300, 'layer3_size':300, 'batch_size':200, 'noise':0.1}

car_params =   {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145
                , 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2
                , 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}

reward_signal = {'goal_reached':0, 'out_of_bounds':-1, 'max_steps':0, 'collision':-2, 
                    'backwards':-0.01, 'park':-1, 'time_step':-0.01, 'progress':0, 'distance':0.3, 
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
        , 'map_name': 'redbull_ring'
        , 'max_steps': 3000
        , 'control_steps': 10
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


# agent_name = 'redbull_3'
# main_dict['name'] = agent_name
# env_dict['control_steps'] = 5
# agent_td3_dict['layer1_size'] = 600
# agent_td3_dict['layer2_size'] = 450
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

# agent_name = 'redbull_4'
# main_dict['name'] = agent_name
# env_dict['control_steps'] = 10
# agent_td3_dict['layer1_size'] = 650
# agent_td3_dict['layer2_size'] = 450
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

# agent_name = 'redbull_5'
# main_dict['name'] = agent_name
# env_dict['control_steps'] = 5
# agent_td3_dict['layer1_size'] = 400
# agent_td3_dict['layer2_size'] = 300
# action_space_dict['vel_select'] = [2,3]
# env_dict['reward_signal']['distance'] = 0.4
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

# agent_name = 'redbull'
# main_dict['name'] = agent_name
# env_dict['control_steps'] = 5
# agent_td3_dict['layer1_size'] = 400
# agent_td3_dict['layer2_size'] = 300
# action_space_dict['vel_select'] = [3,5]
# env_dict['reward_signal']['distance'] = 0.3
# env_dict['reward_signal']['collision'] = -2
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

# agent_name = 'redbull_6'
# main_dict['name'] = agent_name
# env_dict['control_steps'] = 10
# agent_td3_dict['layer1_size'] = 400
# agent_td3_dict['layer2_size'] = 300
# action_space_dict['vel_select'] = [3,5]
# env_dict['reward_signal']['distance'] = 0.3
# env_dict['reward_signal']['collision'] = -10
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

# agent_name = 'redbull_7'
# main_dict['name'] = agent_name
# env_dict['control_steps'] = 10
# agent_td3_dict['layer1_size'] = 400
# agent_td3_dict['layer2_size'] = 300
# action_space_dict['vel_select'] = [3,5]
# env_dict['reward_signal']['distance'] = 0.25
# env_dict['reward_signal']['collision'] = -10
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

# agent_name = 'redbull_8'
# main_dict['name'] = agent_name
# env_dict['control_steps'] = 10
# agent_td3_dict['layer1_size'] = 400
# agent_td3_dict['layer2_size'] = 300
# action_space_dict['vel_select'] = [3,5]
# env_dict['reward_signal']['distance'] = 0.25
# env_dict['reward_signal']['collision'] = -5
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})


# agent_name = 'redbull_9'
# main_dict['name'] = agent_name
# env_dict['control_steps'] = 10
# agent_td3_dict['layer1_size'] = 400
# agent_td3_dict['layer2_size'] = 300
# action_space_dict['vel_select'] = [2,4]
# env_dict['reward_signal']['distance'] = 0.3
# env_dict['reward_signal']['collision'] = -2
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

# agent_name = 'redbull_10'
# main_dict['name'] = agent_name
# env_dict['control_steps'] = 10
# agent_td3_dict['layer1_size'] = 400
# agent_td3_dict['layer2_size'] = 300
# action_space_dict['vel_select'] = [2,4]
# env_dict['reward_signal']['distance'] = 0.3
# env_dict['reward_signal']['collision'] = -10
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})







# agent_name = 'batch_50'
# main_dict['name'] = agent_name
# main_dict['only_lidar'] = False
# lidar_dict['is_lidar'] = True
# main_dict['lidar_dict'] = lidar_dict
# main_dict['control_steps'] = 20
# main_dict['noise_dict'] = {'xy':0, 'theta':0, 'v':0, 'lidar':0}
# agent_td3_dict['batch_size'] = 50
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

# agent_name = 'batch_100'
# main_dict['name'] = agent_name
# main_dict['only_lidar'] = False
# lidar_dict['is_lidar'] = True
# main_dict['lidar_dict'] = lidar_dict
# main_dict['control_steps'] = 20
# agent_td3_dict['batch_size'] = 100
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

# agent_name = 'batch_150'
# main_dict['name'] = agent_name
# main_dict['only_lidar'] = False
# lidar_dict['is_lidar'] = True
# main_dict['lidar_dict'] = lidar_dict
# main_dict['control_steps'] = 20
# agent_td3_dict['batch_size'] = 150
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

# agent_name = 'batch_200'
# main_dict['name'] = agent_name
# main_dict['only_lidar'] = False
# lidar_dict['is_lidar'] = True
# main_dict['lidar_dict'] = lidar_dict
# main_dict['control_steps'] = 20
# agent_td3_dict['batch_size'] = 200
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

# agent_name = 'batch_200_1'
# main_dict['name'] = agent_name
# main_dict['runs'] = 1
# main_dict['only_lidar'] = False
# lidar_dict['is_lidar'] = True
# main_dict['lidar_dict'] = lidar_dict
# main_dict['control_steps'] = 20
# agent_td3_dict['batch_size'] = 200
# agent_td3_dict['warmup'] = 200
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

# agent_name = 'batch_300'
# main_dict['name'] = agent_name
# main_dict['runs'] = 3
# main_dict['only_lidar'] = False
# lidar_dict['is_lidar'] = True
# main_dict['lidar_dict'] = lidar_dict
# main_dict['control_steps'] = 20
# agent_td3_dict['batch_size'] = 300
# agent_td3_dict['warmup'] = 300
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

# agent_name = 'batch_400'
# main_dict['name'] = agent_name
# main_dict['only_lidar'] = False
# lidar_dict['is_lidar'] = True
# main_dict['lidar_dict'] = lidar_dict
# main_dict['control_steps'] = 20
# agent_td3_dict['batch_size'] = 400
# agent_td3_dict['warmup'] = 400
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

# agent_name = 'batch_600'
# main_dict['name'] = agent_name
# main_dict['only_lidar'] = False
# lidar_dict['is_lidar'] = True
# main_dict['lidar_dict'] = lidar_dict
# main_dict['control_steps'] = 20
# agent_td3_dict['batch_size'] = 600
# agent_td3_dict['warmup'] = 600
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

# agent_name = 'batch_1000'
# main_dict['name'] = agent_name
# main_dict['only_lidar'] = False
# lidar_dict['is_lidar'] = True
# main_dict['lidar_dict'] = lidar_dict
# main_dict['control_steps'] = 20
# agent_td3_dict['batch_size'] = 1000
# agent_td3_dict['warmup'] = 1000
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})


#5 beams
# agent_name = 'lidar_5'
# main_dict['name'] = agent_name
# main_dict['only_lidar'] = False
# lidar_dict['is_lidar'] = True
# lidar_dict['n_beams'] = 5
# env_dict['lidar_dict'] = lidar_dict
# env_dict['noise_dict'] = {'xy':0, 'theta':0, 'v':0, 'lidar':0}
# env_dict['control_steps'] = 20
# agent_td3_dict['batch_size'] = 150
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})


# agent_name = 'lidar_10'
# main_dict['name'] = agent_name
# main_dict['only_lidar'] = False
# lidar_dict['is_lidar'] = True
# lidar_dict['n_beams'] = 10
# env_dict['lidar_dict'] = lidar_dict
# env_dict['control_steps'] = 20
# agent_td3_dict['batch_size'] = 150
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})


# agent_name = 'lidar_20'
# main_dict['name'] = agent_name
# main_dict['only_lidar'] = False
# lidar_dict['is_lidar'] = True
# lidar_dict['n_beams'] = 20
# env_dict['lidar_dict'] = lidar_dict
# env_dict['control_steps'] = 20
# agent_td3_dict['batch_size'] = 150
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})


# agent_name = 'lidar_50'
# main_dict['name'] = agent_name
# main_dict['only_lidar'] = False
# lidar_dict['is_lidar'] = True
# lidar_dict['n_beams'] = 50
# env_dict['lidar_dict'] = lidar_dict
# env_dict['control_steps'] = 20
# agent_td3_dict['batch_size'] = 150
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})


# agent_name = 'lidar_100'
# main_dict['name'] = agent_name
# main_dict['only_lidar'] = False
# lidar_dict['is_lidar'] = True
# lidar_dict['n_beams'] = 100
# env_dict['lidar_dict'] = lidar_dict
# env_dict['control_steps'] = 20
# agent_td3_dict['batch_size'] = 150
# agent_td3_dict['warmup'] = 150
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

# agent_name = 'lidar_200'
# main_dict['name'] = agent_name
# main_dict['only_lidar'] = False
# lidar_dict['is_lidar'] = True
# lidar_dict['n_beams'] = 100
# env_dict['lidar_dict'] = lidar_dict
# env_dict['control_steps'] = 20
# agent_td3_dict['batch_size'] = 200
# agent_td3_dict['warmup'] = 200
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

#only LiDAR
# agent_name = 'only_LiDAR'
# main_dict['name'] = agent_name
# env_dict['only_lidar'] = True
# lidar_dict['is_lidar'] = True
# env_dict['lidar_dict'] = lidar_dict
# env_dict['noise_dict'] = {'xy':0, 'theta':0, 'v':0, 'lidar':0}
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})


# #Only pose
# agent_name = 'only_pose'
# main_dict['name'] = agent_name
# main_dict['only_lidar'] = False
# lidar_dict['is_lidar'] = False
# env_dict['lidar_dict'] = lidar_dict
# env_dict['noise_dict'] = {'xy':0, 'theta':0, 'v':0, 'lidar':0}
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})


# agent_name = 'f_agent_3'
# main_dict['name'] = agent_name
# lidar_dict['is_lidar'] = True
# lidar_dict['n_beams'] = 10
# env_dict['lidar_dict'] = lidar_dict
# env_dict['control_steps'] = 33
# agent_td3_dict['batch_size'] = 150
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

# agent_name = 'f_agent_5'
# main_dict['name'] = agent_name
# env_dict['control_steps'] = 20
# agent_td3_dict['batch_size'] = 150
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

# agent_name = 'f_agent_10'
# main_dict['name'] = agent_name
# env_dict['control_steps'] = 10
# agent_td3_dict['batch_size'] = 150
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

# agent_name = 'f_agent_20'
# main_dict['name'] = agent_name
# env_dict['control_steps'] = 5
# agent_td3_dict['batch_size'] = 150
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})

# agent_name = 'f_agent_50'
# main_dict['name'] = agent_name
# env_dict['control_steps'] = 2
# agent_td3_dict['batch_size'] = 150
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True, noise={'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01})





















# agent_names =  ['batch_size_1', 'sample_5hz_batch_140_noise']
# agent_names =  ['batch_180',  'batch_220']
# agent_names =  ['sample_5hz_batch_140_noise']
# agent_names =  ['time_steps']
# agent_names = ['lidar_5', 'lidar_10', 'lidar_20', 'lidar_50', 'lidar_100', 'lidar_200']
# agent_names = ['train_noise']
# agent_names = ['batch_150','train_noise']
# agent_names = ['batch_400']
# agent_names = ['lidar_200']
# agent_names = ['only_LiDAR', 'only_pose', 'batch_150']
# agent_names = ['only_LiDAR']
# agent_names = ['only_pose']
# agent_names = ['time_steps']
# agent_names = ['sample_3hz', 'sample_5hz', 'sample_10hz', 'sample_20hz', 'sample_50hz']
# agent_names = ['porto_ete_v5_gamma_0','porto_ete_v5_gamma_1', 'porto_ete_v5_gamma_2', 'porto_ete_v5_r_collision_5', 'porto_ete_v5_gamma_4']
# agent_names = ['porto_ete_v5_gamma_2']
# agent_names = ['porto_ete_v5_alpha_0', 'porto_ete_v5_r_collision_5', 'porto_ete_v5_alpha_1']
# agent_names = ['redbull_1']
# agent_names = ['f_agent_3', 'f_agent_5', 'f_agent_10', 'f_agent_20', 'f_agent_50']
# agent_names = ['redbull']

# legend = ['no noise', 'noise']
# legend = ['180', '220']
# legend = ['5', '10', '20']
legend = ['Trained without noise', 'Trained with noise']
legend_title = ''
ns=[1,0,0,0,0,0]

# agent_names = ['porto_ete_v5_r_collision_5']
# legend = []
# legend_title = ''
# ns=[0]






# display_results_multiple.learning_curve_lap_time_average(agent_names, legend, legend_title, ns)
# display_results_multiple.learning_curve_reward_average(agent_names, legend, legend_title)

# for agent_name in agent_names:
#     print('------------------------------' + '\n' + agent_name + '\n' + '------------------------------')
#     display_results_multiple.display_train_parameters(agent_name=agent_name)

# for agent_name in agent_names:
#     print('------------------------------' + '\n' + agent_name + '\n' + '------------------------------')
#     display_results_multiple.display_lap_results(agent_name=agent_name)


# mismatch_parameters = [['C_Sr', 'mu'], ['C_Sr', 'mu']]
# frac_vary = [[0, 0], [0, 0]]
# noise_dicts = [{'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}, {'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}, {'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}, {'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}]
# # start_condition = {'x':10, 'y':4.5, 'v':3, 'theta':np.pi, 'delta':0, 'goal':0}



# Columbia
# start_condition = {'x':5.7, 'y':7.25, 'v':3, 'theta':0, 'delta':0, 'goal':0}
start_condition = []


# NB!!!! Error: Path is junk when no mismatch is present, when displaying 2 agents

# display_results_multiple.display_moving_agent(agent_names=agent_names, ns=ns, legend_title=legend_title,          
#                                              legend=legend, mismatch_parameters=mismatch_parameters, frac_vary=frac_vary, noise_dicts=noise_dicts,
#                                              start_condition=start_condition)

# display_results_multiple.display_path_multiple(agent_names=agent_names, ns=ns, legend_title=legend_title,          
#                                              legend=legend, mismatch_parameters=mismatch_parameters, frac_vary=frac_vary, noise_dicts=noise_dicts,
#                                              start_condition=start_condition)

# display_results_multiple.display_path_mismatch_multiple_by_state(agent_names=agent_names, ns=ns, legend_title=legend_title,          
#                                              legend=legend, mismatch_parameters=mismatch_parameters, frac_vary=frac_vary, noise_dicts=noise_dicts,
#                                              start_condition=start_condition)

