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




agent_name = 'end_to_end_display'

main_dict = {'name':agent_name, 'max_episodes':10000, 'max_steps':3e6, 'learning_method':'td3', 'runs':3, 'comment':''}

agent_dqn_dict = {'gamma':0.99, 'epsilon':1, 'eps_end':0.01, 'eps_dec':1/1000, 'lr':0.001, 'batch_size':64, 'max_mem_size':500000, 
                'fc1_dims': 64, 'fc2_dims': 64, 'fc3_dims':64}

agent_dueling_dqn_dict = {'gamma':0.99, 'epsilon':1, 'eps_end':0.01, 'eps_dec':1/1000, 'alpha':0.001, 'batch_size':64, 'max_mem_size':500000, 
                        'replace':100, 'fc1_dims':64, 'fc2_dims':64}

agent_dueling_ddqn_dict = {'gamma':0.99, 'epsilon':1, 'eps_end':0.01, 'eps_dec':1/1000, 'alpha':0.001, 'batch_size':64, 'max_mem_size':500000, 
                        'replace':100, 'fc1_dims':64, 'fc2_dims':64}

agent_rainbow_dict = {'gamma':0.99, 'epsilon':1, 'eps_end':0.01, 'eps_dec':1/1000, 'alpha':0.001/4, 'batch_size':64, 'max_mem_size':500000, 
                        'replay_alpha':0.6, 'replay_beta_0':0.7, 'replace':100, 'fc1_dims':100, 'fc2_dims':100}

agent_reinforce_dict = {'alpha':0.001, 'gamma':0.99, 'fc1_dims':256, 'fc2_dims':256}

agent_actor_critic_sep_dict = {'gamma':0.99, 'actor_dict': {'alpha':0.00001, 'fc1_dims':2048, 'fc2_dims':512}, 'critic_dict': {'alpha': 0.00001, 'fc1_dims':2048, 'fc2_dims':512}}

agent_actor_critic_com_dict = {'gamma':0.99, 'alpha':0.00001, 'fc1_dims':2048, 'fc2_dims':512}

agent_actor_critic_cont_dict = {'gamma':0.99, 'alpha':0.000005, 'beta':0.00001, 'fc1_dims':256, 'fc2_dims':256}

agent_ddpg_dict = {'alpha':0.000025, 'beta':0.00025, 'tau':0.001, 'gamma':0.99, 'max_size':1000000, 'layer1_size':400, 'layer2_size':300, 'batch_size':64}

agent_td3_dict = {'alpha':0.001, 'beta':0.001, 'tau':0.005, 'gamma':0.99, 'update_actor_interval':2, 'warmup':100, 
                    'max_size':1000000, 'layer1_size':400, 'layer2_size':300, 'layer3_size':300, 'batch_size':100, 'noise':0.1}

car_params =   {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145
                , 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2
                , 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}

reward_signal = {'goal_reached':0, 'out_of_bounds':-1, 'max_steps':0, 'collision':-2, 
                    'backwards':-0.01, 'park':-1, 'time_step':-0.01, 'progress':0, 'distance':0.2, 
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
        , 'display': True
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

#Training with v=6
if True:
    # agent_name = 'porto_ete_LiDAR_3'
    # main_dict['name'] = agent_name
    # env_dict['lidar_dict']['is_lidar'] = True
    # env_dict['lidar_dict']['n_beams'] = 3
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_LiDAR_10'
    # main_dict['name'] = agent_name
    # env_dict['lidar_dict']['n_beams'] = 10
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_LiDAR_20'
    # main_dict['name'] = agent_name
    # env_dict['lidar_dict']['n_beams'] = 20
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_LiDAR_50'
    # main_dict['name'] = agent_name
    # env_dict['lidar_dict']['n_beams'] = 50
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_cs_1'
    # main_dict['name'] = agent_name
    # env_dict['lidar_dict']['n_beams'] = 10
    # env_dict['control_steps'] = 1
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_cs_5'
    # main_dict['name'] = agent_name
    # env_dict['control_steps'] = 5
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_cs_10'
    # main_dict['name'] = agent_name
    # env_dict['control_steps'] = 10
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_cs_15'
    # main_dict['name'] = agent_name
    # env_dict['control_steps'] = 15
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_cs_25'
    # main_dict['name'] = agent_name
    # env_dict['control_steps'] = 25
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_v_5'
    # main_dict['name'] = agent_name
    # env_dict['control_steps'] = 20
    # env_dict['action_space_dict']['vel_select'] = [3,5]
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_v_7'
    # main_dict['name'] = agent_name
    # env_dict['action_space_dict']['vel_select'] = [3,7]
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_v_8'
    # main_dict['name'] = agent_name
    # env_dict['action_space_dict']['vel_select'] = [3,8]
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_only_LiDAR'
    # main_dict['name'] = agent_name
    # env_dict['only_lidar'] = True
    # env_dict['lidar_dict']['n_beams'] = 20
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)



    # agent_name = 'porto_ete_r_0'
    # main_dict['name'] = agent_name
    # env_dict['only_lidar'] = False
    # env_dict['lidar_dict']['is_lidar'] = True
    # env_dict['lidar_dict']['n_beams'] = 10
    # env_dict['control_steps'] = 20
    # env_dict['action_space_dict']['vel_select'] = [3,6]
    # env_dict['reward_signal']['distance'] = 1
    # env_dict['reward_signal']['time_step'] = 0
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_r_1'
    # main_dict['name'] = agent_name
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['reward_signal']['distance'] = 1
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_r_2'
    # main_dict['name'] = agent_name
    # env_dict['reward_signal']['distance'] = 0.7
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_r_3'
    # main_dict['name'] = agent_name
    # env_dict['reward_signal']['distance'] = 0.5
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_r_4'
    # main_dict['name'] = agent_name
    # env_dict['reward_signal']['distance'] = 0.1
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_r_5'
    # main_dict['name'] = agent_name
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['collision'] = -0.5
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_r_6'
    # main_dict['name'] = agent_name
    # env_dict['reward_signal']['collision'] = -2
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)


    # agent_name = 'porto_ete_r_7'
    # main_dict['name'] = agent_name
    # env_dict['only_lidar'] = False
    # env_dict['lidar_dict']['is_lidar'] = True
    # env_dict['lidar_dict']['n_beams'] = 10
    # env_dict['control_steps'] = 20
    # env_dict['action_space_dict']['vel_select'] = [3,6]
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['reward_signal']['collision'] = 0
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_ddpg_test'
    # main_dict['name'] = agent_name
    # main_dict['learning_method'] = 'ddpg'
    # env_dict['only_lidar'] = False
    # env_dict['lidar_dict']['is_lidar'] = True
    # env_dict['lidar_dict']['n_beams'] = 10
    # env_dict['control_steps'] = 20
    # env_dict['action_space_dict']['vel_select'] = [3,6]
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['reward_signal']['collision'] = -1
    # a = main_multiple.trainingLoop(main_dict, agent_ddpg_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)


    # agent_name = 'porto_ete_actor_critic_cont'
    # main_dict['name'] = agent_name
    # main_dict['learning_method'] = 'actor_critic_cont'
    # env_dict['only_lidar'] = False
    # env_dict['lidar_dict']['is_lidar'] = True
    # env_dict['lidar_dict']['n_beams'] = 10
    # env_dict['control_steps'] = 20
    # env_dict['action_space_dict']['vel_select'] = [3,6]
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['reward_signal']['collision'] = -1
    # a = main_multiple.trainingLoop(main_dict, agent_actor_critic_cont_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # collision = 9.3 agent!!!!
    # agent_name = 'porto_ete_actor_critic_cont'
    # main_dict['name'] = agent_name
    # main_dict['learning_method'] = 'td3'
    # env_dict['only_lidar'] = False
    # env_dict['lidar_dict']['is_lidar'] = True
    # env_dict['lidar_dict']['n_beams'] = 10
    # env_dict['control_steps'] = 20
    # env_dict['action_space_dict']['vel_select'] = [3,6]
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['reward_signal']['collision'] = -9.3
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    pass

#training with v=5 
if True:
    # agent_name = 'porto_ete_v5_cs_10'
    # main_dict['name'] = agent_name
    # env_dict['control_steps'] = 10
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_v5_cs_15'
    # main_dict['name'] = agent_name
    # env_dict['control_steps'] = 15
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_v5_cs_25'
    # main_dict['name'] = agent_name
    # env_dict['control_steps'] = 25
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)


    # agent_name = 'porto_ete_v5_r_time_0'
    # main_dict['name'] = agent_name
    # env_dict['control_steps'] = 20
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['time_step'] = 0
    # env_dict['reward_signal']['collision'] = -2
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_v5_r_dist_1'
    # main_dict['name'] = agent_name
    # env_dict['reward_signal']['distance'] = 0.1
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['reward_signal']['collision'] = -2
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_v5_r_dist_2'
    # main_dict['name'] = agent_name
    # env_dict['reward_signal']['distance'] = 0.5
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['reward_signal']['collision'] = -2
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_v5_r_dist_3'
    # main_dict['name'] = agent_name
    # env_dict['reward_signal']['distance'] = 0.7
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['reward_signal']['collision'] = -2
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_v5_r_dist_4'
    # main_dict['name'] = agent_name
    # env_dict['reward_signal']['distance'] = 1
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['reward_signal']['collision'] = -2
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_v5_r_collision_0'
    # main_dict['name'] = agent_name
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['reward_signal']['collision'] = 0
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_v5_r_collision_1'
    # main_dict['name'] = agent_name
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['reward_signal']['collision'] = -1
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_v5_r_collision_2'
    # main_dict['name'] = agent_name
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['reward_signal']['collision'] = -4
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)


    # agent_name = 'porto_ete_v5_r_collision_3'
    # main_dict['name'] = agent_name
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['reward_signal']['collision'] = -6
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_v5_r_collision_4'
    # main_dict['name'] = agent_name
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['reward_signal']['collision'] = -8
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_v5_r_collision_5'
    # main_dict['name'] = agent_name
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['reward_signal']['collision'] = -10
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_v5_r_collision_5_attempt_2'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 3e6
    # env_dict['velocity_control'] = False
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['reward_signal']['collision'] = -10
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_v5_r_dist_02_attempt_2'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 3e6
    # env_dict['velocity_control'] = False
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['reward_signal']['distance'] = 0.2
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['reward_signal']['collision'] = -2
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)


    # agent_name = 'porto_ete_v5_r_collision'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 2e6
    # env_dict['velocity_control'] = False
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['reward_signal']['distance'] = 0.2
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['reward_signal']['collision'] = -0.5
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)


    # agent_name = 'porto_ete_v5_r_collision_7'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 2e6
    # env_dict['velocity_control'] = False
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['reward_signal']['distance'] = 0.2
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['reward_signal']['collision'] = -20
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_v5_r_collision_8'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 2e6
    # env_dict['velocity_control'] = False
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['reward_signal']['distance'] = 0.7
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['reward_signal']['collision'] = -1
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)






    # agent_name = 'porto_ete_v5_gamma_0'
    # main_dict['name'] = agent_name
    # agent_td3_dict['gamma'] = 0.9
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_v5_gamma_1'
    # main_dict['name'] = agent_name
    # agent_td3_dict['gamma'] = 0.95
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_v5_gamma_2'
    # main_dict['name'] = agent_name
    # agent_td3_dict['gamma'] = 0.98
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_v5_gamma_3'
    # main_dict['name'] = agent_name
    # agent_td3_dict['gamma'] = 0.999
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_v5_gamma_4'
    # main_dict['name'] = agent_name
    # agent_td3_dict['gamma'] = 1
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_v5_alpha_0'
    # main_dict['name'] = agent_name
    # agent_td3_dict['alpha'] = 0.0001
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_v5_alpha_1'
    # main_dict['name'] = agent_name
    # agent_td3_dict['alpha'] = 0.001*2
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)





    pass

#training all tracks with v=5
if True:
    # agent_name = 'circle_ete'
    # main_dict['name'] = agent_name
    # main_dict['map_name']='circle'
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'berlin_ete'
    # main_dict['name'] = agent_name
    # main_dict['map_name']='berlin'
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'torino_ete'
    # main_dict['name'] = agent_name
    # main_dict['map_name']='torino'
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'redbull_ring_ete'
    # main_dict['name'] = agent_name
    # main_dict['map_name']='redbull_ring'
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)


    # steer_control_dict['steering_control'] = True
    # steer_control_dict['path_strategy'] = 'circle'  #circle or linear or polynomial or gradient
    # steer_control_dict['control_strategy'] = 'pure_pursuit'  #pure_pursuit or stanley
    # steer_control_dict['track_dict'] = {'k':0.1, 'Lfc':1}

    # agent_name = 'porto_pete_s'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict'] = steer_control_dict
    # env_dict['map_name']='porto_1'
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'columbia_pete_s'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = True
    # main_dict['map_name']='columbia_1'
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'circle_pete_s'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = True
    # main_dict['map_name']='circle'
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'berlin_pete_s'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = True
    # main_dict['map_name']='berlin'
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'torino_pete_s'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = True
    # main_dict['map_name']='torino'
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'redbull_ring_pete_s'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = True
    # main_dict['map_name']='redbull_ring'
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)




    # agent_name = 'porto_pete_v'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # env_dict['map_name']='porto_1'
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'columbia_pete_v'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # main_dict['map_name']='columbia_1'
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'circle_pete_v'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # main_dict['map_name']='circle'
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'berlin_pete_v'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # main_dict['map_name']='berlin'
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'torino_pete_v'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # main_dict['map_name']='torino'
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'redbull_ring_pete_v'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # main_dict['map_name']='redbull_ring'
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)


    # agent_name = 'porto_pete_sv'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['velocity_control'] = True
    # env_dict['map_name']='porto_1'
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'columbia_pete_sv'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['velocity_control'] = True
    # main_dict['map_name']='columbia_1'
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'circle_pete_sv'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['velocity_control'] = True
    # main_dict['map_name']='circle'
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'berlin_pete_sv'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['velocity_control'] = True
    # main_dict['map_name']='berlin'
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'torino_pete_sv'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['velocity_control'] = True
    # main_dict['map_name']='torino'
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'redbull_ring_pete_sv'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['velocity_control'] = True
    # main_dict['map_name']='redbull_ring'
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    pass

# Tuning parameters for steering control porto
if True:
    # agent_name = 'porto_pete_s_r_dist_0'
    # main_dict['name'] = agent_name
    # env_dict['map_name']='porto_1'
    # env_dict['reward_signal']['collision'] = -1
    # env_dict['reward_signal']['distance'] = 0.15
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_s_r_dist_1'
    # main_dict['name'] = agent_name
    # env_dict['reward_signal']['collision'] = -1
    # env_dict['reward_signal']['distance'] = 0.2
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=False)

    # agent_name = 'porto_pete_s_r_dist_2'
    # main_dict['name'] = agent_name
    # env_dict['reward_signal']['collision'] = -1
    # env_dict['reward_signal']['distance'] = 0.25
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_s_r_dist_3'
    # main_dict['name'] = agent_name
    # env_dict['reward_signal']['collision'] = -1
    # env_dict['reward_signal']['distance'] = 0.4
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)
    
    # agent_name = 'porto_pete_s_r_dist_4'
    # main_dict['name'] = agent_name
    # env_dict['map_name']='porto_1'
    # env_dict['reward_signal']['collision'] = -1
    # env_dict['reward_signal']['distance'] = 0.5
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)
    
    # agent_name = 'porto_pete_s_r_dist_5'
    # main_dict['name'] = agent_name
    # env_dict['map_name']='porto_1'
    # env_dict['reward_signal']['collision'] = -1
    # env_dict['reward_signal']['distance'] = 0.3
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)
    

    # agent_name = 'porto_pete_s_r_collision_0'
    # main_dict['name'] = agent_name
    # env_dict['map_name']='porto_1'
    # env_dict['reward_signal']['collision'] = -2
    # env_dict['reward_signal']['distance'] = 0.2
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_s_r_collision_1'
    # main_dict['name'] = agent_name
    # env_dict['map_name']='porto_1'
    # env_dict['reward_signal']['collision'] = -4
    # env_dict['reward_signal']['distance'] = 0.2
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)


    # agent_name = 'porto_pete_s_r_collision_2'
    # main_dict['name'] = agent_name
    # env_dict['map_name']='porto_1'
    # env_dict['reward_signal']['collision'] = -8
    # env_dict['reward_signal']['distance'] = 0.2
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)


    # agent_name = 'porto_pete_s_lfc_0'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['track_dict']['Lfc'] = 0.5
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_s_lfc_1'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['track_dict']['Lfc'] = 1.5
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_s_lfc_2'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['track_dict']['Lfc'] = 2
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_s_lfc_3'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['track_dict']['Lfc'] = 2.5
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    pass

#Tune parameters for velocity control
if True:
    
    # agent_name = 'porto_pete_v_k_0'
    # main_dict['name'] = agent_name
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 1
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_v_k_1_attempt_2'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 3e6 
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.2
    # env_dict['reward_signal']['collision'] = -2
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)



    # agent_name = 'porto_pete_v_r_dist_025'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 1
    # env_dict['reward_signal']['distance'] = 0.25
    # env_dict['reward_signal']['collision'] = -2
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_v_r_dist_03'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 1
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['collision'] = -2
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_v_r_collision_0'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 1
    # env_dict['reward_signal']['distance'] = 0.2
    # env_dict['reward_signal']['collision'] = -4
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_v_r_collision_1'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 1
    # env_dict['reward_signal']['distance'] = 0.2
    # env_dict['reward_signal']['collision'] = -6
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_v_r_collision_2'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 1
    # env_dict['reward_signal']['distance'] = 0.2
    # env_dict['reward_signal']['collision'] = -8
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_v_r_collision_3'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 1
    # env_dict['reward_signal']['distance'] = 0.25
    # env_dict['reward_signal']['collision'] = -4
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_v_r_collision_4'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 1
    # env_dict['reward_signal']['distance'] = 0.25
    # env_dict['reward_signal']['collision'] = -8
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_v_r_collision_5'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 1
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['collision'] = -4
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_v_r_collision_6'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 1
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['collision'] = -8
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_v_r_collision_6_attempt_2'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 3e6 
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 1
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['collision'] = -8
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)








    # agent_name = 'porto_pete_v_k2_r_dist_025'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.25
    # env_dict['reward_signal']['collision'] = -2
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_v_k2_r_dist_03'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['collision'] = -2
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_v_k2_r_collision_0'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.2
    # env_dict['reward_signal']['collision'] = -4
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_v_k2_r_collision_1'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.2
    # env_dict['reward_signal']['collision'] = -6
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_v_k2_r_collision_2'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.2
    # env_dict['reward_signal']['collision'] = -8
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_v_k2_r_collision_3'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.25
    # env_dict['reward_signal']['collision'] = -4
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_v_k2_r_collision_4'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.25
    # env_dict['reward_signal']['collision'] = -8
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_v_k2_r_collision_5'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['collision'] = -4
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_v_k2_r_collision_6'
    # main_dict['name'] = agent_name
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['collision'] = -8
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    pass


#Tune parameters for steering and velocity control
if True:

    # agent_name = 'porto_pete_sv_c_r_0'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 3e6
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['steer_control_dict']['path_strategy'] = 'circle'
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.2
    # env_dict['reward_signal']['collision'] = -2
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_sv_c_r_1'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 3e6
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['steer_control_dict']['path_strategy'] = 'circle'
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['collision'] = -8
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_sv_p_r_0'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 3e6
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['steer_control_dict']['path_strategy'] = 'polynomial'
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.2
    # env_dict['reward_signal']['collision'] = -2
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_sv_p_r_1'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 3e6
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['steer_control_dict']['path_strategy'] = 'polynomial'
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['collision'] = -8
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_s_p'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 3e6
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['steer_control_dict']['path_strategy'] = 'polynomial'
    # env_dict['velocity_control'] = False
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.2
    # env_dict['reward_signal']['collision'] = -2
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # noise = {'xy':0, 'theta':0, 'v':0, 'lidar':0}
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True, noise=noise)


    # agent_name = 'porto_pete_sv_c_r_3'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 5e6
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['steer_control_dict']['path_strategy'] = 'circle'
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.2
    # env_dict['reward_signal']['collision'] = -2
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_sv_c_r_4'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 5e6
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['steer_control_dict']['path_strategy'] = 'circle'
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.2
    # env_dict['reward_signal']['collision'] = -4
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_sv_c_r_5'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 5e6
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['steer_control_dict']['path_strategy'] = 'circle'
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.2
    # env_dict['reward_signal']['collision'] = -6
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_sv_c_r_6'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 5e6
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['steer_control_dict']['path_strategy'] = 'circle'
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.2
    # env_dict['reward_signal']['collision'] = -8
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_sv_c_r_7'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 5e6
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['steer_control_dict']['path_strategy'] = 'circle'
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['collision'] = -4
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_sv_c_r_7'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 5e6
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['steer_control_dict']['path_strategy'] = 'circle'
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['collision'] = -6
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_sv_c_r_8'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 5e6
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['steer_control_dict']['path_strategy'] = 'circle'
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['collision'] = -8
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)
    pass


# Training with noise
if True:
    # agent_name = 'porto_ete_v5_r_collision_5_noise'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 3e6
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['steer_control_dict']['path_strategy'] = 'circle'
    # env_dict['velocity_control'] = False
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['collision'] = -10
    # env_dict['noise'] = {'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)
    # main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True, noise=env_dict['noise'])

    # agent_name = 'porto_pete_s_r_collision_0_noise'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 3e6
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['steer_control_dict']['path_strategy'] = 'circle'
    # env_dict['velocity_control'] = False
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.2
    # env_dict['reward_signal']['collision'] = -2
    # env_dict['noise'] = {'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)
    # main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True, noise=env_dict['noise'])

    # agent_name = 'porto_pete_s_polynomial_noise'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 3e6
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['steer_control_dict']['path_strategy'] = 'polynomial'
    # env_dict['velocity_control'] = False
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.2
    # env_dict['reward_signal']['collision'] = -2
    # env_dict['noise'] = {'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)
    # main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True, noise=env_dict['noise'])

    # agent_name = 'porto_pete_v_k_1_attempt_2_noise_1'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 3e6
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['steer_control_dict']['path_strategy'] = 'circle'
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.2
    # env_dict['reward_signal']['collision'] = -2
    # env_dict['noise'] = {'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)
    # main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True, noise=env_dict['noise'])

    # agent_name = 'porto_pete_sv_c_r_8_noise'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 5e6
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['steer_control_dict']['path_strategy'] = 'circle'
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['collision'] = -8
    # env_dict['noise'] = {'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)
    # main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True, noise=env_dict['noise'])

    # agent_name = 'porto_pete_sv_p_r_0_noise'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 3e6
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['steer_control_dict']['path_strategy'] = 'polynomial'
    # env_dict['velocity_control'] = True
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.2
    # env_dict['reward_signal']['collision'] = -2
    # env_dict['noise'] = {'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)
    # main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True, noise=env_dict['noise'])

    pass

    # agent_name = 'porto_ete_v5_r_collision_5_noise'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 3e6
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['steer_control_dict']['path_strategy'] = 'circle'
    # env_dict['velocity_control'] = False
    # env_dict['velocity_gain'] = 2
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['collision'] = -10
    # env_dict['noise'] = {'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)
    # main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True, noise=env_dict['noise'])

# Algorithm hyper-parameters
if True: 
    # agent_name = 'target_update_0'
    # main_dict['name'] = agent_name
    # main_dict['max_steps'] = 3e6
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = False
    # env_dict['reward_signal']['distance'] = 0.3
    # env_dict['reward_signal']['collision'] = -10
    # env_dict['noise'] = {'xy':0, 'theta':0, 'v':0, 'lidar':0}
    
    # agent_td3_dict = {'alpha':0.001, 'beta':0.001, 'tau':0.003, 'gamma':0.99, 'update_actor_interval':2, 'warmup':100, 
    #                 'max_size':1000000, 'layer1_size':400, 'layer2_size':300, 'layer3_size':300, 'batch_size':100, 'noise':0.1}
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)
    
    # agent_name = 'target_update_1'
    # main_dict['name'] = agent_name
    # agent_td3_dict = {'alpha':0.001, 'beta':0.001, 'tau':0.007, 'gamma':0.99, 'update_actor_interval':2, 'warmup':100, 
    #                 'max_size':1000000, 'layer1_size':400, 'layer2_size':300, 'layer3_size':300, 'batch_size':100, 'noise':0.1}
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)


    # agent_name = 'update_actor_interval_0'
    # main_dict['name'] = agent_name
    # agent_td3_dict = {'alpha':0.001, 'beta':0.001, 'tau':0.005, 'gamma':0.99, 'update_actor_interval':1, 'warmup':100, 
    #                 'max_size':1000000, 'layer1_size':400, 'layer2_size':300, 'layer3_size':300, 'batch_size':100, 'noise':0.1}
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'update_actor_interval_1'
    # main_dict['name'] = agent_name
    # agent_td3_dict = {'alpha':0.001, 'beta':0.001, 'tau':0.005, 'gamma':0.99, 'update_actor_interval':3, 'warmup':100, 
    #                 'max_size':1000000, 'layer1_size':400, 'layer2_size':300, 'layer3_size':300, 'batch_size':100, 'noise':0.1}
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'batch_size_0'
    # main_dict['name'] = agent_name
    # agent_td3_dict = {'alpha':0.001, 'beta':0.001, 'tau':0.005, 'gamma':0.99, 'update_actor_interval':2, 'warmup':100, 
    #                 'max_size':1000000, 'layer1_size':400, 'layer2_size':300, 'layer3_size':300, 'batch_size':60, 'noise':0.1}
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'batch_size_1'
    # main_dict['name'] = agent_name
    # agent_td3_dict = {'alpha':0.001, 'beta':0.001, 'tau':0.005, 'gamma':0.99, 'update_actor_interval':2, 'warmup':100, 
    #                 'max_size':1000000, 'layer1_size':400, 'layer2_size':300, 'layer3_size':300, 'batch_size':140, 'noise':0.1}
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'explore_policy_0'
    # main_dict['name'] = agent_name
    # agent_td3_dict = {'alpha':0.001, 'beta':0.001, 'tau':0.005, 'gamma':0.99, 'update_actor_interval':2, 'warmup':100, 
    #                 'max_size':1000000, 'layer1_size':400, 'layer2_size':300, 'layer3_size':300, 'batch_size':100, 'noise':0.05}
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'explore_policy_1'
    # main_dict['name'] = agent_name
    # agent_td3_dict = {'alpha':0.001, 'beta':0.001, 'tau':0.005, 'gamma':0.99, 'update_actor_interval':2, 'warmup':100, 
    #                 'max_size':1000000, 'layer1_size':400, 'layer2_size':300, 'layer3_size':300, 'batch_size':100, 'noise':0.2}
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'layer_0'
    # main_dict['name'] = agent_name
    # agent_td3_dict = {'alpha':0.001, 'beta':0.001, 'tau':0.005, 'gamma':0.99, 'update_actor_interval':2, 'warmup':100, 
    #                 'max_size':1000000, 'layer1_size':300, 'layer2_size':200, 'layer3_size':300, 'batch_size':100, 'noise':0.2}
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)

    # agent_name = 'layer_1'
    # main_dict['name'] = agent_name
    # agent_td3_dict = {'alpha':0.001, 'beta':0.001, 'tau':0.005, 'gamma':0.99, 'update_actor_interval':2, 'warmup':100, 
    #                 'max_size':1000000, 'layer1_size':300, 'layer2_size':500, 'layer3_size':400, 'batch_size':100, 'noise':0.2}
    # a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # main_multiple.lap_time_test(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True)
    pass


#tests with v=6
if True:
    # agent_names = ['porto_ete_r_1', 'porto_ete_r_2', 'porto_ete_r_3', 'porto_ete_LiDAR_10', 'porto_ete_r_4']
    # legend = ['1', '0.7', '0.5', '0.3', '0.1']
    # legend_title = ''
    # ns=[0, 0, 0, 0, 0]

    # agent_names = ['porto_ete_r_5', 'porto_ete_r_6']
    # legend = ['-0.5', '2']
    # legend_title = ''
    # ns=[0, 0]

    # agent_names = ['porto_ete_LiDAR_10']
    # legend = ['']
    # legend_title = ''
    # ns=[0]

    # agent_names = ['porto_ete_cs_1', 'porto_ete_cs_5', 'porto_ete_cs_10', 'porto_ete_cs_15', 'porto_ete_LiDAR_10', 'porto_ete_cs_25']
    # legend = ['']
    # legend_title = ''
    # ns=[0, 0, 0, 0, 0, 0]

    # agent_names = ['porto_ete_v_5', 'porto_ete_LiDAR_10', 'porto_ete_v_7', 'porto_ete_v_8']
    # legend = ['']
    # legend_title = ''
    # ns=[0, 0, 0, 0]

    # agent_names = ['porto_ete_v_5', 'porto_ete_v5_r_dist', 'porto_ete_v5_cs_10']
    # legend = ['0.3', '0.4', '10Hz']
    # legend_title = ''
    # ns=[0, 0, 0]

    # agent_names = ['porto_ete_v_5']
    # legend = ['0.3']
    # legend_title = ''
    # ns=[2]

    # agent_names = ['porto_ete_v5_r_dist']
    # legend = ['0.4']
    # legend_title = ''
    # ns=[0]

    # agent_names = ['porto_ete_LiDAR_10', 'porto_ete_ddpg']
    # legend = ['TD3', 'DDPG']
    # legend_title = 'Learning \nmethod'
    # ns=[0, 0]
    # filename = 'learning_method_reward'

    # agent_names = ['porto_pete_sv_1', 'porto_ete_LiDAR_10']
    # legend = ['pete 6 m/s', 'ete 6 m/s']
    # legend_title = ''
    # ns=[0, 0]
    #filename = 'learning_method_reward'
    pass

#Tests with v=5
if True:
    # agent_names = ['porto_ete_v5_r_dist_1', 'porto_ete_v_5', 'porto_ete_v5_r_dist_2', 'porto_ete_v5_r_dist_4']
    # legend = ['0.1', '0.3', '0.5', '1']
    # legend_title = ''
    # ns=[0, 0, 0, 0]

    # agent_names = ['porto_ete_v5_cs_1', 'porto_ete_v5_cs_10', 'porto_ete_v5_cs_15', 'porto_ete_v_5', 'porto_ete_v5_cs_25']
    # legend = ['1', '10', '15', '20', '25']
    # legend_title = ''
    # ns=[0, 0, 0, 0, 0]

    # agent_names = ['porto_ete_v5_r_time_0', 'porto_ete_v_5']

    # agent_names = ['porto_ete_v5_r_collision_0', 'porto_ete_v5_r_collision_1', 'porto_ete_v_5', 'porto_ete_v5_r_collision_2']
    # legend = ['0', '-1', '-2', '-4']
    # legend_title = 'collision penalty'
    # ns=[0, 0, 0, 0]

    # agent_names = ['porto_ete_v5_r_collision_2']
    # legend = ['-4']
    # legend_title = 'collision penalty'
    # ns=[0]

    # agent_names = ['porto_ete_v5_r_collision_3', 'porto_ete_v5_r_collision_4', 'porto_ete_v5_r_collision_5']
    # legend = ['-6', '-8', '-10']
    # legend_title = 'collision penalty'
    # ns=[0, 0, 0]

    # agent_names = ['porto_ete_v5_r_collision_2', 'porto_ete_v5_r_collision_5']
    # legend = ['-4', '-10']
    # legend_title = 'collision penalty'
    # ns=[0, 0]

    # agent_names = ['porto_ete_v5_gamma_0', 'porto_ete_v5_gamma_1', 'porto_ete_v5_gamma_2', 'porto_ete_v5_r_collision_5', 'porto_ete_v5_gamma_3', 'porto_ete_v5_gamma_4']
    # legend = ['0.9', '0.95', '0.98', '0.99', '0.999', '1']
    # legend_title = 'gamma'
    # ns=[0, 0, 0, 0, 0, 0]

    # agent_names = ['porto_ete_v5_gamma_0', 'porto_ete_v5_gamma_1', 'porto_ete_v5_r_collision_5', 'porto_ete_v5_gamma_4']
    # legend = ['0.9', '0.95', '0.99', '1']
    # legend_title = 'gamma'
    # ns=[0, 0, 0, 0]

    # agent_names = ['porto_ete_v5_alpha_0', 'porto_ete_v5_r_collision_5', 'porto_ete_v5_alpha_1']
    # legend = ['$10^{-4}$','$10^{-3}$', '$2 \cdot 10^{-3}$']
    # legend_title = 'Learning rate'
    # ns=[0, 0, 0]

    # agent_names = ['porto_ete_v5_r_collision_5']
    # legend = ['']
    # legend_title = ''
    # ns=[0]
    

    # agent_names = ['porto_pete_v_k_0']
    # legend = ['']
    # legend_title = ''
    # ns=[0]

    pass



#Tests all tracks
if True:
    # agent_names = ['circle_ete', 'circle_pete_s', 'circle_pete_v', 'circle_pete_sv']
    # agent_names = ['porto_ete_v5_r_collision_5', 'porto_pete_s', 'porto_pete_v', 'porto_pete_sv']
    # agent_names = ['columbia_ete', 'columbia_pete_s', 'columbia_pete_v', 'columbia_pete_sv']
    # agent_names = ['berlin_ete', 'berlin_pete_s', 'berlin_pete_v', 'berlin_pete_sv']
    # agent_names = ['torino_ete', 'torino_pete_s', 'torino_pete_v', 'torino_pete_sv']
    # agent_names = ['redbull_ring_ete', 'redbull_ring_pete_s', 'redbull_ring_pete_v', 'redbull_ring_pete_sv']
    # agent_names = ['porto_ete_1', 'porto_pete_s_1', 'porto_pete_v_1', 'porto_pete_sv_1']

    # agent_names =  ['porto_pete_s_r_dist_0', 'porto_pete_s', 'porto_pete_s_r_dist_1', 'porto_pete_s_r_dist_2', 'porto_pete_s_r_dist_3']
    # agent_names = ['porto_pete_s_r_dist_0', 'porto_pete_s_r_dist_1']
    # legend = ['0.25', '0.3', '0.4', '0.5', '0.6']     
    # legend_title = ['r_dist']
    # ns = [0,0,0,0,0]
    
    
    # agent_names = ['redbull_ring_ete']
    # agent_names = ['porto_ete_v5_r_collision_5']

    # legend = ['None', 'Steering', 'Velocity', 'Steering and velocity']
    # legend_title = 'Controller'
    ns=[0, 0, 0 ,0]

    pass

# test porto pete steer control
if True:
    # agent_names = ['porto_pete_s_r_dist_1', 'porto_pete_s_r_collision_0', 'porto_pete_s_r_collision_1', 'porto_pete_s_r_collision_2']
    # legend = ['-1', '-2', '-4', '-8']
    # legend_title = 'Collision penalty'
    # ns=[0, 0, 0, 0]

    # agent_names = ['porto_pete_s_r_dist_0', 'porto_pete_s_r_dist_1', 'porto_pete_s_r_dist_2', 'porto_pete_s_r_dist_3', 'porto_pete_s_r_dist_4']
    # legend = ['0.15', '0.2', '0.25', '0.4', '0.5']
    # legend_title = 'Distance reward'
    # ns=[0, 0, 0, 0, 0]

    #agent_names = ['porto_pete_s_r_collision_3', 'porto_pete_s_r_collision_4', 'porto_pete_s_r_collision_5', 'porto_pete_s']

    # agent_names = ['porto_pete_s_r_dist_1']
    # legend = ['-2', '-4', '-8', '-10']
    # legend_title = 'Collision penalty'
    # ns=[2]


    # agent_names = ['porto_pete_s_polynomial']
    # ns = [1]
    # legend = ['polynomial']
    # legend_title = 'path'

    # agent_names = ['porto_ete_v5_r_collision_5', 'porto_pete_s_r_collision_0', 'porto_pete_s_polynomial']
    # ns = [0, 0, 0]
    # legend = ['No path', 'Circular path', 'Polynomial path']
    # legend_title = ''

    # agent_names = ['porto_ete_v5_r_collision_5', 'porto_pete_s_r_collision_0']
    # ns = [0, 0]
    # legend = ['No path', 'Circular path']
    # legend_title = ''


    # agent_names = ['porto_pete_s_lfc_4']
    # ns = [0]
    # legend = ['polynomial']
    # legend_title = 'circle'


    #lfc
    # agent_names = ['porto_pete_s_lfc_4', 'porto_pete_s_lfc_0', 'porto_pete_s_r_collision_0', 'porto_pete_s_lfc_1', 'porto_pete_s_lfc_2',  'porto_pete_s_lfc_3']
    # legend = ['0.3', '0.5', '1', '1.5', '2', '2.5']
    # legend_title = 'L_fc'
    # ns=[0,0,0,0,0,0]

    # agent_names = ['porto_pete_s_lfc_0', 'porto_pete_s_r_collision_0', 'porto_pete_s_lfc_2']
    # legend = ['0.5', '1', '2']
    # legend_title = 'L_fc'
    # ns=[0, 0, 0]

    # agent_names = ['porto_pete_s_lfc_2', 'porto_pete_s_stanley']
    # legend = ['Pure pursuit', 'Stanley']
    # legend_title = 'L_fc'
    # ns=[0,0]

    pass

#Tests porto pete velocity control
if True:
    # agent_names = ['porto_pete_v_k_1']
    # agent_names = ['porto_pete_v']
    # legend = ['']
    # legend_title = ''
    # ns=[0]
    
    
    # agent_names = ['porto_pete_v_k_0', 'porto_pete_v_r_dist_025', 'porto_pete_v_r_dist_03']
    # legend = ['0.2', '0.25', '0.3']
    # legend_title = 'rdist'
    # ns=[0,0,0]
    
    #rdist=0.2, vary r_collision
    # agent_names = ['porto_pete_v_k_0', 'porto_pete_v_r_collision_0',  'porto_pete_v_r_collision_1',  'porto_pete_v_r_collision_2']
    # legend = ['-2', '-4', '-6', '-8']
    # legend_title = 'r_collision'
    # ns=[0,0,0,0]

    # #r_dist=0.25, vary r_collision
    # agent_names = ['porto_pete_v_r_dist_025', 'porto_pete_v_r_collision_3',  'porto_pete_v_r_collision_4']
    # legend = ['-2', '-4', '-6', '-8']
    # legend_title = 'r_collision'
    # ns=[0,0,0]


    # #r_dist=0.3, vary r_collision
    # agent_names = ['porto_pete_v_r_dist_03', 'porto_pete_v_r_collision_5',  'porto_pete_v_r_collision_6']
    # legend = ['-2', '-4', '-8']
    # legend_title = 'r_collision'
    # ns=[0,0,0]

    #successful agent = 'porto_pete_v_r_collision_6'
    # agent_names = ['porto_pete_v_r_collision_6']
    # legend = ['-8']
    # legend_title = 'r_collision'
    # ns=[0]

    #k=2, r_dist=0.2, vary r_collision
    # agent_names = ['porto_pete_v_k_1', 'porto_pete_v_k2_r_collision_0',  'porto_pete_v_k2_r_collision_1', 'porto_pete_v_k2_r_collision_2']
    # legend = ['-2', '-4', '-6', '-8']
    # legend_title = 'r_collision'
    # ns=[0,0,0,0]

    #k=2, r_dist=0.25, vary r_collision
    # agent_names = ['porto_pete_v_r_dist_025', 'porto_pete_v_k2_r_collision_3',  'porto_pete_v_k2_r_collision_4']
    # legend = ['-2', '-4', '-8']
    # legend_title = 'r_collision'
    # ns=[0,0,0]

    #k=2, r_dist=0.3, vary r_collision
    # agent_names = ['porto_pete_v_r_dist_03', 'porto_pete_v_k2_r_collision_5',  'porto_pete_v_k2_r_collision_6']
    # legend = ['-2', '-4', '-8']
    # legend_title = 'r_collision'
    # ns=[0,0,0]

    
    
    
    
    # agent_names = ['porto_pete_v_r_collision_6', 'porto_ete_v5_r_collision_5']
    # legend = ['True', 'False']
    # legend_title = 'Velocity controller'
    # ns=[1,0]

    # agent_names = ['porto_pete_v_r_collision_6', 'porto_pete_v_k2_r_collision_6']
    # legend = ['1', '2']
    # legend_title = 'k'
    # ns=[1,0]

    # 2nd attempts with max_steps=3e6
    # agent_names = ['porto_pete_v_k_1', 'porto_pete_v_k_1_attempt_2']
    # legend = ['1', '2']
    # legend_title = 'attempt'
    # ns=[0, 0]

    # agent_names = [ 'porto_pete_v_r_collision_6', 'porto_pete_v_r_collision_6_attempt_2']
    # legend = ['1', '2']
    # legend_title = 'attempt'
    # ns=[0, 0]

    # agent_names = [ 'porto_pete_v_r_collision_6_attempt_2',  'porto_pete_v_k_1_attempt_2']
    # legend = ['1', '2']
    # legend_title = 'attempt'
    # ns=[0, 0]
    
    # agent_names = ['porto_pete_v_k_1_attempt_2']
    # legend = ['1']
    # legend_title = 'attempt'
    # ns=[0,0]
    
    # agent_names = ['porto_pete_v_k_1_attempt_2', 'porto_ete_v5_r_collision_5']
    # legend = ['True', 'False']
    # legend_title = 'Velocity controller'
    # ns=[0,0]
    
    # agent_names = ['porto_ete_v5_r_collision_5', 'porto_ete_v5_r_collision_5_attempt_2']
    # legend = ['1', '2']
    # legend_title = 'attempt'
    # ns=[0,0]
    
    # agent_names = ['porto_ete_v5_r_dist_02_attempt_2']
    # legend = ['1', '2']
    # legend_title = 'attempt'
    # ns=[0]


    #Reward signal retuning

    #rdist=0.2, vary r_collision
    # agent_names = ['porto_pete_v_k_0',  'porto_pete_v_r_collision_2']
    # legend = ['-2', '-8']
    # legend_title = 'r_collision'
    # ns=[0,0]

    # #r_dist=0.3, vary r_collision
    # agent_names = ['porto_pete_v_r_dist_03', 'porto_pete_v_r_collision_6']
    # legend = ['-2', '-8']
    # legend_title = 'r_collision'
    # ns=[0,0]


    # agent_names = ['porto_pete_v_r_dist_03', 'porto_pete_v_r_collision_6', 'porto_pete_v_k_0',  'porto_pete_v_r_collision_2']
    # legend = ['$r_{\mathrm{dist}}=0.3, r_{\mathrm{collision}}=-2$', '$r_{\mathrm{dist}}=0.3, r_{\mathrm{collision}}=-8$', '$r_{\mathrm{dist}}=0.2, r_{\mathrm{collision}}=-2$', '$r_{\mathrm{dist}}=0.2, r_{\mathrm{collision}}=-8$']
    # legend_title = 'Reward signal'
    # ns=[0,0,0,0]

    # agent_names = ['porto_pete_v_k_0']
    # legend = ['-2']
    # legend_title = 'r_collision'
    # ns=[0]

    # agent_names = ['porto_pete_v_k_0', 'porto_pete_v_r_collision_6']
    # legend = ['-2', '-8']
    # legend_title = '$r_{collision}$'
    # ns=[0,0,0,0]

    # agent_names = [ 'porto_ete_v5_r_dist_3']
    # legend = ['-2', '-8']
    # legend_title = '$r_{collision}$'
    # ns=[0]

    # agent_names = ['porto_ete_v5_r_collision_7']
    # legend = ['-2', '-8']
    # legend_title = '$r_{collision}$'
    # ns=[0]

    # agent_names = ['porto_pete_v_k_1_attempt_2', 'porto_pete_s_r_collision_0', 'porto_ete_v5_r_collision_5']
    # legend = ['Only velocity control', 'Only steering control', 'End-to-end']
    # legend_title = ''
    # ns=[0,2,0]
    # filename = 'velocity_control_learning_curves'
    # xlim = 5000
    # xspace = 1000



    # agent_names = ['porto_pete_sv_c_0']
    # legend = ['']
    # legend_title = ''
    # ns=[0]


    pass

# tests porto pete steer and velocity control tests 
if True:
    # agent_names = ['porto_pete_sv_c_r_0']
    # legend = ['pete sv']
    # legend_title = 'Architecture'
    # ns=[0]

    # agent_names = ['porto_pete_sv_c_r_8']
    # agent_names = ['porto_pete_sv_p_r_0']
    # legend = ['pete sv']
    # legend_title = 'Architecture'
    # ns=[0]

    pass

# Test run with noise
if True:

    # agent_name = 'porto_ete_v5_r_collision_5'
    # n_episodes = 100
    # detect_issues = False
    # initial_conditions = True
    # noise_param = 'xy'
    # noise_std = np.linspace(0,0.5,20)
    # main_multiple.lap_time_test_noise(agent_name, n_episodes, detect_issues, initial_conditions, noise_param, noise_std)
    # noise_param = 'theta'
    # noise_std = np.linspace(0, 10*np.pi/180, 20)
    # main_multiple.lap_time_test_noise(agent_name, n_episodes, detect_issues, initial_conditions, noise_param, noise_std)
    # noise_param = 'v'
    # noise_std = np.linspace(0,0.5,20)
    # main_multiple.lap_time_test_noise(agent_name, n_episodes, detect_issues, initial_conditions, noise_param, noise_std)
    # noise_param = 'lidar'
    # noise_std = np.linspace(0,0.3,20)
    # main_multiple.lap_time_test_noise(agent_name, n_episodes, detect_issues, initial_conditions, noise_param, noise_std)

    # agent_name = 'porto_pete_s_polynomial'
    # n_episodes = 100
    # detect_issues = False
    # initial_conditions = True
    # noise_param = 'xy'
    # noise_std = np.linspace(0,0.5,20)
    # main_multiple.lap_time_test_noise(agent_name, n_episodes, detect_issues, initial_conditions, noise_param, noise_std)
    # noise_param = 'theta'
    # noise_std = np.linspace(0, 10*np.pi/180, 20)
    # main_multiple.lap_time_test_noise(agent_name, n_episodes, detect_issues, initial_conditions, noise_param, noise_std)
    # noise_param = 'v'
    # noise_std = np.linspace(0,0.5,20)
    # main_multiple.lap_time_test_noise(agent_name, n_episodes, detect_issues, initial_conditions, noise_param, noise_std)
    # noise_param = 'lidar'
    # noise_std = np.linspace(0,0.3,20)
    # main_multiple.lap_time_test_noise(agent_name, n_episodes, detect_issues, initial_conditions, noise_param, noise_std)

    # agent_name = 'porto_pete_v_k_1_attempt_2'
    # n_episodes = 100
    # detect_issues = False
    # initial_conditions = True
    # noise_param = 'xy'
    # noise_std = np.linspace(0,0.5,20)
    # main_multiple.lap_time_test_noise(agent_name, n_episodes, detect_issues, initial_conditions, noise_param, noise_std)
    # noise_param = 'theta'
    # noise_std = np.linspace(0, 10*np.pi/180, 20)
    # main_multiple.lap_time_test_noise(agent_name, n_episodes, detect_issues, initial_conditions, noise_param, noise_std)
    # noise_param = 'v'
    # noise_std = np.linspace(0,0.5,20)
    # main_multiple.lap_time_test_noise(agent_name, n_episodes, detect_issues, initial_conditions, noise_param, noise_std)
    # noise_param = 'lidar'
    # noise_std = np.linspace(0,0.3,20)
    # main_multiple.lap_time_test_noise(agent_name, n_episodes, detect_issues, initial_conditions, noise_param, noise_std)

    # agent_name = 'porto_pete_sv_p_r_0'
    # n_episodes = 100
    # detect_issues = False
    # initial_conditions = True
    # noise_param = 'xy'
    # noise_std = np.linspace(0,0.5,20)
    # main_multiple.lap_time_test_noise(agent_name, n_episodes, detect_issues, initial_conditions, noise_param, noise_std)
    # noise_param = 'theta'
    # noise_std = np.linspace(0, 10*np.pi/180, 20)
    # main_multiple.lap_time_test_noise(agent_name, n_episodes, detect_issues, initial_conditions, noise_param, noise_std)
    # noise_param = 'v'
    # noise_std = np.linspace(0,0.5,20)
    # main_multiple.lap_time_test_noise(agent_name, n_episodes, detect_issues, initial_conditions, noise_param, noise_std)
    # noise_param = 'lidar'
    # noise_std = np.linspace(0,0.3,20)
    # main_multiple.lap_time_test_noise(agent_name, n_episodes, detect_issues, initial_conditions, noise_param, noise_std)



    # agent_name = 'porto_ete_v5_r_collision_5'
    # noise = {'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}
    # main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True, noise=noise)

    # agent_name = 'porto_pete_s_r_collision_0'
    # noise = {'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}
    # main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True, noise=noise)

    # agent_name = 'porto_pete_s_polynomial'
    # noise = {'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}
    # main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True, noise=noise)

    # agent_name = 'porto_pete_v_k_1_attempt_2'
    # noise = {'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}
    # main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True, noise=noise)

    # agent_name = 'porto_pete_sv_c_r_8'
    # noise = {'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}
    # main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True, noise=noise)

    # agent_name = 'porto_pete_sv_p_r_0'
    # noise = {'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}
    # main_multiple.lap_time_test_with_noise(agent_name=agent_name, n_episodes=n_test, detect_issues=False, initial_conditions=True, noise=noise)

    
    pass

# agent_names = ['porto_ete_v5_r_collision_5', 'porto_pete_s_r_collision_0', 'porto_pete_s_polynomial', 
#                 'porto_pete_v_k_1_attempt_2', 'porto_pete_sv_c_r_8', 'porto_pete_sv_p_r_0']
# noise_params = ['xy', 'theta', 'v', 'lidar']
# # noise_params = ['xy']
# legend_title = 'Agent architecture'
# legend = ['End-to-end',
#             'Steering control,\ncircular path',
#             'Steering control, \npolynomial path',
#             'Velocity control',
#             'Steering and velocity \ncontrol, circular path',
#             'Steering and velocity \ncontrol, polynomial path']
# display_results_multiple.display_lap_noise_results_multiple(agent_names, noise_params, legend_title, legend)
# display_results_multiple.display_lap_noise_results_single(agent_names, noise_params, legend_title, legend)

# Final porto agents!!!
# agent_names = ['porto_ete_v5_r_collision_5']  
# agent_names = ['porto_ete_v5_r_collision_5', 'porto_ete_v5_r_collision_5']    
# agent_names = ['porto_pete_s_r_collision_0']
# agent_names = ['porto_pete_s_polynomial', 'porto_pete_s_polynomial']   
# agent_names = ['porto_pete_v_k_1_attempt_2', 'porto_pete_v_k_1_attempt_2']
# agent_names = ['porto_pete_sv_c_r_8']
# agent_names = ['porto_pete_sv_p_r_0', 'porto_pete_sv_p_r_0']
# agent_names = ['porto_pete_sv_p_r_0']

# agent_names = ['porto_ete_v5_r_collision_5', 'porto_pete_s_polynomial', 'porto_pete_v_k_1_attempt_2', 'porto_pete_sv_p_r_0']
# legend = ['End-to-end', 'Steering control',  'Velocity control', 'Steering and velocity control']
# legend_title = ''
# ns=[0,1,0,1]

# agent_names = ['porto_ete_v5_r_collision_5_noise']    
# agent_names = ['porto_pete_s_r_collision_0_noise']
# agent_names = ['porto_pete_s_polynomial_noise']   
# agent_names = ['porto_pete_v_k_1_attempt_2_noise_1']
# agent_names = ['porto_pete_sv_c_r_8_noise']
# agent_names = ['porto_pete_sv_p_r_0_noise']


#Neural network and TD3 parameters
# agent_names = ['target_update_0', 'target_update_1']
# agent_names = ['update_actor_interval_0', 'update_actor_interval_1']
# agent_names = ['batch_size_0', 'batch_size_1']
# agent_names = ['explore_policy_0', 'explore_policy_1']
# agent_names = ['layer_0', 'layer_1']

# agent_names = ['batch_size_1']
# legend = agent_names
# ns=[0]

agent_names = ['time_steps']
# agent_names = ['porto_pete_sv_p_r_0']

# agent_names = ['porto_ete_only_LiDAR']
# agent_names = ['porto_ete_no_LiDAR']
# agent_names = ['batch_size_1']
# agent_names =  ['sample_5hz_batch_140_noise']

# agent_names =  ['batch_size_1', 'sample_5hz_batch_140_noise']
# agent_names =  ['batch_180',  'batch_220']
# agent_names =  ['sample_5hz_batch_140_noise']
# agent_names =  ['time_steps']
# agent_names = ['lidar_5', 'lidar_10', 'lidar_20', 'lidar_50', 'lidar_100', 'lidar_200']
# agent_names = ['train_noise']
# agent_names = ['batch_150','train_noise']
agent_names = ['batch_400']
# agent_names = ['lidar_5', 'lidar_10', 'lidar_20' ,'lidar_50']
# agent_names = ['lidar_200']
# agent_names = ['only_LiDAR', 'only_pose', 'batch_150']
# agent_names = ['only_LiDAR']
# agent_names = ['only_pose']
# agent_names = ['time_steps']
# agent_names = ['sample_3hz']
# agent_names = ['porto_ete_v5_gamma_0','porto_ete_v5_gamma_1', 'porto_ete_v5_gamma_2', 'porto_ete_v5_r_collision_5', 'porto_ete_v5_gamma_4']
# agent_names = ['porto_ete_v5_gamma_2']
# agent_names = ['porto_ete_v5_alpha_0', 'porto_ete_v5_r_collision_5', 'porto_ete_v5_alpha_1']
# agent_names = ['redbull']


# legend = ['no noise', 'noise']
# legend = ['180', '220']
# legend = ['5', '10', '20']
# legend = ['Trained without noise', 'Trained with noise']
legend = ['']
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


mismatch_parameters = [['C_Sr', 'mu'], ['C_Sr', 'mu']]
frac_vary = [[0, 0], [0, 0]]
noise_dicts = [{'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}, {'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}, {'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}, {'xy':0.025, 'theta':0.05, 'v':0.1, 'lidar':0.01}]
start_condition = {'x':10, 'y':4.5, 'v':3, 'theta':np.pi, 'delta':0, 'goal':0}



# Columbia
# start_condition = {'x':5.7, 'y':7.25, 'v':3, 'theta':0, 'delta':0, 'goal':0}
# start_condition = []


# NB!!!! Error: Path is junk when no mismatch is present, when displaying 2 agents

# display_results_multiple.display_moving_agent(agent_names=agent_names, ns=ns, legend_title=legend_title,          
#                                              legend=legend, mismatch_parameters=mismatch_parameters, frac_vary=frac_vary, noise_dicts=noise_dicts,
#                                              start_condition=start_condition)

display_results_multiple.display_path_multiple(agent_names=agent_names, ns=ns, legend_title=legend_title,          
                                             legend=legend, mismatch_parameters=mismatch_parameters, frac_vary=frac_vary, noise_dicts=noise_dicts,
                                             start_condition=start_condition)

# display_results_multiple.display_path_mismatch_multiple_by_state(agent_names=agent_names, ns=ns, legend_title=legend_title,          
#                                              legend=legend, mismatch_parameters=mismatch_parameters, frac_vary=frac_vary, noise_dicts=noise_dicts,
#                                              start_condition=start_condition)




if True:
    # agent_name = 'circle_pete_s_1'
    # main_dict['name'] = agent_name
    # env_dict['map_name'] = 'circle'
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['velocity_control'] = False
    # a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

    # agent_name = 'circle_pete_v_1'
    # main_dict['name'] = agent_name
    # env_dict['map_name'] = 'circle'
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

    # agent_name = 'circle_ete_1'
    # main_dict['name'] = agent_name
    # env_dict['map_name'] = 'circle'
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = False
    # a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

    # agent_name = 'columbia_pete_sv_1'
    # main_dict['name'] = agent_name
    # env_dict['map_name'] = 'columbia_1'
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['velocity_control'] = True
    # a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

    # agent_name = 'columbia_pete_s_1'
    # main_dict['name'] = agent_name
    # env_dict['map_name'] = 'columbia_1'
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['velocity_control'] = False
    # a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

    # agent_name = 'columbia_pete_v_1'
    # main_dict['name'] = agent_name
    # env_dict['map_name'] = 'columbia_1'
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

    # agent_name = 'columbia_ete_1'
    # main_dict['name'] = agent_name
    # env_dict['map_name'] = 'columbia_1'
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = False
    # a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_sv_1'
    # main_dict['name'] = agent_name
    # env_dict['map_name'] = 'porto_1'
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['velocity_control'] = True
    # a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_s_1'
    # main_dict['name'] = agent_name
    # env_dict['map_name'] = 'porto_1'
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['velocity_control'] = False
    # a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_pete_v_1'
    # main_dict['name'] = agent_name
    # env_dict['map_name'] = 'porto_1'
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

    # agent_name = 'porto_ete_1'
    # main_dict['name'] = agent_name
    # env_dict['map_name'] = 'porto_1'
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = False
    # a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

    # agent_name = 'berlin_pete_sv_1'
    # main_dict['name'] = agent_name
    # env_dict['map_name'] = 'berlin'
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['velocity_control'] = True
    # a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

    # agent_name = 'berlin_pete_s_1'
    # main_dict['name'] = agent_name
    # env_dict['map_name'] = 'berlin'
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['velocity_control'] = False
    # a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

    # agent_name = 'berlin_pete_v_1'
    # main_dict['name'] = agent_name
    # env_dict['map_name'] = 'berlin'
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

    # agent_name = 'berlin_ete_1'
    # main_dict['name'] = agent_name
    # env_dict['map_name'] = 'berlin'
    # env_dict['reward_signal']['time_step'] = -0.01
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = False
    # a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

    # agent_name = 'torino_pete_sv_1'
    # main_dict['name'] = agent_name
    # env_dict['map_name'] = 'torino'
    # env_dict['reward_signal']['time_step'] = -0.005
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['velocity_control'] = True
    # a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

    # agent_name = 'torino_pete_s_1'
    # main_dict['name'] = agent_name
    # env_dict['map_name'] = 'torino'
    # env_dict['reward_signal']['time_step'] = -0.005
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['velocity_control'] = False
    # a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

    # agent_name = 'torino_pete_v_1'
    # main_dict['name'] = agent_name
    # env_dict['map_name'] = 'torino'
    # env_dict['reward_signal']['time_step'] = -0.005
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

    # agent_name = 'torino_ete_1'
    # main_dict['name'] = agent_name
    # env_dict['map_name'] = 'torino'
    # env_dict['reward_signal']['time_step'] = -0.005
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = False
    # a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

    # agent_name = 'redbull_ring_pete_sv_1'
    # main_dict['name'] = agent_name
    # env_dict['map_name'] = 'redbull_ring'
    # env_dict['reward_signal']['time_step'] = -0.005
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['velocity_control'] = True
    # a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

    # agent_name = 'redbull_ring_pete_s_1'
    # main_dict['name'] = agent_name
    # env_dict['map_name'] = 'redbull_ring'
    # env_dict['reward_signal']['time_step'] = -0.005
    # env_dict['steer_control_dict']['steering_control'] = True
    # env_dict['velocity_control'] = False
    # a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

    # agent_name = 'redbull_ring_pete_v_1'
    # main_dict['name'] = agent_name
    # env_dict['map_name'] = 'redbull_ring'
    # env_dict['reward_signal']['time_step'] = -0.005
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = True
    # a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

    # agent_name = 'redbull_ring_ete_1'
    # main_dict['name'] = agent_name
    # env_dict['map_name'] = 'redbull_ring'
    # env_dict['reward_signal']['time_step'] = -0.005
    # env_dict['steer_control_dict']['steering_control'] = False
    # env_dict['velocity_control'] = False
    # a = trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
    # a.train()
    # lap_time_test(agent_name=agent_name, n_episodes=500, detect_issues=False, initial_conditions=True)

    # n_tests=400
    # n_fracs=21
    # agent_name = 'circle_pete_sv'
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # agent_name = 'circle_pete_s'
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # agent_name = 'circle_pete_v'
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # agent_name = 'circle_ete'
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,n_fracs) )

    # agent_name = 'columbia_pete_sv'
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # agent_name = 'columbia_pete_s'
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # agent_name = 'columbia_pete_v'
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # agent_name = 'columbia_ete'
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,n_fracs) )

    # agent_name = 'porto_pete_sv'
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # agent_name = 'porto_pete_s'
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # agent_name = 'porto_pete_v'
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # agent_name = 'porto_ete'
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,n_fracs) )

    # agent_name = 'berlin_pete_sv'
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # agent_name = 'berlin_pete_s'
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # agent_name = 'berlin_pete_v'
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # agent_name = 'berlin_ete'
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,n_fracs) )

    # agent_name = 'torino_pete_sv'
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # agent_name = 'torino_pete_s'
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # agent_name = 'torino_pete_v'
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # agent_name = 'torino_ete'
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,n_fracs) )

    # agent_name = 'redbull_ring_pete_sv'
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # agent_name = 'redbull_ring_pete_s'
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # agent_name = 'redbull_ring_pete_v'
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # agent_name = 'redbull_ring_ete'
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    # lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=np.linspace(-0.2,0.2,n_fracs) )
    
    pass

# model mismatch
if True:

    # agent_names = ['porto_ete_v5_r_collision_5']    
    # agent_names = ['porto_pete_s_r_collision_0']
    # agent_names = ['porto_pete_s_polynomial']   
    # agent_names = ['porto_pete_v_k_1_attempt_2']
    # agent_names = ['porto_pete_sv_c_r_8']
    # agent_names = ['porto_pete_sv_p_r_0']


    n_tests=100
    frac_variation = np.linspace(-0.2,0.2,21)
    mu_frac_variation = np.linspace(-1,1,21)
    # frac_variation = np.array([-2, -1, -0.5, 0.5, 1])

    # agent_name = 'porto_ete_v5_r_collision_5' 
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='mu', frac_variation=mu_frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_S', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='lf', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='h', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='m', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='I', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='sv', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='a_max', frac_variation=frac_variation)

    # agent_name = 'porto_pete_s_r_collision_0'
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='mu', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_S', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='lf', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='h', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='m', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='I', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='sv', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='a_max', frac_variation=frac_variation)


    # agent_name = 'porto_pete_s_polynomial' 
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='mu', frac_variation=mu_frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_S', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='lf', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='h', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='m', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='I', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='sv', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='a_max', frac_variation=frac_variation)

    # agent_name = 'porto_pete_v_k_1_attempt_2'
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='mu', frac_variation=mu_frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_S', frac_variation=frac_variation)
    # # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='lf', frac_variation=frac_variation)
    # # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='h', frac_variation=frac_variation)
    # # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='m', frac_variation=frac_variation)
    # # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='I', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='sv', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='a_max', frac_variation=frac_variation)

    # agent_name = 'porto_pete_sv_c_r_8'
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='mu', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_S', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='lf', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='h', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='m', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='I', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='sv', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='a_max', frac_variation=frac_variation)

    # agent_name = 'porto_pete_sv_p_r_0'
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='mu', frac_variation=mu_frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sf', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_Sr', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='C_S', frac_variation=frac_variation)
    # # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='lf', frac_variation=frac_variation)
    # # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='h', frac_variation=frac_variation)
    # # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='m', frac_variation=frac_variation)
    # # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='I', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='sv', frac_variation=frac_variation)
    # main_multiple.lap_time_test_mismatch(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, parameter='a_max', frac_variation=frac_variation)
    pass

#unknown mass
if True:

    # n_tests=100
    # mass = car_params['m']*0.1
    # length = car_params['lf']+car_params['lr']
    # distances = np.linspace(-0.05, length+0.05, 20)
    
    # agent_name = 'porto_ete_v5_r_collision_5' 
    # main_multiple.lap_time_test_unknown_mass(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, mass=mass, distances=distances)
  
    # agent_name = 'porto_pete_s_r_collision_0'
    # main_multiple.lap_time_test_unknown_mass(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, mass=mass, distances=distances)
  
    # agent_name = 'porto_pete_s_polynomial'
    # main_multiple.lap_time_test_unknown_mass(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, mass=mass, distances=distances)
     
    # agent_name = 'porto_pete_v_k_1_attempt_2'
    # main_multiple.lap_time_test_unknown_mass(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, mass=mass, distances=distances)
  
    # agent_name = 'porto_pete_sv_c_r_8'
    # main_multiple.lap_time_test_unknown_mass(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, mass=mass, distances=distances)
  
    # agent_name = 'porto_pete_sv_p_r_0'
    # main_multiple.lap_time_test_unknown_mass(agent_name=agent_name, n_episodes=n_tests, detect_issues=False, initial_conditions=True, mass=mass, distances=distances)
  
    pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    pass

# https://www.tutorialspoint.com/map-values-to-colors-in-matplotlib
# https://stackoverflow.com/questions/11550669/how-to-plot-in-different-colors-in-matplotlib