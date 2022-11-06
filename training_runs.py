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




agent_name = 'porto_ete_no_LiDAR'

main_dict = {'name':agent_name, 'max_episodes':50000, 'max_steps':2e6, 'learning_method':'td3', 'runs':3, 'comment':''}

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

reward_signal = {'goal_reached':0, 'out_of_bounds':-1, 'max_steps':0, 'collision':-1, 
                    'backwards':-0.01, 'park':-1, 'time_step':-0.01, 'progress':0, 'distance':0.3, 
                    'max_progress':0}    

action_space_dict = {'action_space': 'continuous', 'vel_select':[3,7], 'R_range':[2]}

#action_space_dict = {'action_space': 'discrete', 'n_waypoints': 10, 'vel_select':[7], 'R_range':[6]}

steer_control_dict = {'steering_control': False, 'wpt_arc':np.pi/2}

if  steer_control_dict['steering_control'] == True:
    steer_control_dict['path_strategy'] = 'circle'  #circle or linear or polynomial or gradient
    steer_control_dict['control_strategy'] = 'pure_pursuit'  #pure_pursuit or stanley

    if steer_control_dict['control_strategy'] == 'pure_pursuit':
        steer_control_dict['track_dict'] = {'k':0.1, 'Lfc':1}
    if steer_control_dict['control_strategy'] == 'stanley':
        steer_control_dict['track_dict'] = {'l_front': car_params['lf'], 'k':5, 'max_steer':car_params['s_max']}

lidar_dict = {'is_lidar':False, 'lidar_res':0.1, 'n_beams':20, 'max_range':20, 'fov':np.pi}

env_dict = {'sim_conf': functions.load_config(sys.path[0], "config")
        , 'save_history': False
        , 'map_name': 'porto_1'
        , 'max_steps': 3000
        , 'control_steps': 20
        , 'display': False
        , 'velocity_control': False
        , 'steer_control_dict': steer_control_dict
        , 'car_params':car_params
        , 'reward_signal':reward_signal
        , 'lidar_dict':lidar_dict
        , 'only_lidar':False
        , 'action_space_dict':action_space_dict
        } 

a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
a.train()
main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

agent_name = 'porto_ete_LiDAR_3'
main_dict['name'] = agent_name
env_dict['lidar_dict']['is_lidar'] = True
env_dict['lidar_dict']['n_beams'] = 3
a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
a.train()
main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

agent_name = 'porto_ete_LiDAR_10'
main_dict['name'] = agent_name
env_dict['lidar_dict']['n_beams'] = 10
a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
a.train()
main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

agent_name = 'porto_ete_LiDAR_20'
main_dict['name'] = agent_name
env_dict['lidar_dict']['n_beams'] = 20
a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
a.train()
main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

agent_name = 'porto_ete_LiDAR_50'
main_dict['name'] = agent_name
env_dict['lidar_dict']['n_beams'] = 50
a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
a.train()
main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

agent_name = 'porto_ete_cs_1'
main_dict['name'] = agent_name
env_dict['lidar_dict']['n_beams'] = 10
env_dict['control_steps'] = 1
a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
a.train()
main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

agent_name = 'porto_ete_cs_5'
main_dict['name'] = agent_name
env_dict['control_steps'] = 5
a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
a.train()
main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

agent_name = 'porto_ete_cs_10'
main_dict['name'] = agent_name
env_dict['control_steps'] = 10
a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
a.train()
main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

agent_name = 'porto_ete_cs_15'
main_dict['name'] = agent_name
env_dict['control_steps'] = 15
a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
a.train()
main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

agent_name = 'porto_ete_cs_25'
main_dict['name'] = agent_name
env_dict['control_steps'] = 25
a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
a.train()
main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

agent_name = 'porto_ete_v_5'
main_dict['name'] = agent_name
env_dict['control_steps'] = 20
env_dict['action_space_dict']['vel_select'] = [3,5]
a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
a.train()
main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

agent_name = 'porto_ete_v_7'
main_dict['name'] = agent_name
env_dict['action_space_dict']['vel_select'] = [3,7]
a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
a.train()
main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

agent_name = 'porto_ete_v_8'
main_dict['name'] = agent_name
env_dict['action_space_dict']['vel_select'] = [3,8]
a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
a.train()
main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

agent_name = 'porto_ete_only_LiDAR'
main_dict['name'] = agent_name
env_dict['only_lidar'] = True
env_dict['lidar_dict']['n_beams'] = 20
a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
a.train()
main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)


# agent_name = 'porto_ete_r_1'
# main_dict['name'] = agent_name
# env_dict['only_lidar'] = False
# env_dict['lidar_dict']['is_lidar'] = True
# env_dict['lidar_dict']['n_beams'] = 10
# env_dict['control_steps'] = 20
# env_dict['action_space_dict']['vel_select'] = [3,6]
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















# agent_name = 'porto_ete_optimal_0'
# main_dict['name'] = agent_name
# env_dict['only_lidar'] = False
# env_dict['lidar_dict']['n_beams'] = 20
# env_dict['action_space_dict']['vel_select'] = [3,7]
# env_dict['control_steps'] = 15
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)

# agent_name = 'porto_ete_optimal_1'
# main_dict['name'] = agent_name
# env_dict['only_lidar'] = False
# env_dict['lidar_dict']['n_beams'] = 20
# env_dict['action_space_dict']['vel_select'] = [3,6]
# env_dict['control_steps'] = 15
# a = main_multiple.trainingLoop(main_dict, agent_td3_dict, env_dict, load_agent='')
# a.train()
# main_multiple.lap_time_test(agent_name=agent_name, n_episodes=100, detect_issues=False, initial_conditions=True)


# agent_names = ['porto_ete_no_LiDAR', 'porto_ete_LiDAR_3', 'porto_ete_LiDAR_10', 'porto_ete_LiDAR_20', 'porto_ete_LiDAR_50']
# agent_names = ['porto_ete_LiDAR_3', 'porto_ete_LiDAR_10', 'porto_ete_LiDAR_20', 'porto_ete_LiDAR_50']
# legend = ['0', '3', '10', '20', '50']
# legend_title = 'Number of beams'
# display_results_multiple.learning_curve_lap_time(agent_names, legend, legend_title, show_average=True, show_median=True)
# display_results_multiple.learning_curve_reward(agent_names, legend, legend_title, show_average=True, show_median=True)

# agent_names = ['porto_ete_no_LiDAR', 'porto_ete_LiDAR_3', 'porto_ete_LiDAR_10', 'porto_ete_LiDAR_20', 'porto_ete_LiDAR_50']
# # agent_names = ['porto_ete_optimal_0']
# for agent_name in agent_names:
#     print('------------------------------' + '\n' + agent_name + '\n' + '------------------------------')
#     display_results_multiple.display_lap_results(agent_name=agent_name)



# agent_names = ['porto_ete_no_LiDAR', 'porto_ete_LiDAR_3', 'porto_ete_LiDAR_10', 'porto_ete_LiDAR_20', 'porto_ete_LiDAR_50']
#agent_names = ['porto_ete_optimal_0']
# legend_title = ''
# legend = ['0', '3', '10', '20', '50']
# ns = [0, 0, 0, 0, 0]
# mismatch_parameters = ['C_Sf']
# frac_vary = [0]
# start_condition = {'x':10, 'y':4.5, 'v':3, 'theta':np.pi, 'delta':0, 'goal':0}
# display_results_multiple.display_path_multiple(agent_names=agent_names, ns=ns, legend_title=legend_title,          
#                                              legend=legend, mismatch_parameters=mismatch_parameters, frac_vary=frac_vary, 
#                                              start_condition=start_condition)

# agent_names = ['porto_ete_no_LiDAR', 'porto_ete_LiDAR_50']
# legend_title = ''
# legend = ['0', '20']
# ns = [0, 0]
# mismatch_parameters = ['C_Sf']
# frac_vary = [0]
# start_condition = {'x':10, 'y':4.5, 'v':3, 'theta':np.pi, 'delta':0, 'goal':0}
# #start_condition = []
# display_results_multiple.display_path_multiple(agent_names=agent_names, ns=ns, legend_title=legend_title,          
#                                             legend=legend, mismatch_parameters=mismatch_parameters, frac_vary=frac_vary, 
#                                             start_condition=start_condition)


# agent_names = ['porto_ete_cs_1', 'porto_ete_cs_5', 'porto_ete_cs_10', 'porto_ete_cs_15', 'porto_ete_LiDAR_10', 'porto_ete_cs_25']
# legend = ['1', '5', '10', '15', '20', '25']
# legend_title = 'Simulation time steps per action'
# display_results_multiple.learning_curve_lap_time(agent_names, legend, legend_title, show_average=True, show_median=True)
# display_results_multiple.learning_curve_reward(agent_names, legend, legend_title, show_average=True, show_median=True)

# agent_names = ['porto_ete_cs_1', 'porto_ete_cs_5', 'porto_ete_cs_10', 'porto_ete_cs_15', 'porto_ete_LiDAR_10', 'porto_ete_cs_25']
# legend_title = ''
# legend = ['1', '5', '10', '15', '20', '25']
# ns = [0, 0, 0, 0, 0, 0]
# mismatch_parameters = ['C_Sf']
# frac_vary = [0]
#start_condition = {'x':10, 'y':4.5, 'v':3, 'theta':np.pi, 'delta':0, 'goal':0}
#display_results_multiple.display_path_multiple(agent_names=agent_names, ns=ns, legend_title=legend_title,          
#                                            legend=legend, mismatch_parameters=mismatch_parameters, frac_vary=frac_vary, 
#                                            start_condition=start_condition)


# agent_names = ['porto_ete_v_5', 'porto_ete_LiDAR_10', 'porto_ete_v_7', 'porto_ete_v_8']
# legend = ['5', '6', '7', '8']
# legend_title = 'Maximum velocity'
# display_results_multiple.learning_curve_lap_time(agent_names, legend, legend_title, show_average=True, show_median=True)
# display_results_multiple.learning_curve_reward(agent_names, legend, legend_title, show_average=True, show_median=True)

# agent_names = ['porto_ete_v_5', 'porto_ete_LiDAR_10', 'porto_ete_v_7', 'porto_ete_v_8']
# legend_title = ''
# legend = ['5', '6', '7', '8']
# ns = [0, 0, 0, 0]
# mismatch_parameters = ['C_Sf']
# frac_vary = [0]
#start_condition = {'x':10, 'y':4.5, 'v':3, 'theta':np.pi, 'delta':0, 'goal':0}
#display_results_multiple.display_path_multiple(agent_names=agent_names, ns=ns, legend_title=legend_title,          
#                                            legend=legend, mismatch_parameters=mismatch_parameters, frac_vary=frac_vary, 
#                                            start_condition=start_condition)



# agent_names = ['porto_ete_only_LiDAR', 'porto_ete_LiDAR_20']
# legend = ['Only LiDAR, 20 beams', '20 LiDAR beams and state']
# legend_title = ''
# display_results_multiple.learning_curve_lap_time(agent_names, legend, legend_title, show_average=True, show_median=True)
# display_results_multiple.learning_curve_reward(agent_names, legend, legend_title, show_average=True, show_median=True)
# ns = [0, 0, 0]
# mismatch_parameters = ['C_Sf']
# frac_vary = [0]
# start_condition = {'x':10, 'y':4.5, 'v':3, 'theta':np.pi, 'delta':0, 'goal':0}
# display_results_multiple.display_path_multiple(agent_names=agent_names, ns=ns, legend_title=legend_title,          
#                                             legend=legend, mismatch_parameters=mismatch_parameters, frac_vary=frac_vary, 
#                                             start_condition=start_condition)


# agent_names = ['porto_ete_average']
# legend = ['Only LiDAR, 20 beams']
# legend_title = ''
# ns = [2]
# display_results_multiple.learning_curve_lap_time(agent_names, legend, legend_title, ns, show_average=True, show_median=True)
# display_results_multiple.learning_curve_lap_time_average(agent_names, legend, legend_title, ns, show_average=True, show_median=True)
# display_results_multiple.learning_curve_reward(agent_names, legend, legend_title, ns=ns, show_average=True, show_median=True)
# display_results_multiple.learning_curve_reward_average(agent_names, legend, legend_title, show_average=True, show_median=True)



# agent_names = ['porto_ete_optimal_0', 'porto_ete_optimal_1']
# legend = ['0', '1']
# legend_title = ''
# display_results_multiple.learning_curve_lap_time(agent_names, legend, legend_title, show_average=True, show_median=True)
# display_results_multiple.learning_curve_reward(agent_names, legend, legend_title, show_average=True, show_median=True)
# ns = [0,0]
# mismatch_parameters = ['C_Sf']
# frac_vary = [0]
# start_condition = {'x':10, 'y':4.5, 'v':3, 'theta':np.pi, 'delta':0, 'goal':0}
# display_results_multiple.display_path_multiple(agent_names=agent_names, ns=ns, legend_title=legend_title,          
#                                             legend=legend, mismatch_parameters=mismatch_parameters, frac_vary=frac_vary, 
#                                             start_condition=start_condition)







































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
