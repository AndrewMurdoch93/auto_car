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


#Final porto agents!!!
# agent_names = ['porto_ete_v5_r_collision_5']    
# agent_names = ['porto_pete_s_r_collision_0']
# agent_names = ['porto_pete_s_polynomial']   
# agent_names = ['porto_pete_v_k_1_attempt_2']
# agent_names = ['porto_pete_sv_c_r_8']
# agent_names = ['porto_pete_sv_p_r_0']

agent_names = ['porto_ete_v5_r_collision_5', 'porto_pete_s_r_collision_0', 'porto_pete_s_polynomial']
agent_names = ['porto_pete_v_k_1_attempt_2', 'porto_pete_sv_c_r_8', 'porto_pete_sv_p_r_0']

n_episodes = 100
detect_issues = False
initial_conditions = True

for agent_name in agent_names:
    noise_param = 'xy'
    noise_std = np.arange(0,2.05,0.1)
    main_multiple.lap_time_test_noise(agent_name, n_episodes, detect_issues, initial_conditions, noise_param, noise_std)
    noise_param = 'theta'
    noise_std = np.arange(0, 40*np.pi/180, np.pi/180)
    main_multiple.lap_time_test_noise(agent_name, n_episodes, detect_issues, initial_conditions, noise_param, noise_std)
    noise_param = 'v'
    noise_std = np.arange(0,2,0.05)
    main_multiple.lap_time_test_noise(agent_name, n_episodes, detect_issues, initial_conditions, noise_param, noise_std)
    noise_param = 'lidar'
    noise_std = np.arange(0,1,0.025)
    main_multiple.lap_time_test_noise(agent_name, n_episodes, detect_issues, initial_conditions, noise_param, noise_std)


