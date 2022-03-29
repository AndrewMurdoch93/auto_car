import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class DuelingLinearDeepQNetwork(nn.Module):
    #def __init__(self, ALPHA, n_actions, name, input_dims, chkpt_dir='tmp/dqn'):
    def __init__(self, ALPHA, n_actions, input_dims, fc1_dims, fc2_dims):
        super(DuelingLinearDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.V = nn.Linear(fc2_dims, 1)
        self.A = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        #self.checkpoint_dir = chkpt_dir
        #self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_dqn')

    def forward(self, state):
        l1 = F.relu(self.fc1(state))
        l2 = F.relu(self.fc2(l1))
        V = self.V(l2)
        A = self.A(l2)

        return V, A


class agent(object):
    #def __init__(self, gamma, epsilon, alpha, n_actions, input_dims,
    #             mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
    #             replace=1000, chkpt_dir='tmp/dqn'):
    def __init__(self, agent_dict):
        
        self.agent_dict = agent_dict
        self.name = self.agent_dict['name']
        self.gamma = self.agent_dict['gamma']
        self.epsilon = self.agent_dict['epsilon']
        self.eps_end = self.agent_dict['eps_end']
        self.eps_dec = self.agent_dict['eps_dec']
        self.action_space = [i for i in range(self.agent_dict['n_actions'])]
        self.learn_step_counter = 0
        self.batch_size = self.agent_dict['batch_size']
        self.replace_target_cnt = self.agent_dict['replace']
        
        self.memory = ReplayBuffer(self.agent_dict['max_mem_size'], self.agent_dict['input_dims'], self.agent_dict['n_actions'])

        self.q_eval = DuelingLinearDeepQNetwork(self.agent_dict['alpha'], self.agent_dict['n_actions'], input_dims=self.agent_dict['input_dims'],
                                                fc1_dims=self.agent_dict['fc1_dims'], fc2_dims=self.agent_dict['fc2_dims'])

        self.q_next = DuelingLinearDeepQNetwork(self.agent_dict['alpha'], self.agent_dict['n_actions'], input_dims=self.agent_dict['input_dims'],
                                                fc1_dims=self.agent_dict['fc1_dims'], fc2_dims=self.agent_dict['fc2_dims'])
     

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(observation, dtype=T.float32).to(self.q_eval.device)
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def replace_target_network(self):
        if self.replace_target_cnt is not None and self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrease_epsilon(self):
        if self.epsilon > self.eps_end:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.epsilon = self.eps_end

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        # using T.Tensor seems to reset datatype to float
        # using T.tensor preserves source data type
        
        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        indices = np.arange(self.batch_size)

        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1,keepdim=True)))
        
        max_actions = T.argmax(q_eval, dim=1)

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next[indices, max_actions]
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

    def save_agent(self):
        T.save(self.q_eval.state_dict(), 'agents/' + self.name + 'q_eval_weights')
        T.save(self.q_next.state_dict(), 'agents/' + self.name + 'q_next_weights')
        self.agent_dict['epsilon'] = self.epsilon
        
        outfile = open('agents/' + self.name + '_hyper_parameters', 'wb')
        pickle.dump(self.agent_dict, outfile)
        outfile.close()

    def load_weights(self, name):
        self.q_eval.load_state_dict(T.load('agents/' + name + 'q_eval_weights'))
        self.q_eval.load_state_dict(T.load('agents/' + name + 'q_next_weights'))
