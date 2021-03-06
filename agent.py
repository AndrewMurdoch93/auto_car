import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
from collections import OrderedDict

class deepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions):
        super(deepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        #self.fc4 = nn.Linear(self.fc3_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #self.device = 'cpu'
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state.float()))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        action_values = self.fc3(x)
        
        return action_values


class agent():
    def __init__(self, agent_dict):
        
        self.agent_dict = agent_dict
        self.name = self.agent_dict['name']
        self.gamma = self.agent_dict['gamma']
        self.epsilon = self.agent_dict['epsilon']
        self.eps_min = self.agent_dict['eps_end']
        self.eps_dec = self.agent_dict['eps_dec']
        self.lr = self.agent_dict['lr']
        self.n_actions = self.agent_dict['n_actions']
        self.action_space = [i for i in range(self.n_actions)]
        self.mem_size = self.agent_dict['max_mem_size']
        self.batch_size = self.agent_dict['batch_size']
        self.input_dims = self.agent_dict['input_dims']
        self.mem_cntr = 0
        self.iter_cntr = 0 
        self.replace_target = 100

        self.fc1_dims=self.agent_dict['fc1_dims']
        self.fc2_dims=self.agent_dict['fc2_dims']
        self.fc3_dims=self.agent_dict['fc2_dims']

        self.Q_eval = deepQNetwork(self.lr, n_actions=self.n_actions, input_dims=self.input_dims, fc1_dims=self.fc1_dims, fc2_dims=self.fc2_dims, fc3_dims=self.fc3_dims)


        self.state_memory = np.zeros((self.mem_size, self.input_dims), dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, self.input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)


    def store_transition(self, state, action, reward, next_state, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal
        self.mem_cntr += 1
    

    def get_batch(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        next_state_batch = T.tensor(self.next_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        return batch, batch_index, state_batch, next_state_batch, action_batch, reward_batch, terminal_batch


    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action


    def decrease_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.epsilon = self.eps_min

    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()
        batch, batch_index, state_batch, next_state_batch, action_batch, reward_batch, terminal_batch = self.get_batch()

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(next_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma*T.max(q_next,dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
    

    def save_agent(self):
        T.save(self.Q_eval.state_dict(), 'agents/' + self.name + '_weights')
        self.agent_dict['epsilon'] = self.epsilon
        
        outfile = open('agents/' + self.name + '_hyper_parameters', 'wb')
        pickle.dump(self.agent_dict, outfile)
        outfile.close()


    def load_weights(self, name):
        self.Q_eval.load_state_dict(T.load('agents/' + name + '_weights'))


            
