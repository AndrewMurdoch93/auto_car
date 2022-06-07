import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle

#blank_trans = (0, np.zeros((12), dtype=np.float32), 0.0, 0.0,  np.zeros(12), False)

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

class SegmentTree():
    def __init__(self, size):
        self.index = 0
        self.size = size
        self.full = False  # Used to track actual capacity
        self.tree_start = 2**(size-1).bit_length()-1  # Put all used node leaves on last tree level
        self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
        self.data = np.array([blank_trans] * size, dtype=Transition_dtype)
        self.max = 1  # Initial max value to return (1 = 1^ω)

    def _update_nodes(self, indices):
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

    def _propagate(self, indices):
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self._update_nodes(unique_parents)
        if parents[0] != 0:
            self._propagate(parents)

    def _propagate_index(self, index):
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate_index(parent)
    
    def update(self, indices, values):
        self.sum_tree[indices] = values  # Set new values
        self._propagate(indices)  # Propagate values
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)
    
    def _update_index(self, index, value):
        self.sum_tree[index] = value  # Set new value
        self._propagate_index(index)  # Propagate value
        self.max = max(value, self.max)
    
    def append(self, data, value):
        self.data[self.index] = data  # Store data in underlying data structure
        self._update_index(self.index + self.tree_start, value)  # Update tree
        self.index = (self.index + 1) % self.size  # Update index
        self.full = self.full or self.index == 0  # Save when capacity reached
        self.max = max(value, self.max)

    def _retrieve(self, indices, values):
        children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1)) # Make matrix of children indices
        # If indices correspond to leaf nodes, return them
        if children_indices[0, 0] >= self.sum_tree.shape[0]:
            return indices
        # If children indices correspond to leaf nodes, bound rare outliers in case total slightly overshoots
        elif children_indices[0, 0] >= self.tree_start:
            children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)
        left_children_values = self.sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values).astype(np.int32)  # Classify which values are in left or right branches
        successor_indices = children_indices[successor_choices, np.arange(indices.size)] # Use classification to index into the indices matrix
        successor_values = values - successor_choices * left_children_values  # Subtract the left branch values when searching in the right branch
        return self._retrieve(successor_indices, successor_values)
    
    # Searches for values in sum tree and returns values, data indices and tree indices
    def find(self, values):
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start
        return (self.sum_tree[indices], data_index, indices)  # Return values, data indices, tree indices

    # Returns data given a data index
    def get(self, data_index):
        return self.data[data_index % self.size]

    def total(self):
        return self.sum_tree[0]
    
class ReplayMemory:
    def __init__(self, max_size, batch_size, replay_alpha):
        self.batch_size=batch_size
        self.capacity = max_size
        self.replay_alpha = replay_alpha
        self.transitions = SegmentTree(self.capacity)
        self.t = 0

    def store_transition(self, state, action, reward, next_state, done):
        self.transitions.append((self.t, state, action, reward, next_state, done), self.transitions.max)  # Store new transition with maximum priority
        self.t = 0 if done else self.t + 1  # Start new episodes with t = 0
    
    def sample(self, replay_beta):
        capacity = self.capacity if self.transitions.full else self.transitions.index
        while True:
            p_total=self.transitions.total()
            samples = np.random.uniform(0, p_total, self.batch_size)
            probs, data_idxs, tree_idxs = self.transitions.find(samples)
            if np.all(data_idxs<=capacity):
                break
        
        data = self.transitions.get(data_idxs)
        probs = probs / p_total
        #weights = (capacity * probs) ** -replay_beta  # Compute importance-sampling weights w
        #weights = weights / weights.max()  # Normalise by max importance-sampling weight from batch
        
        if np.any(probs==0):
            print('Probs are 0')
        if capacity==0:
            print('Probs are 0')

        weights = np.power(np.multiply(np.divide(1, capacity), np.divide(1, probs)), replay_beta)
        if np.any(weights==np.inf):
            print('weights are inf')
        if np.any(weights==0):
            print('weights are 0')
        
        norm_weights = np.divide(weights, np.max(weights))
        if np.max(weights)==np.inf:
            print('weights are inf')
        if np.max(weights)==0:
            print('weights are 0')

        #return tree_idxs, states, actions, returns, next_states, nonterminals, weights
        return tree_idxs, data, norm_weights

    def update_priorities(self, idxs, priorities):
        priorities = np.power(np.abs(priorities.cpu().detach().numpy()), self.replay_alpha)
        self.transitions.update(idxs, priorities)


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name):
        super(CriticNetwork, self).__init__()
        
        self.name = name
        self.file_name = 'agents/' + self.name + '_weights'
        
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        f3 = 0.003
        self.q = nn.Linear(self.fc2_dims, 1)
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.file_name)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.file_name))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name):
        super(ActorNetwork, self).__init__()
        self.name = name
        self.file_name = 'agents/' + self.name + '_weights'
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x))

        return x

    def save_checkpoint(self):
        T.save(self.state_dict(), self.file_name)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.file_name))


class agent(object):

    def __init__(self, agent_dict):
        
        self.agent_dict = agent_dict

        global Transition_dtype
        global blank_trans 
        Transition_dtype = np.dtype([('timestep', np.int32), ('state', np.float32, (self.agent_dict['input_dims'])), ('action', np.float32), ('reward', np.float32), ('next_state', np.float32, (self.agent_dict['input_dims'])), ('done', np.bool_)])
        blank_trans = (0, np.zeros((agent_dict['input_dims']), dtype=np.float32), 0.0, 0.0,  np.zeros((agent_dict['input_dims']), dtype=np.float32), False)
        
        self.name = agent_dict['name']
        self.gamma = agent_dict['gamma']
        self.tau = agent_dict['tau']
        self.noise_constant = 1
        self.learn_step_counter = 0
        self.batch_size = agent_dict['batch_size']
        #self.memory = ReplayBuffer(agent_dict['max_size'], agent_dict['input_dims'], agent_dict['n_actions'])
        self.memory = ReplayMemory(max_size=self.agent_dict['max_mem_size'], batch_size=self.agent_dict['batch_size'], replay_alpha=self.agent_dict['replay_alpha'])

        self.actor = ActorNetwork(agent_dict['alpha'], agent_dict['input_dims'], agent_dict['layer1_size'], agent_dict['layer2_size'], 
                                    agent_dict['n_actions'], name=self.name+'_actor')

        self.critic = CriticNetwork(agent_dict['beta'], agent_dict['input_dims'], agent_dict['layer1_size'], agent_dict['layer2_size'], 
                                    agent_dict['n_actions'], name=self.name+'_critic')

        self.target_actor = ActorNetwork(agent_dict['alpha'], agent_dict['input_dims'], agent_dict['layer1_size'], agent_dict['layer2_size'], 
                                    agent_dict['n_actions'], name=self.name+'_target_actor')

        self.target_critic = CriticNetwork(agent_dict['beta'], agent_dict['input_dims'], agent_dict['layer1_size'], agent_dict['layer2_size'], 
                                    agent_dict['n_actions'], name=self.name+'_target_critic')

        self.noise = OUActionNoise(mu=np.zeros( agent_dict['n_actions']))

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)*self.noise_constant
        self.actor.train()
        return mu_prime.cpu().detach().numpy()


    def choose_greedy_action(self, observation):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        action = self.actor.forward(observation).to(self.actor.device)
        self.actor.train()
        return action.cpu().detach().numpy()


    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)


    def learn(self, replay_beta):
        if self.memory.transitions.index<self.batch_size and self.memory.transitions.full==False:
            return
        
        
        #state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        #reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        #done = T.tensor(done).to(self.critic.device)
        #new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        #action = T.tensor(action, dtype=T.float).to(self.critic.device)
        #state = T.tensor(state, dtype=T.float).to(self.critic.device)

        tree_idxs, data, weights = self.memory.sample(replay_beta)

        state = T.tensor(np.copy(data[:]['state'])).to(self.critic.device)
        reward = T.tensor(np.copy(data[:]['reward'])).to(self.critic.device)
        done = T.tensor(np.copy(data[:]['done'])).to(self.critic.device)
        action = T.tensor(np.copy(data[:]['action'])).to(self.critic.device).view(self.batch_size, 1)
        new_state = T.tensor(np.copy(data[:]['next_state'])).to(self.critic.device)

        indices = np.arange(self.batch_size)
        
        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action.view(self.batch_size, 1).type(T.float))

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*~done[j])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        #target = (reward+self.gamma*critic_value.squeeze()*done).view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        errors = T.sub(target, critic_value).to(self.critic.device)

        self.learn_step_counter += 1
        self.memory.update_priorities(tree_idxs, abs(errors).view(self.batch_size)+1e-6)

        critic_loss = T.mean(T.multiply(T.square(errors.view(self.batch_size)).to(self.critic.device), T.tensor(weights).to(self.critic.device)))
        #critic_loss = T.mean(T.multiply(errors.view(self.batch_size).to(self.critic.device), T.tensor(weights).to(self.critic.device)))
        
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

        #self.learn_step_counter += 1
        #self.memory.update_priorities(tree_idxs, abs(errors))

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                      (1-tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                      (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

        """
        #Verify that the copy assignment worked correctly
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(target_critic_params)
        actor_state_dict = dict(target_actor_params)
        print('\nActor Networks', tau)
        for name, param in self.actor.named_parameters():
            print(name, T.equal(param, actor_state_dict[name]))
        print('\nCritic Networks', tau)
        for name, param in self.critic.named_parameters():
            print(name, T.equal(param, critic_state_dict[name]))
        input()
        """
    '''
    def save_agent(self):
        T.save(self.actor.state_dict(), 'agents/' + self.name + '_actor_weights')
        T.save(self.target_actor.state_dict(), 'agents/' + self.name + '_target_actor_weights')
        T.save(self.critic.state_dict(), 'agents/' + self.name + '_target_actor_weights')
        T.save(self.target_critic.state_dict(), 'agents/' + self.name + '_target_critic_weights')
        
        outfile = open('agents/' + self.name + '_hyper_parameters', 'wb')
        pickle.dump(self.agent_dict, outfile)
        outfile.close()

    def load_weights(self, name):
        self.actor.load_state_dict(T.load('agents/' + name + '_actor_weights'))
        self.target_actor.load_state_dict(T.load('agents/' + name + '_target_actor_weights'))
        self.critic.load_state_dict(T.load('agents/' + name + '_target_actor_weights'))
        self.target_critic.load_state_dict(T.load('agents/' + name + '_target_critic_weights'))
    '''

    def save_agent(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

        outfile = open('agents/' + self.name + '_hyper_parameters', 'wb')
        pickle.dump(self.agent_dict, outfile)
        outfile.close()

    def load_weights(self, name):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
    
    def check_actor_params(self):
        current_actor_params = self.actor.named_parameters()
        current_actor_dict = dict(current_actor_params)
        original_actor_dict = dict(self.original_actor.named_parameters())
        original_critic_dict = dict(self.original_critic.named_parameters())
        current_critic_params = self.critic.named_parameters()
        current_critic_dict = dict(current_critic_params)
        print('Checking Actor parameters')

        for param in current_actor_dict:
            print(param, T.equal(original_actor_dict[param], current_actor_dict[param]))
        print('Checking critic parameters')
        for param in current_critic_dict:
            print(param, T.equal(original_critic_dict[param], current_critic_dict[param]))
        input()

    def decrease_noise_factor(self):
        self.noise_constant *= 0.995
