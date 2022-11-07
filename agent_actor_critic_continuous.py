import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle

class GenericNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions, filename):
        super(GenericNetwork, self).__init__()
        self.filename = filename
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save_checkpoint(self, name):
        T.save(self.state_dict(), self.filename+name)

    def load_checkpoint(self, name):
        self.load_state_dict(T.load(self.filename+name))

        
class ActorCriticNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.pi = nn.Linear(self.fc2_dims, self.n_actions)
        self.v = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)

        return pi, v

class agent_separate(object):

    def __init__(self, agent_dict):
        self.name = agent_dict["name"]
        self.agent_dict=agent_dict
        self.name = agent_dict['name']
        self.gamma = agent_dict['gamma']
        self.log_probs = None
        self.n_outputs = 1
        
        self.actor = GenericNetwork(agent_dict['alpha'], agent_dict['input_dims'], agent_dict['fc1_dims'],
                                           agent_dict['fc2_dims'], n_actions=2, filename=self.name)
        
        self.critic = GenericNetwork(agent_dict['beta'], agent_dict['input_dims'], agent_dict['fc1_dims'],
                                           agent_dict['fc2_dims'], n_actions=1, filename=self.name)
        

    def choose_action(self, observation):
        mu, sigma  = self.actor.forward(observation)#.to(self.actor.device)
        sigma = T.exp(sigma)
        action_probs = T.distributions.Normal(mu, sigma)
        probs = action_probs.sample(sample_shape=T.Size([self.n_outputs]))
        self.log_probs = action_probs.log_prob(probs).to(self.actor.device)
        action = T.tanh(probs)

        return action.item()

    def learn(self, state, reward, new_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        critic_value_ = self.critic.forward(new_state)
        critic_value = self.critic.forward(state)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        delta = ((reward + self.gamma*critic_value_*(1-int(done))) - critic_value)

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()

        self.actor.optimizer.step()
        self.critic.optimizer.step()

    def save_agent(self, name, run):
        self.actor.save_checkpoint(name + '_actor_n_' + str(run))
        
        #T.save(self.actor.state_dict(), 'agents/' + self.name + '_actor_weights')
        #T.save(self.critic.state_dict(), 'agents/' + self.name + '_critic_weights')
        
        #outfile = open('agents/' + self.name + '_hyper_parameters', 'wb')
        #pickle.dump(self.agent_dict, outfile)
        #outfile.close()

    def load_weights(self, name, run):
        self.actor.load_checkpoint(name + '_actor_n_' + str(run))

        #self.actor.load_state_dict(T.load('agents/' + name + '_actor_weights'))
        #self.critic.load_state_dict(T.load('agents/' + name + '_critic_weights'))



class agent_combined(object):
    def __init__(self, alpha, beta, input_dims, gamma=0.99, n_actions=2,
                 layer1_size=256, layer2_size=256, n_outputs=1):
        self.gamma = gamma
        self.log_probs = None
        self.n_outputs = n_outputs
        self.actor_critic = ActorCriticNetwork(alpha, input_dims, layer1_size,
                                        layer2_size, n_actions=n_actions)

    def choose_action(self, observation):
        pi, v = self.actor_critic.forward(observation)

        mu, sigma = pi
        sigma = T.exp(sigma)
        action_probs = T.distributions.Normal(mu, sigma)
        probs = action_probs.sample(sample_shape=T.Size([self.n_outputs]))
        self.log_probs = action_probs.log_prob(probs).to(self.actor_critic.device)
        action = T.tanh(probs)

        return action.item()

    def learn(self, state, reward, new_state, done):
        self.actor_critic.optimizer.zero_grad()

        _, critic_value_ = self.actor_critic.forward(new_state)
        _, critic_value = self.actor_critic.forward(state)

        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)
        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()

        self.actor_critic.optimizer.step()