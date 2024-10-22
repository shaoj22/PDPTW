'''
File: SAC.py
Project: DRLH
File Created: Thursday, 1st June 2023 7:52:18 am
Author: Charles Lee (lmz22@mails.tsinghua.edu.cn)
'''


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import random
import gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import rl_utils
import datetime

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.preprocess = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_layer = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.preprocess(x))
        return F.softmax(self.output_layer(x), dim=1)


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        self.preprocess = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_layer = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.preprocess(x))
        return self.output_layer(x)

class SAC:
    ''' 处理离散动作的SAC算法 '''
    def __init__(self, args):
        # read params
        state_dim = args.state_dim
        hidden_dim = args.hidden_dim
        action_dim = args.action_dim
        actor_lr = args.actor_lr
        critic_lr = args.critic_lr
        alpha_lr = args.alpha_lr
        target_entropy = args.target_entropy
        tau = args.tau
        gamma = args.gamma
        device = args.device
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 第一个Q网络
        self.critic_1 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        # 第二个Q网络
        self.critic_2 = QValueNet(state_dim, hidden_dim, action_dim).to(device)
        self.target_critic_1 = QValueNet(state_dim, hidden_dim,
                                         action_dim).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNet(state_dim, hidden_dim,
                                         action_dim).to(device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device
        # set writer
        self.global_step = 0

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)  # 动作不再是float类型
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        # 直接根据概率计算熵
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)  # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        # 记录训练信息
        loss_info = {
            'actor_loss': actor_loss.item(),
            'critic_1_loss': critic_1_loss.item(),
            'critic_2_loss': critic_2_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'entropy' : entropy.mean().item(),
        }
        self.global_step += 1
        return loss_info
    
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.actor.state_dict(), path + '\\actor.pth')
        torch.save(self.critic_1.state_dict(), path + '\\critic_1.pth')
        torch.save(self.critic_2.state_dict(), path + '\\critic_2.pth')
    
    def load(self, path):
        self.actor.load_state_dict(torch.load(path + '\\actor.pth'))
        self.critic_1.load_state_dict(torch.load(path + '\\critic_1.pth'))
        self.critic_2.load_state_dict(torch.load(path + '\\critic_2.pth'))

def train_off_policy_agent(args, agent, env, writer):
    replay_buffer = rl_utils.ReplayBuffer(args.buffer_size)
    with tqdm(total=args.num_episodes) as pbar:
        train_step = 0
        for i_episode in range(args.num_episodes):
            episode_reward = 0
            state, _ = env.reset()
            done = False
            step = 0
            while not done:
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                if replay_buffer.size() > args.minimal_size and step % args.update_step == 0:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(args.batch_size)
                    transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                    loss_info = agent.update(transition_dict)
                    for key, value in loss_info.items():
                        writer.add_scalar("loss/"+key, value, train_step)
                    train_step += 1
                step += 1
            writer.add_scalar("reward/episode_reward", episode_reward, i_episode)
            pbar.set_postfix({'episode': '%d' % (i_episode), 'reward': '%.3f' % episode_reward})
            pbar.update(1)
            if i_episode % args.save_interval == 0:
                agent.save(args.model_dir)




