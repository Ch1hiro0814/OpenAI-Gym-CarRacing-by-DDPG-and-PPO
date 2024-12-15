import argparse
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import cv2
from utils import DrawLine

parser = argparse.ArgumentParser(description='Train a DDPG agent for the CarRacing-v2')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--action-repeat', type=int, default=4, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='interval between training logs (default: 10)')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)


class Env():
    def __init__(self):
        self.env = gym.make('CarRacing-v2')
        self.env.action_space.seed(args.seed)

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()
        self.die = False
        img_rgb = self.env.reset()[0]
        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * args.img_stack
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for i in range(args.action_repeat):
            img_rgb, reward, terminated, truncated, info = self.env.step(action)
            die = terminated or truncated
            if die:
                reward += 100
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        return np.array(self.stack), total_reward, done, die

    def render(self):
        self.env.render()

    @staticmethod
    def rgb2gray(rgb, norm=True):
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        if norm:
            gray = gray / 128. - 1.
        return gray

    @staticmethod
    def reward_memory():
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory

class Actor(nn.Module):
    def __init__(self, state_dim=(args.img_stack, 96, 96), action_dim=3):
        super(Actor, self).__init__()
        # CNN特征提取
        self.cnn_base = nn.Sequential(
            nn.Conv2d(args.img_stack, 8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Linear(100, action_dim),
            nn.Tanh()
        )
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        action = self.fc(x)
        return action

class Critic(nn.Module):
    def __init__(self, state_dim=(args.img_stack,96,96), action_dim=3):
        super(Critic, self).__init__()
        self.cnn_base = nn.Sequential(
            nn.Conv2d(args.img_stack, 8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(256+action_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x, a):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        x = torch.cat([x, a], dim=1)
        q_value = self.fc(x)
        return q_value


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def store(self, s, a, r, s_, done):
        self.buffer.append((s, a, r, s_, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        s = np.array([t[0] for t in transitions], dtype=np.float64)
        a = np.array([t[1] for t in transitions], dtype=np.float64)
        r = np.array([t[2] for t in transitions], dtype=np.float64).reshape(-1,1)
        s_ = np.array([t[3] for t in transitions], dtype=np.float64)
        d = np.array([t[4] for t in transitions], dtype=np.float64).reshape(-1,1)
        return s, a, r, s_, d

    def __len__(self):
        return len(self.buffer)


def soft_update(net, target_net, tau=0.005):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class DDPGAgent:
    def __init__(self, state_dim=(args.img_stack,96,96), action_dim=3, lr_actor=1e-4, lr_critic=1e-4):
        self.action_dim = action_dim
        self.actor = Actor(state_dim, action_dim).double().to(device)
        self.critic = Critic(state_dim, action_dim).double().to(device)

        self.actor_target = Actor(state_dim, action_dim).double().to(device)
        self.critic_target = Critic(state_dim, action_dim).double().to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = args.gamma
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 256
        self.min_buffer_size = 10000
        self.tau = 0.005

    def select_action(self, state, noise_scale=0.1):
        state_tensor = torch.from_numpy(state).double().to(device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        action = action + noise_scale * np.random.randn(self.action_dim)
        return np.clip(action, -1, 1)

    def store_transition(self, s, a, r, s_, done):
        self.replay_buffer.store(s, a, r, s_, done)

    def update(self):
        if len(self.replay_buffer) < self.min_buffer_size:
            return None, None

        s, a, r, s_, d = self.replay_buffer.sample(self.batch_size)
        s = torch.tensor(s, dtype=torch.double).to(device)
        a = torch.tensor(a, dtype=torch.double).to(device)
        r = torch.tensor(r, dtype=torch.double).to(device)
        s_ = torch.tensor(s_, dtype=torch.double).to(device)
        d = torch.tensor(d, dtype=torch.double).to(device)

        with torch.no_grad():
            a_next = self.actor_target(s_)
            q_next = self.critic_target(s_, a_next)
            target_q = r + (1 - d) * self.gamma * q_next

        q = self.critic(s, a)
        critic_loss = F.mse_loss(q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        soft_update(self.actor, self.actor_target, self.tau)
        soft_update(self.critic, self.critic_target, self.tau)

        return actor_loss.item(), critic_loss.item()

    def save_param(self, filename='param/ddpg_actor_critic_params.pkl'):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, filename)


if __name__ == "__main__":
    print(device)
    env = Env()
    agent = DDPGAgent()
    max_score = -9999
    running_score = 0
    last_model_score = -9999
    actor_losses = []
    critic_losses = []

    for i_ep in range(100000):
        state = env.reset()
        score = 0
        for t in range(1000):
            if args.render:
                env.render()
            action = agent.select_action(state)
            steer = action[0]
            gas = (action[1] + 1) / 2
            brake = (action[2] + 1) / 2
            action_env = np.array([steer, gas, brake])
            state_, reward, done, die = env.step(action_env)
            agent.store_transition(state, action, reward, state_, done or die)
            actor_loss, critic_loss = agent.update()
            if actor_loss is not None and critic_loss is not None:
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            score += reward
            state = state_
            if done or die:
                break

        avg_actor_loss = sum(actor_losses) / len(actor_losses) if actor_losses else 0
        avg_critic_loss = sum(critic_losses) / len(critic_losses) if critic_losses else 0

        with open('critic_loss.txt', 'a') as log_file:
            log_file.write(f'Episode {i_ep}\tCritic Loss: {avg_critic_loss:.6f}\n')
        with open('actor_loss.txt', 'a') as log_file:
            log_file.write(f'Episode {i_ep}\tActor Loss: {avg_actor_loss:.6f}\n')

        running_score = running_score * 0.99 + score * 0.01
        if i_ep % args.log_interval == 0:
            if running_score > last_model_score:
                agent.save_param('param/ddpg_last_params_2.pkl')
                print("The last model has been saved")
                last_model_score = running_score
        if score >= max_score:
            max_score = score
            agent.save_param('param/ddpg_best_params_2.pkl')
            print("The best model has been saved")
        log_message = 'Ep {}\tLast Score: {:.2f}\tRunning Score: {:.2f}\tMax Score: {:.2f}'.format(
            i_ep, score, running_score, max_score)
        print(log_message)
        with open('training_log_test.txt', 'a') as log_file:
            log_file.write(log_message + '\n')
