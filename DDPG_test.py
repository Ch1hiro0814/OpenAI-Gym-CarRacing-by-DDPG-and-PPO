import argparse
import numpy as np
import gym
import torch
import torch.nn as nn
import cv2
from collections import deque
import random

parser = argparse.ArgumentParser(description='Test a DDPG agent for the CarRacing-v2')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--action-repeat', type=int, default=4, metavar='N', help='repeat action in N frames (default: 8)')
parser.add_argument('--img-stack', type=int, default=4, metavar='N', help='stack N image in a state (default: 4)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', default=True, help='render the environment')
parser.add_argument('--model-path', type=str, default='G:\Important\Course\RL\Project\car_caring\model\ddpg_best_params.pkl', help='path to the saved model parameters')
parser.add_argument('--episodes', type=int, default=10, help='number of test episodes')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

class Env():
    def __init__(self):
        self.env = gym.make('CarRacing-v2', render_mode='human' if args.render else None)
        self.env.action_space.seed(args.seed)
        self.reward_threshold = self.env.spec.reward_threshold

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

def load_model(actor, model_path):
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    actor.load_state_dict(checkpoint['actor'])

if __name__ == "__main__":
    env = Env()
    actor = Actor().double().to(device)
    load_model(actor, args.model_path)

    for i_ep in range(args.episodes):
        state = env.reset()
        score = 0
        for t in range(1000):
            state_tensor = torch.from_numpy(state).double().to(device).unsqueeze(0)
            with torch.no_grad():
                action = actor(state_tensor).cpu().data.numpy().flatten()
            steer = action[0]
            gas = (action[1] + 1) / 2
            brake = (action[2] + 1) / 2
            action_env = np.array([steer, gas, brake])

            state_, reward, done, die = env.step(action_env)
            score += reward
            state = state_
            if args.render:
                env.render()
            if done or die:
                break
        print('Test Episode: {}\tScore: {:.2f}'.format(i_ep, score))
