
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from model import ReplayMemory, DQN
from basic_game_cnn import basic_game
from basic_game_cnn import test_model
from basic_game_cnn import test_model_adversarial

'''


'''



BATCH_SIZE = 128
GAMMA = 0.99
#epsilon greedy values
EPS_START = 0.5
EPS_END = 0.001
EPS_DECAY = 8888
TARGET_UPDATE = 1
num_episodes = 1000

basic_game(BATCH_SIZE,GAMMA,EPS_START,EPS_END,EPS_DECAY,TARGET_UPDATE,num_episodes)
test_model('./model_params',num_episodes=100)
test_model_adversarial('./model_params',num_episodes=100)
