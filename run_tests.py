
from cProfile import label
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

from cartpole_control import cartpole_control
from basic_game_NN import nn_train
from basic_game_NN_test import nn_test
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

#basic_game(BATCH_SIZE,GAMMA,EPS_START,EPS_END,EPS_DECAY,TARGET_UPDATE,num_episodes)
#test_model('./model_params',num_episodes=100)
#test_model_adversarial('./model_params',num_episodes=100)


print("#######################")
print("TRAINING MODEL")
#nn_train(display=False)
print("#######################")
print("TESTING MODEL")
#nn_test(adversarial=False,display=False)
print("#######################")
print("Testing Model Against Perturbations")
#nn_test(adversarial=True,display=False)
print("#######################")
print("Running Control Test")
#cartpole_control(adversarial=False,display=False)





'''
print("#######################")
print("Testing Control on Perturbations")

print("LINEAR PERTURBATIONS")
print("PERTURBATION STRENGTH: 0.01")
testA = cartpole_control(test_number=1,adversarial=True,display=False,perturbation_strength=0.01,linear_p=True)
print("PERTURBATION STRENGTH: 0.1")
testB = cartpole_control(test_number=2,adversarial=True,display=False,perturbation_strength=0.1,linear_p=True)
print("PERTURBATION STRENGTH: 0.5")
testC = cartpole_control(test_number=3,adversarial=True,display=False,perturbation_strength=0.5,linear_p=True)
print("PERTURBATION STRENGTH: 1 (Double)")
testD = cartpole_control(test_number=4,adversarial=True,display=False,perturbation_strength=1,linear_p=True)
print("PERTURBATION STRENGTH: 2 (Triple)")
testE = cartpole_control(test_number=5,adversarial=True,display=False,perturbation_strength=2,linear_p=True)

plt.plot(list(range(len(testA))),testA, label='p=0.01')
plt.plot(list(range(len(testB))),testB, label='p=0.1')
plt.plot(list(range(len(testC))),testC, label='p=0.5')
plt.plot(list(range(len(testD))),testD, label='p=1')
plt.plot(list(range(len(testE))),testE, label='p=2')
plt.legend(title='Severity')
plt.title('Control Policy on Linear Perturbations')
plt.savefig('./results/control_linear_pururbations.jpg')
plt.clf()


print("NOISY PERTURBATIONS")
print("PERTURBATION STRENGTH: 0.01")
testA = cartpole_control(test_number=1,adversarial=True,display=False,perturbation_strength=0.01,linear_p=False)
print("PERTURBATION STRENGTH: 0.1")
testB = cartpole_control(test_number=2,adversarial=True,display=False,perturbation_strength=0.1,linear_p=False)
print("PERTURBATION STRENGTH: 0.5")
testC = cartpole_control(test_number=3,adversarial=True,display=False,perturbation_strength=0.5,linear_p=False)
print("PERTURBATION STRENGTH: 1 (Double)")
testD = cartpole_control(test_number=4,adversarial=True,display=False,perturbation_strength=1,linear_p=False)
print("PERTURBATION STRENGTH: 2 (Triple)")
testE = cartpole_control(test_number=5,adversarial=True,display=False,perturbation_strength=2,linear_p=False)

plt.plot(list(range(len(testA))),testA, label='p=0.01')
plt.plot(list(range(len(testB))),testB, label='p=0.1')
plt.plot(list(range(len(testC))),testC, label='p=0.5')
plt.plot(list(range(len(testD))),testD, label='p=1')
plt.plot(list(range(len(testE))),testE, label='p=2')
plt.legend(title='Severity')
plt.title('Control Policy on Noisy Perturbations, Noise STD: .05')
plt.savefig('./results/control_noisy_pururbations.jpg')
plt.clf()
'''
print("########### Tests Finished ############")

