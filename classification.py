# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 22:16:34 2021

"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from imageio import imread

input_size = 784 # 28*28
hidden_size = 100
num_classes = 2
learning_rate = 0.001

class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNet, self).__init__()
    self.l1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.l2 = nn.Linear(hidden_size, num_classes)
  
  def forward(self, x):
    out = self.l1(x)
    out = self.relu(out)
    out = self.l2(out)
    return out

class pred():
    def classifier(image_path):
        # load image
        image = imread(image_path)
        # image = plt.imread(image_path)
        image_tensor = torch.from_numpy(image)
        image_tensor_reshaped = image_tensor.float().reshape(-1, 28*28)
        
        # model initialization
        model = NeuralNet(input_size, hidden_size, num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        
        # load model from checkpoint
        chkpoint_path = 'mnist_checkpoint.pth'
        checkpoint = torch.load(chkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # prediction
        out = model(image_tensor_reshaped)
        _, prediction = torch.max(out, dim = 1)
        return prediction.item()
        
