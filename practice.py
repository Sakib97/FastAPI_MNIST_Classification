# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:07:48 2021


"""

import torch
t = torch.tensor([1, 2])
torch.save(t, 'demo.pth')

# loaded_num = torch.load('demo.pth')
# print(loaded_num)

loaded_num = torch.load('demo_from_colab_compatible.pth')
print(loaded_num)