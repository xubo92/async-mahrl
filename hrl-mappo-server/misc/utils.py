import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary   
import torchvision 
from torchvision import models
import numpy as np

import matplotlib.pyplot as plt
from typing import Tuple
from collections import deque
import warnings
import threading


import ai2thor
import matplotlib.pyplot as plt
from ai2thor.controller import Controller

# Pytorch resnet-18 [totally untrainable]
class ResNetEmbedder(nn.Module):
    def __init__(self, device=None):
        super(ResNetEmbedder, self).__init__()
        self.device = torch.device("cpu") if device is None else device
        self.model = torchvision.models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(self.device)

    def forward(self, x):
        with torch.no_grad():
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            return x
    
    def vis_any_feature(self, x, layer_name):
        for layer_na, layer in self.model._modules.items():
            x = layer(x)
            if layer_na == layer_name:
                return x

def dist2d(p0, p1):
    if type(p0) is dict and type(p1) is dict:
        x0 = p0["x"]
        z0 = p0["z"]

        x1 = p1["x"]
        z1 = p1["z"]
    elif type(p0) is tuple and type(p1) is tuple:
        x0 = p0[0]
        z0 = p0[2]

        x1 = p1[0]
        z1 = p1[2]
    else:
        raise ValueError("invalid data structure for points")
    return np.sqrt((x0-x1) ** 2 + (z0-z1) ** 2) 

def get_shortest_path(start, end, reachable_poses, grid_size=0.25):
    """
    input: start, end, and reachable_poses are all dicts
    return: shortest path, each point is tuple
    """
    reachable_poses_tuple = [(p["x"], p["y"], p["z"]) for p in reachable_poses]
    start_tuple = (start["x"], start["y"], start["z"])
    end_tuple = (end["x"], end["y"], end["z"])
    neighbors = dict()
    for position in reachable_poses_tuple:
        position_neighbors = set()
        for p in reachable_poses_tuple:
            if position != p and (
                (
                    abs(position[0] - p[0]) < 1.5 * grid_size
                    and abs(position[2] - p[2]) < 0.5 * grid_size
                )
                or (
                    abs(position[0] - p[0]) < 0.5 * grid_size
                    and abs(position[2] - p[2]) < 1.5 * grid_size
                )
            ):
                position_neighbors.add(p)
        neighbors[position] = position_neighbors
    
    def closest_grid_point(world_point):
        min_dist = float("inf")
        closest_point = None
        assert len(reachable_poses_tuple) > 0
        for p in reachable_poses_tuple:
            dist = dist2d(p, world_point)
            if dist < min_dist:
                min_dist = dist
                closest_point = p
        return closest_point
    
    start = closest_grid_point(start_tuple)
    end = closest_grid_point(end_tuple)
    # print(start, end)

    if start == end:
        return [start]

    q = deque()
    q.append([start])

    visited = set()

    while q:
        path = q.popleft()
        pos = path[-1]

        if pos in visited:
            continue

        visited.add(pos)
        for neighbor in neighbors[pos]:
            if neighbor == end:
                return path + [neighbor]
            q.append(path + [neighbor])

    raise Exception("Invalid state. Must be a bug!")

def get_rot(old_rot, start, end):
    if start["x"] == end["x"]:
        if end["z"] > start["z"]:
            rot = {"x":0, "y":0, "z":0}
        elif end["z"] < start["z"]:
            rot = {"x":0, "y":180, "z":0}
        else:
            rot = old_rot

    elif start["z"] == end["z"]:
        if end["x"] > start["x"]:
            rot = {"x":0, "y":90, "z":0}
        elif end["x"] < start["x"]:
            rot = {"x":0, "y":270, "z":0}
        else:
            rot = old_rot
    else:
        warnings.warn("start and end have different x and z at the same time")   
    return rot
