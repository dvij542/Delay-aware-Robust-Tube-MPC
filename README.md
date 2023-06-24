# Delay-aware-Robust-Tube-MPC

The codes for :- 
1. [Delay aware robust control for safe autonomous driving](https://ieeexplore.ieee.org/document/9827111) 
2. [Delay aware robust control for safe autonomous driving and racing](https://arxiv.org/abs/2208.13856)

## Folder structure

- Controller A/ and Controller B/ : All the code files for Controller A and B implementation of [Delay aware robust control for safe autonomous driving](https://ieeexplore.ieee.org/document/9827111) which includes delay aware robust tube MPC formulation for global frame of reference with simulations in ROS Gazebo

- Controller racing A/ and Controller racing B/ : All the code files for Controller A and B implementation of [Delay aware robust control for safe autonomous driving and racing](https://ieeexplore.ieee.org/document/9827111) which includes delay aware robust tube MPC formulation for frenet frame of reference with simulations on race-like environments on Carla

- global_racetrajectory_optimization/ : Adapted from [Link](https://github.com/TUMFTM/global_racetrajectory_optimization). Contains code files for generating global racing line reference for 2nd work.

- launch/ : Contains launch files to be used for ROS gazebo simulations

- models/ : Contains additional models used in Gazebo simulation

- worlds/ : Contains world files for gazebo simulation

## Requirements 

- Carla (>=0.9.7)
- ROS (relevant version depending on OS)
- Gazebo (Required only for 1st work)
- Python (>3.8)
- pytope
- pygame
- scipy
- casadi

## Steps to run 2nd work simulations

