# Delay-aware-Robust-Tube-MPC

The codes for :- 
1. [Delay aware robust control for safe autonomous driving](https://ieeexplore.ieee.org/document/9827111) 
2. [Delay aware robust control for safe autonomous driving and racing](https://arxiv.org/abs/2208.13856)

NOTE : There are some notation errors in our IEEE IV conference paper (1). The original non-linear model is linearized around the current state at every time step into a linear system. However, in the MPC's prediction horizon, we assume time-invariant dynamics. An updated version can be found here : [Updated arxiv version](https://arxiv.org/abs/2109.07101) 

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

1. Set up carla(>=0.9.7, tested on 0.9.8) under 'Controller racing A' directory
2. Install additional map Town07 following the instructions from :-
  
### Plan A experiments
1. Change directory to 'Controller racing A'
```
cd Controller\ racing\ A
```
2. (optional) Invariant set has already been pre-calculated, but if needed to recalculate for different parameters, change parameters in header file of inv_set_calc.py and run:-
```
python inv_set_calc.py
```
3. In each new terminal from now on, source ros at the beginning. In a new terminal run :-
```
roscore
``` 
4. In a new terminal, launch carla, make sure Town07 map is installed. 
5. Set SCENE='standalone' for experiment 2, 'one_vehicle' for experiment 3, 'one_vehicle_turn' for experiment 4 in carla_utils.py
6. In a new terminal, for experiment 2, set hyperparameters in header of mpc_robust_frenet_without_obstacles.py as required and run :-
```
python mpc_robust_frenet_without_obstacles.py 
```  
For experiment 3 and 4, change hyperparameters in header of mpc_robust_frenet.py and run :-
```
python mpc_robust_frenet.py 
```
7. In a new terminal, launch :-
```
python pre_compensator.py
``` 
The vehicle should be moving at this point. The experimental result files and plots should be saved at the end of the experiment under 'outputs_<scenario name>' folder with suffix 'with_comp' and 'without_comp' respectively based on if delay compensation is enabled or not in header of mpc_robust_frenet_without_obstacles.py or mpc_robust_frenet.py accordingly

### Plan B experiments
1. Change directory to 'Controller racing B'
```
cd Controller\ racing\ B
```
2. In each new terminal from now on, source ros at the beginning. In a new terminal run :-
```
roscore
``` 
3. In a new terminal, launch carla, make sure Town07 map is installed. 
4. Set SCENE='standalone' for experiment 1, 'one_vehicle' for experiment 3, 'one_vehicle_turn' for experiment 4 in carla_utils.py
5. In a new terminal, for experiment 2, set hyperparameters in header of mpc_robust_frenet_without_obstacles.py as required and run :-
```
python mpc_robust_frenet_without_obstacles.py 
```  
For experiment 3 and 4, change hyperparameters in header of mpc_robust_frenet.py and run :-
```
python mpc_robust_frenet.py 
```
6. In a new terminal, launch :-
```
python pre_compensator.py
``` 
