from turtle import width
import numpy as np
import matplotlib.pyplot as plt
import sys

from FrenetOptimalTrajectory.frenet_optimal_trajectory import *

traj_without_comp = np.loadtxt('trajectory_without_comp.csv')
traj_with_comp = np.loadtxt('trajectory_with_comp.csv')

with_i = 0
while traj_with_comp[-with_i,1]>150  :
    with_i+=1
without_i = 0
while traj_without_comp[-without_i,1]>150 :
    without_i+=1

veh_dims = np.array([[-1.4,-0.7],[-1.4,0.7],[1.4,0.7],[1.4,-0.7],[-1.4,-0.7]])

def plot_vehicle(x,y,theta,color='blue') :
    rot_mat = np.array([[math.cos(theta),math.sin(theta)],[-math.sin(theta),math.cos(theta)]])
    points = np.array([[x,y]]) + np.matmul(veh_dims,rot_mat)
    plt.plot(points[:,0],points[:,1],color=color)

start_i = 200
end_i = 300
traj_with_comp = traj_with_comp[:-with_i+2]
traj_without_comp = traj_without_comp[:-without_i+2]
# print(len(traj_with_comp))
for i in range(start_i,end_i) :
    x,y,theta = traj_with_comp[i,0],traj_with_comp[i,1],math.atan2(traj_with_comp[i+1,1]-traj_with_comp[i,1],traj_with_comp[i+1,0]-traj_with_comp[i,0])
    plot_vehicle(x,y,theta,color='purple')

start_i = 200
end_i = 300

for i in range(start_i,end_i) :
    x,y,theta = traj_without_comp[i,0],traj_without_comp[i,1],math.atan2(traj_without_comp[i+1,1]-traj_without_comp[i,1],traj_without_comp[i+1,0]-traj_without_comp[i,0])
    plot_vehicle(x,y,theta,color='red')

# left_boundary = np.loadtxt('left_boundary_with_comp.csv')
# right_boundary = np.loadtxt('right_boundary_with_comp.csv')
opt_racing_line = np.loadtxt('../waypoints_new.csv',delimiter=',')

file_centre_line='../racetrack_waypoints.txt'
if file_centre_line != None:
    centre_line = np.loadtxt(file_centre_line,delimiter = ",")
else :
    centre_line=None
centre_line[:,1] = -centre_line[:,1]
tx_center, ty_center, tyaw_center, tc_center, ts_center, csp_center = generate_target_course(centre_line[:,0], centre_line[:,1])

# plt.plot(left_boundary)
# plt.plot(right_boundary)

# Start line
plt.plot([-67.4,-67.4],[238,254],linewidth=5.0,color='green')#,marker='o')
plt.text(-67.4,255,'Start line')

# Finish line
plt.plot([-16,0],[152.6,152.6],linewidth=5.0,color='red')#,marker='o')
plt.text(-25,154,'End line')

# plt.plot(-372,65,-358,65,marker='o',size=5)

left_boundary = np.array([tx_center-5.2*np.sin(tyaw_center),ty_center+5.2*np.cos(tyaw_center)]).T
right_boundary = np.array([tx_center+5.2*np.sin(tyaw_center),ty_center-5.2*np.cos(tyaw_center)]).T
plt.plot(opt_racing_line[:,0],-opt_racing_line[:,1],'--',label="Optimal racing line")
plt.plot(left_boundary[:,0],left_boundary[:,1],'--',label="Track left boundary")
plt.plot(right_boundary[:,0],right_boundary[:,1],'--',label="Track right boundary")
plt.plot(traj_without_comp[:,0],traj_without_comp[:,1],'-',label="Without delay compensation")
plt.plot(traj_with_comp[:,0],traj_with_comp[:,1],'-',label="With delay compensation")

plt.xlabel("X")
plt.ylabel("Y")
# plt.title("Comparision in LTV NMPC path tracking with and without \n delay compensation for computation time")
plt.legend()
plt.axis('equal')
# plt.savefig("with_comp_const.png", format='png')
plt.show()

speeds_without_comp = np.loadtxt('forward_speed_without_comp.csv')[:-without_i+2]*1.5
ref_speeds_without_comp = np.loadtxt('reference_signal_without_comp.csv')[:-without_i+2]*1.5

plt.plot(ref_speeds_without_comp[:,0]/1.5,ref_speeds_without_comp[:,1],'--',label="Reference speed")
plt.plot(speeds_without_comp[:,0]/1.5,speeds_without_comp[:,1],'-',label="Speeds without delay compensation")

plt.xlabel("time (in s)")
plt.ylabel("speed (in m/s)")
plt.ylim(0,35*1.5)
plt.legend()
plt.show()

speeds_with_comp = np.loadtxt('forward_speed_with_comp.csv')[:-with_i+2]*1.5
ref_speeds_with_comp = np.loadtxt('reference_signal_with_comp.csv')[:-with_i+2]*1.5

plt.plot(ref_speeds_with_comp[:,0]/1.5,ref_speeds_with_comp[:,1],'--',label="Reference speed")
plt.plot(speeds_with_comp[:,0]/1.5,speeds_with_comp[:,1],'-',label="Speeds with delay compensation")

plt.xlabel("time (in s)")
plt.ylabel("speed (in m/s)")
plt.ylim(0,35*1.5)
plt.legend()
plt.show()
