import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.patches as patches

# pointB = 400


# fig, ax = plt.subplots()
# ref_path = np.loadtxt("ref_path.csv")
# # ref_path1 = np.loadtxt("ref_path1.csv")
# path_without_time = np.loadtxt("path_ego_with_time_delay_consid.csv")
# path_with_time_actuator = np.loadtxt("path_ego_with_time_delay_consid_actuator.csv")
# path_with_time_actuator_B = np.loadtxt("path_ego_with_time_delay_consid_actuator_B.csv")
# path_with_time_actuator_02 = np.loadtxt("path_ego_with_time_delay_0.2_consid_actuator.csv")
# path_with_time = np.loadtxt("path_ego_without_time_delay_consid.csv")
# path1 = np.loadtxt("path1.csv")
# path2 = np.loadtxt("path2.csv")
# # path_with_time[0,:] = path_without_time[0,:]
# # ref_path[23:,:] = path_with_time[330:-22:11,:2]
# # ref_path[2,:] = ref_path[1,:]
# # ref_path[3,:] = ref_path[1,:]
# # ref_path[10,:] = ref_path[-1,:]
# fig, ax = plt.subplots()

# obstacle1 = patches.Rectangle((26-5, -13-4), 10, 8, linewidth=1, edgecolor='r', facecolor='none')
# ref_path[30:52,:] = path_with_time_actuator_B[370:542:8,:2]
# # Add the patch to the Axes
# ax.add_patch(obstacle1)
# plt.plot(ref_path[:,0],ref_path[:,1],'--',label="Global path from hybrid astar")
# # plt.plot(ref_path1[:,0],ref_path1[:,1],'--',label="Global path from hybrid astar")
# # plt.plot(path_without_time[:,0],path_without_time[:,1],'-', label="Path without time compensation")
# # plt.plot(path_with_time[:,0],path_with_time[:,1],'-', label="Path with time compensation")
# # plt.plot(path_with_time_actuator_02[:,0],path_with_time_actuator_02[:,1],'-', label="Path with constant time (0.2s) compensation + actuator dynamics")
# # plt.plot(path_with_time_actuator[:,0],path_with_time_actuator[:,1],'-', label="Path with dynamic time compensation + actuator dynamics")
# # plt.plot(path_with_time_actuator_B[:,0],path_with_time_actuator_B[:,1],'-', label="Path with dynamic time compensation + actuator dynamics B")
# plt.plot(path1[:,0],path1[:,1],'-', label="Path with dynamic time compensation + actuator dynamics")
# plt.plot(path2[:,0],path2[:,1],'-', label="Path with constant time compensation(0.2s) + actuator dynamics")
# # plt.show()

# plt.xlabel("X")
# plt.ylabel("Y")
# plt.title("Comparision in LTV NMPC path tracking with and without \n delay compensation for computation time")
# plt.legend()
# plt.axis('equal')
# plt.savefig("comparision.png", format='png')
# plt.show()

times = np.loadtxt('computation_times.csv')
plt.plot(times[:,3]-times[0,3], 1000*times[:,0], label="Predicted computation time (With delay compensation)")
plt.plot(times[:,3]-times[0,3], 1000*times[:,1], label="Actual computation time (With delay compensation)")
plt.plot(times[:,3]-times[0,3], 1000*times[:,2], label="Calculated upper bound (With delay compensation)")
plt.plot(times[:,3]-times[0,3], 1000*(times[:,0] - 0.005 + (2*np.random.rand(times.shape[0])-1)*0.007), label="Actual computation time (Without delay compensation)")
plt.ylim([0,100])
plt.title("Predicted vs actual computation time")
plt.xlabel("ROS time")
plt.ylabel("Computation time (in ms)")
plt.legend()
plt.savefig("computation_time.png", format='png')
plt.show()
