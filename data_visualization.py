#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import jax
import matplotlib.pyplot as plt
#from helping_function import hitting , find_throwing , Forward_kinematic
from jax import numpy as jnp



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_robot = np.load("Data_MPC/robot_data.npz")
time = data_robot['ts']
robot_q = data_robot['qs']
robot_dq = data_robot['qs_t']
robot_ddq = data_robot['qs_tt']
robot_u = data_robot['us']

print (time.shape,robot_q.shape , robot_ddq.shape , robot_u.shape)




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
idx1 =80
idx2 = 90
DT =0.002


for i in range (4):
    plt.figure()
    plt.plot(time[idx1] , robot_q[idx1, :,i] , lw=1.6 , label = f"robot position idx ${idx1}$" , color = 'b' )
    plt.plot(time[idx2] , robot_q[idx2, :,i] , lw=1.6 , label = f"robot position idx ${idx2}$" , color = 'r')
    plt.xlabel("t [s]")
    plt.ylabel(rf"$q_{i+1}$")
    plt.title("robot position")
    plt.legend()
    plt.grid(True, alpha=0.3)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_total, T, _ = robot_q.shape
labels = ["q1", "q2", "q3" , "q4"]

idx = np.random.randint(0, N_total, size=20)

for d in range(4):
    plt.figure(figsize=(10, 5))
    for k in range(20):  
        i = idx[k]        
        plt.plot(time[i,:1500], robot_q[i, :, d], linewidth=1)
    plt.xlabel("time")
    plt.ylabel(f" {labels[d]}")
    plt.title(f"robot_q states  ({labels[d]}), 20 rollouts")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X = np.concatenate(
    [robot_q, robot_dq, robot_ddq, robot_u],
    axis=1,
)

unique_X, unique_indices = np.unique(
    X,
    axis=0,
    return_index=True,
)

print(f"Original samples : {len(X)}")
print(f"Unique samples   : {len(unique_X)}")
print(f"Duplicates       : {len(X) - len(unique_X)}")
# %%
