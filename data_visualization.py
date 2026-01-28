#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import matplotlib.pyplot as plt


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_robot = np.load("Data/robot_throwing.npz")
time = data_robot['ts']
robot_q = data_robot['qs']
robot_dq = data_robot['qs_t']
robot_ddq = data_robot['qs_tt']
robot_u = data_robot['us']

print (time.shape,robot_q.shape , robot_ddq.shape , robot_u.shape)

data_ball = np.load("Data/ball_throwing.npz")
ball_q = data_ball['ball_x']
ball_dq = data_ball['ball_xt']
ball_f = data_ball['f']
ball_c = data_ball['c']


print (ball_q.shape , ball_dq.shape , ball_f.shape)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def find_throwing(ball_f , rest_time):
    idx_throw = np.where(ball_f[rest_time:] < 1e-04)[0]
    flag = False
    i=0
    while flag == False:
        if idx_throw[i+10] - idx_throw[i] == 10:
            idx_throw = int(rest_time + idx_throw[i])
            flag = True
        i=i+1
    return idx_throw


def hitting_ground(ball_q):
    idx_floor = np.where(ball_q[:, 2] - 0.038 < 1e-4)[0]
    return idx_floor[0]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
idx1 = 100
idx2 = 5200
DT =0.002
idx_throw1 = find_throwing(ball_f[idx1] , rest_time = 50)
idx_throw2 = find_throwing(ball_f[idx2] , rest_time = 50)

idx_hit1 = hitting_ground(ball_q[idx1])
idx_hit2 = hitting_ground(ball_q[idx2])

print ("throwing time for trajectory" , idx1 , "is" , idx_throw1*DT)
print ("throwing time for trajectory" , idx2 , "is" , idx_throw2*DT)

print ("hitting ground time for trajectory" , idx1 , "is" , idx_hit1*DT)
print ("throwing time for trajectory" , idx2 , "is" , idx_hit2*DT)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i in range (3):
    plt.figure()
    plt.plot(time[idx1,:idx_hit1] , ball_q[idx1, :idx_hit1,i] , lw=1.6 , label = f"ball position idx ${idx1}$" , color = 'b')
    plt.plot(time[idx2,:idx_hit2] , ball_q[idx2, :idx_hit2,i] , lw=1.6 , label = f"ball position idx ${idx2}$" , color ='r')
    plt.axvline(time[idx1,idx_throw1], linestyle='--', linewidth=1.0 , color='b')
    plt.axvline(time[idx2,idx_throw2], linestyle='--', linewidth=1.0 , color='r')
    plt.axvline(time[idx1,idx_hit1], linestyle='--', linewidth=1.0 , color='b')
    plt.axvline(time[idx2,idx_hit2], linestyle='--', linewidth=1.0 , color='r')
    labels = ["x [m]", "y [m]", "z [m]"]
    plt.xlabel("t [s]")
    plt.ylabel(labels[i])
    plt.title("ball position")
    plt.legend()
    plt.grid(True, alpha=0.3)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plt.figure()
plt.plot(time[idx1,:1000] , ball_f[idx1, :1000] , lw=1.6 , label = f"ball force idx ${idx1}$" , color = 'b')
plt.plot(time[idx2,:1000] , ball_f[idx2, :1000] , lw=1.6 , label = f"ball force idx ${idx2}$" , color = 'r')
plt.axvline(time[idx1,idx_throw1], linestyle='--', linewidth=1.0 , color='b')
plt.axvline(time[idx2,idx_throw2], linestyle='--', linewidth=1.0 , color='r')
plt.xlabel("t [s]")
plt.ylabel(rf"$fn$")
plt.title("ball force")
plt.legend()
plt.grid(True, alpha=0.3)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i in range (4):
    plt.figure()
    plt.plot(time[idx1,:idx_throw1] , robot_q[idx1, :idx_throw1,i] , lw=1.6 , label = f"robot position idx ${idx1}$" , color = 'b' )
    plt.plot(time[idx2,:idx_throw2] , robot_q[idx2, :idx_throw2,i] , lw=1.6 , label = f"robot position idx ${idx2}$" , color = 'r')
    plt.axvline(time[idx1,idx_throw1], linestyle='--', linewidth=1.0 , color='b')
    plt.axvline(time[idx2,idx_throw2], linestyle='--', linewidth=1.0 , color='r')
    plt.xlabel("t [s]")
    plt.ylabel(rf"$q_{i+1}$")
    plt.title("robot position")
    plt.legend()
    plt.grid(True, alpha=0.3)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

labels = ["x", "y", "z"]

idx = np.random.randint(0, 40000, size=50)

for d in range(3):
    plt.figure(figsize=(10, 5))
    for k in range(50):  
        i = idx[k]        
        plt.plot(time[i,:1000], ball_q[i, :1000, d], linewidth=1)
    plt.xlabel("time")
    plt.ylabel(f" {labels[d]}")
    plt.title(f"Ball position  ({labels[d]}), 50 rollouts")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_total, T, _ = robot_q.shape
time_throw = []
for i in range(N_total):
    idx = find_throwing(ball_f[i] , rest_time = 50)
    time_throw.append(idx)

time_throw = np.array(time_throw)

plt.figure(figsize=(10, 5))
plt.hist(time_throw*DT, bins=50)  # bins can be tuned
plt.xlabel("Throw time")
plt.ylabel("Number of trajectories")
plt.title("Distribution of throw times")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx_max = np.argmax(time_throw)
idx_min = np.argmin(time_throw)
t_max = time_throw[idx_max]
t_min = time_throw[idx_min]
print("Index of trajectory with latest throw:", idx_max)
print("Latest throw time index:", t_max)

print("Index of trajectory with earliest throw:", idx_min)
print("Earliest throw time index:", t_min)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from jax import numpy as jnp
q11 = jnp.array([-0.5 , -0.3 , -0.1 , 0 , 0.1 , 0.3 , 0.5])
w1 = jnp.array([-0.3 , -0.2 , -0.1 , 0.1 , 0.2 , 0.3])
q21 = jnp.array([0.8 , 0.9 , 1 , 1.1 , 1.2 , 1.3])
w2 = jnp.array([ 0 , 0.1 ,0.15 , 0.2 , 0.25 , 0.3])
q41 = jnp.array([0.9 , 1 , 1.1 , 1.2 , 1.3 , 1.4])
w4 = jnp.array ([0.1 ,0.2 , 0.3 , 0.4 ,0.5])

Q11, W1, Q21, W2, Q41, W4 = jnp.meshgrid(q11, w1, q21, w2, q41, w4, indexing="ij")
P = jnp.stack([Q11, W1, Q21, W2, Q41, W4], axis=-1).reshape(-1, 6)
mask = (P[:,3] !=0) | (P[: ,5]>0.3)         #omiting all w2= 0 and w4=0.1 , 0.2 ,0.3
P = P[mask]

print (P[t_max])
print (P[t_min])
# %%
