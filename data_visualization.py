#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import jax
import matplotlib.pyplot as plt
from helping_function import hitting , find_throwing , Forward_kinematic
from jax import numpy as jnp



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_robot = np.load("Data/clean_data/third dataset/robot_throwing_train.npz")
time = data_robot['time']
robot_q = data_robot['robot_q']
robot_dq = data_robot['robot_dq']
robot_ddq = data_robot['robot_ddq']
robot_u = data_robot['robot_u']

print (time.shape,robot_q.shape , robot_ddq.shape , robot_u.shape)

data_ball = np.load("Data/clean_data/third dataset/ball_throwing_train.npz")
ball_q = data_ball['ball_q']
ball_dq = data_ball['ball_dq']
ball_f = data_ball['ball_f']
ball_c = data_ball['ball_c']


print (ball_q.shape , ball_dq.shape , ball_f.shape)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
idx1 =18
idx2 = 122
DT =0.002
#print (np.where(ball_c[idx2] == 1)[0])
idx_throw1 = find_throwing(ball_c[idx1] , rest_time = 50)
idx_throw2 = find_throwing(ball_c[idx2] , rest_time = 50)

idx_hit1 = hitting(ball_c[idx1] , idx_throw1)
idx_hit2 = hitting(ball_c[idx2] , idx_throw2)

print ("throwing time for trajectory" , idx1 , "is" , idx_throw1*DT)
print ("throwing time for trajectory" , idx2 , "is" , idx_throw2*DT)

print ("hitting  time for trajectory" , idx1 , "is" , idx_hit1*DT)
print ("hitting time for trajectory" , idx2 , "is" , idx_hit2*DT)

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
plt.plot(time[idx2,:1500] , ball_c[idx2, :1500] , lw=1.6 , label = f"ball contact idx ${idx2}$" , color = 'r')
plt.axvline(time[idx2,idx_throw2], linestyle='--', linewidth=1.0 , color='b')
plt.axvline(time[idx2,idx_hit2], linestyle='--', linewidth=1.0 , color='b')
plt.xlabel("t [s]")
plt.ylabel(rf"$contact value$")
plt.title("ball contact")
plt.legend()
plt.grid(True, alpha=0.3)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i in range (4):
    plt.figure()
    plt.plot(time[idx1,:idx_throw1] , robot_q[idx1, :idx_throw1,i] , lw=1.6 , label = f"robot position idx ${idx1}$" , color = 'b' )
    plt.plot(time[idx2,:idx_throw2] , robot_q[idx2, :idx_throw2,i] , lw=1.6 , label = f"robot position idx ${idx2}$" , color = 'r')
    plt.plot(time[idx1, idx_throw1-20:idx_throw1],robot_q[idx1, idx_throw1-20:idx_throw1, i],'ob', markersize=4)
    plt.plot(time[idx2, idx_throw2-20:idx_throw2],robot_q[idx2, idx_throw2-20:idx_throw2, i],'or', markersize=4)
    plt.axvline(time[idx1,idx_throw1], linestyle='--', linewidth=1.0 , color='b')
    plt.axvline(time[idx2,idx_throw2], linestyle='--', linewidth=1.0 , color='r')
    plt.xlabel("t [s]")
    plt.ylabel(rf"$q_{i+1}$")
    plt.title("robot position")
    plt.legend()
    plt.grid(True, alpha=0.3)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_total, T, _ = robot_q.shape
labels = ["x", "y", "z"]

idx = np.random.randint(0, N_total, size=50)

for d in range(3):
    plt.figure(figsize=(10, 5))
    for k in range(50):  
        i = idx[k]        
        plt.plot(time[i,:1500], ball_q[i, :1500, d], linewidth=1)
    plt.xlabel("time")
    plt.ylabel(f" {labels[d]}")
    plt.title(f"Ball position  ({labels[d]}), 50 rollouts")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def fk_position(q):
    matrix = Forward_kinematic(q)
    return matrix[:3, 3]

fk_time = jax.jit(jax.vmap(fk_position, in_axes=0))

window = 200
offsets = jnp.arange(window) - (window - 1)   # [-199, ..., 0]

all_distances = []
valid_indices = []

N = robot_q.shape[0]

for idx in range(N):
    idx_throw = find_throwing(ball_c[idx], rest_time=50)

    if idx_throw < window - 1:
        continue

    robot_x_sample = fk_time(robot_q[idx])   # (T, 3)

    gather_idx = idx_throw + offsets         # (200,)

    ball_window = ball_q[idx, gather_idx, :3]     
    robot_window = robot_x_sample[gather_idx]     

    distances = jnp.linalg.norm(ball_window - robot_window, axis=1)
    all_distances.append(distances)
    valid_indices.append(idx)

all_distances = jnp.array(all_distances)
valid_indices = jnp.array(valid_indices)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mean_distances = np.array(jnp.mean(all_distances, axis=0))
max_distances  = np.array(jnp.max(all_distances, axis=0))
min_distances  = np.array(jnp.min(all_distances, axis=0))
median_distances = np.array(jnp.median(all_distances, axis=0))
q25 = np.array(jnp.percentile(all_distances, 25, axis=0))
q75 = np.array(jnp.percentile(all_distances, 75, axis=0))

steps = np.arange(-199, 1)

plt.figure(figsize=(9,5))

plt.plot(steps, mean_distances, label="Mean distance")
plt.plot(steps, median_distances, label="Median distance")
plt.plot(steps, max_distances, label="Max distance", linestyle="--")
plt.plot(steps, min_distances, label="Min distance", linestyle="--")

plt.fill_between(steps, q25, q75, alpha=0.3, label="25%-75% band")

plt.axvline(0, linestyle=":")
plt.xlabel("Steps before throw")
plt.ylabel("Distance (ball ↔ cup)")
plt.title("Ball–Cup Distance Before Throw")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def fk_position(q):
    matrix = Forward_kinematic(q)
    return matrix[:3, 3]

jac_fn = jax.jacobian(fk_position)

def compute_velocity(q, dq):
    J = jac_fn(q)
    return J @ dq

def full_state(q , dq):

    pos = jax.vmap(jax.vmap(fk_position))(q)                 # (N, T, 3)
    vel = jax.vmap(jax.vmap(compute_velocity))(q, dq)        # (N, T, 3)
    return jnp.concatenate([pos , vel] , axis =-1)

robot_x = full_state(robot_q , robot_dq)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx =122
step = 10
idx_throw = find_throwing(ball_c[idx] , rest_time = 50)
for i in range (3):
    plt.figure()
    plt.plot(time[idx,:idx_throw] , ball_q[idx, :idx_throw,i] , lw=1.6 , label = f"ball position idx ${idx}$" , color = 'b')
    plt.plot(time[idx,:idx_throw] , robot_x[ idx, :idx_throw,i] , lw=1.6 , label = f"robot cup position ${idx}$" , color ='r')
    plt.plot(time[idx, :idx_throw:step],ball_q[idx, :idx_throw:step, i],'ob', markersize=4)
    plt.plot(time[idx, :idx_throw:step],robot_x[idx, :idx_throw:step, i],'or', markersize=4)
    plt.axvline(time[idx,idx_throw], linestyle='--', linewidth=1.0 , color='b')
    labels = ["x [m]", "y [m]", "z [m]" , "Vx" , "Vy" , "Vz"]
    plt.xlabel("t [s]")
    plt.ylabel(labels[i])
    plt.title("ball states")
    plt.legend()
    plt.grid(True, alpha=0.3)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print ("ïnitial values")
print (ball_q[idx ,0,:])
print (robot_x[idx,0,:])

print ("100 steps before throw values")
print (ball_q[idx ,idx_throw-100,:])
print (robot_x[idx,idx_throw-100,:])
print ("error at throw time")

print (jnp.abs (ball_q[idx ,idx_throw-100,:] - robot_x[idx,idx_throw-100,:]))
print(jnp.linalg.norm(ball_q[idx ,idx_throw-100,:] - robot_x[idx, idx_throw-100,:]))
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
index_throwing = []
for i in range (N_total):
    throw = find_throwing(ball_c[i] , rest_time=50)
    index_throwing.append(throw)

index_throwing = jnp.array(index_throwing)

plt.figure(figsize=(10, 5))
bins = jnp.linspace(index_throwing.min(), index_throwing.max(), 100)
plt.hist(index_throwing, bins=bins, color="tab:blue", alpha=0.7)
plt.xlabel("throwing index")
plt.ylabel("Number of samples")
plt.title("Distribution of throwing index in dataset")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print(index_throwing.min())
print(index_throwing.max())
# %%
