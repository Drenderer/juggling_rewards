#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import jax
from jax import random as jr
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from helping_function import hitting_ground , find_throwing

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_robot = np.load("Data/robot_throwing.npz")
time = data_robot['ts']
robot_q = data_robot['qs']
robot_dq = data_robot['qs_t']
robot_ddq = data_robot['qs_tt']
robot_u = data_robot['us']

print (time.shape,robot_q.shape , robot_ddq.shape , robot_u.shape)

data_ball = np.load("Data/ball_throwing.npz")
ball_x = data_ball['ball_x']
ball_dx = data_ball['ball_xt']
ball_f = data_ball['f']
ball_c = data_ball['c']


print (ball_x.shape , ball_dx.shape , ball_f.shape)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def build_dataset(time ,robot_q , robot_dq , robot_ddq , robot_u,
                  ball_x , ball_dx , ball_f , window_size = 30 , rest_time =50):
    """"
    return
    robot_y : (N , window_size , 16)      robot data [q ,dq , ddq , u] for last window size
    tobot_t :(N , window_size)            time of last window size
    ball_y : (N , 10 , 6)                 ball position and velocity from t=T_throw until T_throw + 10
    index : (N ,)                         time of throw
    """
    key = jr.PRNGKey(0)
    N_total, T, _ = robot_q.shape
    robot_y , robot_t , ball_y , index , contact = [] , [] , [] , [] , []
    for i in range(N_total):

        idx = find_throwing(ball_f[i] , rest_time )
        key, subkey = jr.split(key)
        r = jr.randint(subkey, (), 0, 20)
        robot_data = np.concatenate([robot_q[i] , robot_dq[i] , robot_ddq[i] , robot_u[i]] , axis =-1)
        robot_data = robot_data [idx - window_size +r + 1 : idx + r + 1]             # cutting the last window_size steps for robot 
        ball_data = np.concatenate([ball_x[i] , ball_dx[i]], axis=-1)
        ball_data = ball_data[idx]
        t = time[i , idx - window_size + r + 1 : idx + r + 1]
        c = jnp.full((window_size,) , 0)
        c = c.at[window_size - r - 1].set(1)

        contact.append(c)
        robot_t.append(t)
        robot_y.append(robot_data)
        ball_y.append(ball_data)
        index.append(idx)
    
    return robot_t, robot_y , ball_y , contact, index

    
robot_t , robot_y , ball_y , contact, index = build_dataset(time,robot_q , robot_dq , robot_ddq , robot_u,ball_x , ball_dx , ball_f)
contact = np.array(contact)
robot_t = np.array(robot_t)
robot_y = np.array(robot_y)
ball_y = np.array(ball_y)
index = np.array (index)

print (robot_t.shape, robot_y.shape , ball_y.shape , contact.shape , index.shape)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import matplotlib.pyplot as plt
throw_pos = jnp.argmax(contact == 1, axis=1)

plt.figure(figsize=(7, 4))

plt.hist(throw_pos, bins=20, edgecolor="black")
plt.xlabel("Throw time index inside window")
plt.ylabel("Number of samples")
plt.title("Distribution of throw time inside window")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
np.savez ('Data/prepared_data_with_time.npz' , robot_t= robot_t ,robot_y = robot_y  , ball_y = ball_y ,contact = contact, index = index)
# %%
