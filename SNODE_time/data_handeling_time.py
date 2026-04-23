#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import jax
from jax import random as jr
from jax import numpy as jnp
from jaxtyping import Array, PyTree
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append("..") 
from helping_function import hitting_ground, find_throwing
from normalize import Normalization, coefficients

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
base_path = Path(__file__).resolve().parent
data_path = base_path.parent / "Data/clean_data/second dataset/robot_throwing.npz"
data_robot = np.load(data_path)
time = data_robot['time']
robot_q = data_robot['robot_q']
robot_dq = data_robot['robot_dq']
robot_ddq = data_robot['robot_ddq']
robot_u = data_robot['robot_u']

print (time.shape,robot_q.shape , robot_ddq.shape , robot_u.shape)


base_path = Path(__file__).resolve().parent
data_path = base_path.parent / "Data/clean_data/second dataset/ball_throwing.npz"
data_ball = np.load(data_path)

ball_x = data_ball['ball_q']
ball_dx = data_ball['ball_dq']
ball_f = data_ball['ball_f']
ball_c = data_ball['ball_c']


print (ball_x.shape , ball_dx.shape , ball_f.shape)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def prepare_dataset_time(time ,robot_q , robot_dq , robot_ddq , robot_u,
                  ball_x , ball_dx , ball_c , window_size = 30 , rest_time =50):
    """"
    this function build dataset for training state_less NODE which can predict only time


    return
    robot_y : (N , window_size , 16)      robot data [q ,dq , ddq , u] for last window size
    tobot_t :(N , window_size)            time of last window size
    ball_y : (N , 6)                      ball position and velocity from t=T_throw 
    index : (N ,)                         time of throw
    """

    key = jr.PRNGKey(0)
    N_total, T, _ = robot_q.shape
    robot_y , robot_t , ball_y , contact, index  = [] , [] , [] , [] , []
    for i in range(N_total):
        idx = find_throwing(ball_c[i] , rest_time)
        key, subkey = jr.split(key)
        r = jr.randint(subkey, (), 0, 20)
        robot_data = np.concatenate([robot_q[i] , robot_dq[i] , robot_ddq[i] , robot_u[i]] , axis =-1)
        ball_data = np.concatenate([ball_x[i] , ball_dx[i]], axis=-1)


        if i%6==0:
            robot_data = robot_data[ idx - window_size - r -1: idx -r -1]                # cutting the window without throw  
            ball_data = ball_data[idx-r]
            t = time[i ,idx - r - window_size -1 : idx - r -1 ] 
            c = jnp.full((window_size,) , 0)

        else:
            robot_data = robot_data [idx - window_size + r + 1 : idx + r + 1]             # cutting the window with throw  
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

    
robot_t , robot_y , ball_y, contact, index = prepare_dataset_time(time,robot_q , robot_dq , robot_ddq 
                                                             , robot_u,ball_x , ball_dx , ball_c)
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

plt.hist(throw_pos, bins=30, edgecolor="black")
plt.xlabel("Throw time index inside window")
plt.ylabel("Number of samples")
plt.title("Distribution of throw time inside window")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def _semi_flatten(x: Array) -> Array:
            return x.reshape(-1, x.shape[-1])


mean_x = _semi_flatten(robot_y[:,:,0:4]).mean(axis=0)
std_x = _semi_flatten(robot_y[:,:,0:4]).std(axis=0)
std_dx = _semi_flatten(robot_y[:,:,4:8]).std(axis=0)
std_ddx = _semi_flatten(robot_y[:,:,8:12]).std(axis=0)
mean_u = _semi_flatten(robot_y[:,:,12:16]).mean(axis=0)
std_u = _semi_flatten(robot_y[:,:,12:16]).std(axis=0)

alpha_x, tau_x , alpha_u = coefficients (mean_x , std_x , std_u ,std_dx , std_ddx)

norm = Normalization (mean_q=mean_x, alpha_q=alpha_x, tau_q=tau_x,
                      mean_u=mean_u, alpha_u=alpha_u)

robot_y[:,:,0:4] = norm.transform_qs(robot_y[:,:,0:4])
robot_y[:,:,4:8] = norm.transform_q_ts(robot_y[:,:,4:8])
robot_y[:,:,8:12] = norm.transform_q_tts(robot_y[:,:,8:12])
robot_y[:,:,12:16] = norm.transform_taus(robot_y[:,:,12:16])
robot_t = norm.transform_ts(robot_t)
robot_y = robot_y[: , : , :8]



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_ratio = 0.8
N = robot_y.shape[0]
rng = np.random.default_rng(seed=42)   
perm = rng.permutation(N)

N_train = int(train_ratio * N)

train_idx = perm[:N_train]
test_idx  = perm[N_train:]

robot_time_train = robot_t[train_idx]
robot_train = robot_y[train_idx]
contact_train = contact[train_idx]
ball_train  = ball_y[train_idx]
index_train = index[train_idx]

robot_time_test = robot_t[test_idx]
robot_test = robot_y[test_idx]
contact_test = contact[test_idx]
ball_test  = ball_y[test_idx]
index_test = index[test_idx]

print("Train:", robot_train.shape, ball_train.shape , contact_train.shape)
print("Test :", robot_test.shape, ball_test.shape  , contact_test.shape)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
np.savez ('prepared_samples/train_data.npz' , robot_t= robot_time_train ,robot_y = robot_train  , 
                                             ball_y = ball_train , contact = contact_train , index = index_train)

np.savez ('prepared_samples/test_data.npz' , robot_t= robot_time_test ,robot_y = robot_test  , 
                                             ball_y = ball_test  , contact= contact_test ,  index = index_test)
# %%
