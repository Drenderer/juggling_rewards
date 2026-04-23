#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import jax
from jax import random as jr
from jax import numpy as jnp
from jaxtyping import Array
from dynax import ODESolver ,normalization_coefficients 
import klax
import matplotlib.pyplot as plt
import optax
from helping_function import find_throwing , ball_free_flight_trajecotry
from normalize import coefficients , Normalization

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_robot = np.load("Data/Initial_data/first dataset/robot_throwing.npz")
time = data_robot['ts']
robot_q = data_robot['qs']
robot_dq = data_robot['qs_t']
robot_ddq = data_robot['qs_tt']
robot_u = data_robot['us']

print (time.shape,robot_q.shape , robot_ddq.shape , robot_u.shape)

data_ball = np.load("Data/Initial_data/first dataset/ball_throwing.npz")
ball_x = data_ball['ball_x']
ball_dx = data_ball['ball_xt']
ball_f = data_ball['f']
ball_c = data_ball['c']


print (ball_x.shape , ball_dx.shape , ball_f.shape)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rest_time = 50
key = jr.PRNGKey(0)
N ,T , _ = robot_q.shape

robot_y , ball_y , robot_t= [] , [] , []

for i in range (N):
    if i%5==0:
        idx = find_throwing(ball_c[i] , rest_time )
        key, subkey = jr.split(key)
        robot_data = np.concatenate([robot_q[i] , robot_dq[i] , robot_ddq[i] , robot_u[i]] , axis = -1)
        robot_data = robot_data [idx - 99 : idx + 1] 
        ball_data = np.concatenate([ball_x[i] , ball_dx[i]], axis=-1)
        ball_data = ball_data[idx-99 : idx+1]
        t = time[i , idx - 99 : idx + 1]


        robot_y. append(robot_data)
        ball_y.append(ball_data)
        robot_t.append(t)



robot_t = np.array(robot_t)
robot_y = np.array(robot_y)
ball_y = np.array(ball_y)

print (robot_t.shape ,robot_y.shape , ball_y.shape)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from jaxtyping import Array

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

print (alpha_x, tau_x , alpha_u)

robot_y[:,:,0:4] = norm.transform_qs(robot_y[:,:,0:4])
robot_y[:,:,4:8] = norm.transform_q_ts(robot_y[:,:,4:8])
robot_y[:,:,8:12] = norm.transform_q_tts(robot_y[:,:,8:12])
robot_y[:,:,12:16] = norm.transform_taus(robot_y[:,:,12:16])
robot_t = norm.transform_ts(robot_t)
robot_y = robot_y[: , : , :8]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
''''
mean_ball_x = _semi_flatten(ball_y[: , : ,0:3]).mean(axis=0)
std_ball_x = _semi_flatten(ball_y[: , : ,0:3]).std(axis=0)
std_ball_dx = _semi_flatten(ball_y[: , : ,3:6]).std(axis=0)

alpha_q, tau_q = normalization_coefficients(std_ball_x, std_ball_dx, std_a = None, tol=1e-6)
norm_ball = Normalization (mean_q=mean_ball_x  , alpha_q=alpha_q , tau_q= tau_q 
                           , mean_u=None , alpha_u=None)

ball_norm_y = norm_ball.transform_qs(ball_y[: , : ,0:3])
ball_norm_dy= norm_ball.transform_q_ts(ball_y[: , : ,3:6])

ball_norm = jnp.concat([ball_norm_y , ball_norm_dy] , axis=-1)
print (alpha_q, tau_q)
'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train_ratio = 0.8
N = robot_y.shape[0]
rng = np.random.default_rng(seed=42)   
perm = rng.permutation(N)

N_train = int(train_ratio * N)

train_idx = perm[:N_train]
test_idx  = perm[N_train:]

robot_time_train = robot_t[train_idx]
robot_train = robot_y[train_idx]
#ball_train  = ball_norm[train_idx]
ball_train  = ball_y[train_idx]

robot_time_test = robot_t[test_idx]
robot_test = robot_y[test_idx]
#ball_test  = ball_norm[test_idx]
ball_test  = ball_y[test_idx]

print("Train:", robot_train.shape, ball_train.shape)
print("Test :", robot_test.shape, ball_test.shape)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x_train = jnp.reshape(robot_train , (-1,8))
y_train = jnp.reshape(ball_train , (-1 ,6))

x_test = jnp.reshape(robot_test , (-1,8))
y_test = jnp.reshape(ball_test , (-1 ,6))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def loss_function (model , batch_data , batch_axis):
       x_batch , y_batch = batch_data
       y_pred = jax.vmap(model)(x_batch)
       return jnp.mean(jnp.square(y_pred - y_batch))


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model = klax.nn.MLP( in_size=8 , out_size= 6 , width_sizes= [64,64] , key =key)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model , hist =klax.fit(
                    model,
                    (x_train , y_train),
                    validation_data= (x_test , y_test),
                    batch_size=64,
                    loss_fn=loss_function,
                    optimizer=optax.adam(3e-4),
                    steps = 500000,
                    key=jr.key(0)
)

hist.plot()
plt.show()




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model_ = klax.finalize(model)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


DT = 0.002
ts = jnp.arange(1500) * DT
q_total_true , q_total_pred , total_true_time , total_pred_time = [] , [] , [] , []

for idx in range (1000):
    if idx % 100 == 0:
        print(idx)
    true_initial = ball_test [idx , -1 , :]
    pred_initial = model_(robot_test[idx , -1 , : ])

    #true_initial = true_initial.at[:3].set(norm_ball.inverse_transform_qs(true_initial[:3]))
    #true_initial = true_initial.at[3:6].set(norm_ball.inverse_transform_q_ts(true_initial[3:6]))

    #pred_initial = pred_initial.at[:3].set(norm_ball.inverse_transform_qs(pred_initial[:3]))
    #pred_initial = pred_initial.at[3:6].set(norm_ball.inverse_transform_q_ts(pred_initial[3:6]))


    q_true , true_time = ball_free_flight_trajecotry(true_initial , ts)
    q_pred , pred_time = ball_free_flight_trajecotry(pred_initial , ts)

    q_total_true.append(q_true)
    q_total_pred.append(q_pred)
    total_true_time.append(true_time)
    total_pred_time.append(pred_time)

q_total_true = jnp.array(q_total_true)
q_total_pred = jnp.array(q_total_pred)
total_true_time = jnp.array(total_true_time)
total_pred_time = jnp.array(total_pred_time)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

error = []
error_idx = []
for idx in range(1000):
    time = total_true_time[idx]
    error_x = jnp.abs(q_total_true[idx ,time , 0] - q_total_pred[idx ,time , 0]) 
    error_y = jnp.abs(q_total_true[idx ,time , 1] - q_total_pred[idx ,time , 1]) 
    error_z = jnp.abs(q_total_true[idx ,time , 2] - q_total_pred[idx ,time , 2])

    err1 = jnp.sqrt (error_x**2 + error_y**2 + error_z**2)

    error. append(err1)

error = jnp.array(error)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
D = 0.075  # Diameter of ball
R = D/2
error = jnp.asarray(error)

error_small = error[error <= D]
error_large = error[error > D]

plt.figure(figsize=(10, 5))
bins = jnp.linspace(error.min(), error.max(), 50)

plt.hist(error_small, bins=bins, color="tab:blue", alpha=0.7, label=f"Error ≤ {D}")
plt.hist(error_large, bins=bins, color="tab:red",  alpha=0.7, label=f"Error > {D}")

plt.axvline(D, color="black", linestyle="--", linewidth=2, label="Threshold")

plt.xlabel("Distance error [m]")
plt.ylabel("Number of samples")
plt.title("Distribution of position errors at Hitting Ground ")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_total = error.shape[0]
num_below = jnp.sum(error <= D)
percentage_below = 100.0 * num_below / num_total

print(f"Threshold R = {D} m")
print(f"Samples below R: {int(num_below)} / {num_total}")
print(f"Percentage below R: {float(percentage_below):.2f}%")
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
error0 = []
for idx in range(1000):
    error_x = jnp.abs(q_total_true[idx ,0 , 0] - q_total_pred[idx ,0 , 0]) 
    error_y = jnp.abs(q_total_true[idx ,0, 1] - q_total_pred[idx ,0 , 1]) 
    error_z = jnp.abs(q_total_true[idx ,0 , 2] - q_total_pred[idx ,0 , 2])

    err0 = jnp.sqrt (error_x**2 + error_y**2 + error_z**2)
    error0. append(err0)

error0 = jnp.array(error0)
error0_small = error0[error0 <= D]
error0_large = error0[error0 > D]
bins = jnp.linspace(error0.min(), error0.max(), 50)

plt.figure(figsize=(10, 5))
plt.hist(error0_small, bins=bins, color="tab:blue", alpha=0.7, label=f"Error ≤ {D}")
plt.hist(error0_large, bins=bins, color="tab:red",  alpha=0.7, label=f"Error > {D}")
plt.axvline(D, color="black", linestyle="--", linewidth=2, label="Threshold")
plt.xlabel("Distance error [m]")
plt.ylabel("Number of samples")
plt.title("Distribution of position errors  at Throwing Time")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_total = error0.shape[0]
num_below = jnp.sum(error0 <= D)
percentage_below = 100.0 * num_below / num_total

print(f"Threshold D = {D} m")
print(f"Samples below D: {int(num_below)} / {num_total}")
print(f"Percentage below D: {float(percentage_below):.2f}%")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx = 421

pred = jnp.array(jax.vmap (model_)(robot_test[idx]))

T = pred.shape[0]
dt = 0.002
time = (jnp.arange(T) - (T - 1)) * dt

fig, axes = plt.subplots(3, 2, figsize=(12, 8))
axes = axes.flatten()
labels = ["x", "y", "z" , "vx" , "vy" , "vz"]
plt.suptitle(f"Ball state prediction before throw - sample {idx}", fontsize=14)
for i in range(6):
    axes[i].plot(time, ball_test[idx , :, i], label="Ground Truth")
    axes[i].plot(time, pred[:, i], "--", label="Prediction")
    axes[i].set_title(f"Ball state dimension {i}")
    axes[i].set_xlabel("Time step")
    axes[i].set_ylabel(labels[i])
    axes[i].set_title(labels[i])
    axes[i].legend()

plt.tight_layout()
plt.show()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

time = total_true_time[idx]


fig, axes = plt.subplots(3, 2, figsize=(12, 8))  # SAME as previous
axes = axes.flatten()
plt.suptitle(f"Ball state prediction after throw - sample {idx}", fontsize=14)
for d in range(6):
    axes[d].plot(ts[:time], q_total_true[idx, :time, d], label="True value")
    axes[d].plot(ts[:time], q_total_pred[idx, :time, d], "--", label="Prediction")
    
    axes[d].set_xlabel("time [s]")
    axes[d].set_ylabel(labels[d])
    axes[d].set_title(labels[d])
    axes[d].grid(alpha=0.3)
    axes[d].legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

error_x = jnp.abs(q_total_true[idx ,time , 0] - q_total_pred[idx ,time , 0]) 
error_y = jnp.abs(q_total_true[idx ,time, 1] - q_total_pred[idx ,time, 1]) 
error_z = jnp.abs(q_total_true[idx ,time , 2] - q_total_pred[idx ,time , 2])

err_idx = jnp.sqrt (error_x**2 + error_y**2 + error_z**2)

print ("error in x:" , error_x)
print ("error in y:" , error_y)
print ("error in z:" , error_z)
print ("distance error" , err_idx)


error_x0 = jnp.abs(q_total_true[idx ,0 , 0] - q_total_pred[idx ,0 , 0]) 
error_y0 = jnp.abs(q_total_true[idx ,0, 1] - q_total_pred[idx ,0, 1]) 
error_z0= jnp.abs(q_total_true[idx ,0 , 2] - q_total_pred[idx ,0 , 2])


err0_idx = jnp.sqrt (error_x0**2 + error_y0**2 + error_z0**2)

print ("error in x0:" , error_x0)
print ("error in y0:" , error_y0)
print ("error in z0:" , error_z0)
print ("distance error at throw time" , err0_idx)


error_dx0 = jnp.abs(q_total_true[idx ,0 , 3] - q_total_pred[idx ,0 , 3]) 
error_dy0 = jnp.abs(q_total_true[idx ,0, 4] - q_total_pred[idx ,0, 4]) 
error_dz0= jnp.abs(q_total_true[idx ,0 , 5] - q_total_pred[idx ,0 , 5])

print ("error in vx0:" , error_dx0)
print ("error in vy0:" , error_dy0)
print ("error in vz0:" , error_dz0)


# %%
