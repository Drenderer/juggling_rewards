#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import jax
import numpy as np
import matplotlib.pyplot as plt
import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, PyTree
from dynax import ODESolver
import klax
from node import NODE
from helping_function import hitting_ground , find_throwing , ball_free_flight_trajecotry
from normalize import Normalization, coefficients

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = np.load('Data/prepared_data_with_time.npz')
robot_t = data['robot_t']
robot_y = data['robot_y']
contact = data['contact']
ball_y = data['ball_y']
index = data['index']

print (robot_t.shape ,robot_y.shape , ball_y.shape , index.shape)

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
robot_y = robot_y[: , : , :12]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_indexes = np.load('Data/train_idx.npz')
train_idx = train_indexes['train_idx']

test_indexes = np.load('Data/test_idx.npz')
test_idx = test_indexes['test_idx']

robot_time_train = robot_t[train_idx]
robot_train = robot_y[train_idx]
contact_train = contact[train_idx]
ball_train  = ball_y[train_idx]

robot_time_test = robot_t[test_idx]
contact_test = contact[test_idx]
robot_test = robot_y[test_idx]
ball_test  = ball_y[test_idx]


print("Train:", robot_train.shape, ball_train.shape)
print("Test :", robot_test.shape, ball_test.shape)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# [ ... , -2DT , -DT , 0.000] for each sample
time_train = robot_time_train - robot_time_train[:, -1][:, None]  
time_test = robot_time_test - robot_time_test[:, -1][:, None]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Model (eqx.Module):
    node: NODE
    ode: ODESolver
    nn:eqx.nn.MLP
    scorer : eqx.nn.MLP
    latent_dim: int = eqx.field(static=True)

    def __init__ (self, node  , ode , key , latent_dim):
        k1, k2 = jr.split(key, 2)
        self.node = node
        self.ode = ode
        self.latent_dim = latent_dim
        self.nn = eqx.nn.MLP(
            in_size = latent_dim,
            out_size= 6,
            width_size = 32,
            depth=2,
            activation=jax.nn.softplus,
            key=k1
        )

        self.scorer = eqx.nn.MLP(
             in_size = latent_dim,
             out_size ='scalar',
             width_size= 32,
             depth=2,
             activation=jax.nn.softplus,
             key=k2
        )
        

    def __call__(self , ts_robot , u_robot):
        h0 = jnp.zeros((self.latent_dim,))     
        h = self.ode(ts_robot, h0, us=u_robot)  # (50, latent_dim)

        scores = jax.vmap(lambda ht: self.scorer(ht))(h)
        w = jax.nn.softmax(scores)
        h_final = jnp.sum(h * w[:, None], axis=0)
        #h_last = h[-1]
        y_ball = self.nn(h_final)
        return y_ball, w
    

latent_dim = 16
key = jr.key(0)

encoder = NODE(state_size=latent_dim, input_size=12, width_sizes=[64, 64], key=key)
ode = ODESolver(encoder)
model_template = Model(encoder, ode, latent_dim=latent_dim, key=key)

model_loaded = eqx.tree_deserialise_leaves("Models/trained_model_with_time3.eqx", model_template)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_ = klax.finalize(model_loaded)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DT = 0.002
ts = jnp.arange(1500) * DT

q_total_true , q_total_pred , total_true_time , total_pred_time = [] , [] , [] , []
total_idx_true , total_idx_pred = []  , []
N_test = robot_test.shape[0]

for idx in range (2000):
    if idx % 100 == 0:
        print(idx)
    pred , idx_pred = model_(time_test[idx] ,robot_test[idx])
    true = ball_test[idx]
    idx_true = contact_test[idx]
    q_true , true_time = ball_free_flight_trajecotry(true , ts)
    q_pred , pred_time = ball_free_flight_trajecotry(pred , ts)

    q_total_true.append(q_true)
    q_total_pred.append(q_pred)
    total_true_time.append(true_time)
    total_pred_time.append(pred_time)
    total_idx_true.append(idx_true)
    total_idx_pred.append(idx_pred)

q_total_true = jnp.array(q_total_true)
q_total_pred = jnp.array(q_total_pred)
total_true_time = jnp.array(total_true_time)
total_pred_time = jnp.array(total_pred_time)
total_idx_true = jnp.array(total_idx_true)
total_idx_pred = jnp.array(total_idx_pred)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
error = []
error_idx = []
for idx in range(2000):
    time = total_true_time[idx]
    error_x = jnp.abs(q_total_true[idx ,time , 0] - q_total_pred[idx ,time , 0]) 
    error_y = jnp.abs(q_total_true[idx ,time , 1] - q_total_pred[idx ,time , 1]) 
    error_z = jnp.abs(q_total_true[idx ,time , 2] - q_total_pred[idx ,time , 2])

    err1 = jnp.sqrt (error_x**2 + error_y**2 + error_z**2)

    err2 = jnp.argmax(total_idx_true[idx]) - jnp.argmax(total_idx_pred[idx])
    error. append(err1)
    error_idx.append (err2)

error = jnp.array(error)
error_idx = jnp.array(error_idx)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_total = error.shape[0]
num_below = jnp.sum(error <= D)
percentage_below = 100.0 * num_below / num_total

print(f"Threshold R = {D} m")
print(f"Samples below R: {int(num_below)} / {num_total}")
print(f"Percentage below R: {float(percentage_below):.2f}%")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

err_min = error_idx.min()
err_max = error_idx.max()
bins = np.arange(err_min, err_max + 2) - 0.5

plt.figure(figsize=(10, 5))
plt.hist(error_idx, bins=bins , color="tab:blue", edgecolor="black", alpha=0.7, label="error index")
plt.xlabel("error in index")
plt.ylabel("Number of samples")
plt.title("Distribution of error in index of throwing")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

num_total = error_idx.shape[0]
num_below = jnp.sum(jnp.abs(error_idx)<=1)
percentage_below = 100.0 * num_below / num_total
print(f"Samples errors below 2 steps: {int(num_below)} / {num_total}")
print(f"Percentage error below 2 steps: {float(percentage_below):.2f}%")



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

error0 = []
for idx in range(2000):
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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_total = error0.shape[0]
num_below = jnp.sum(error0 <= D)
percentage_below = 100.0 * num_below / num_total

print(f"Threshold D = {D} m")
print(f"Samples below D: {int(num_below)} / {num_total}")
print(f"Percentage below D: {float(percentage_below):.2f}%")



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx = 36
time = total_true_time[idx]
labels = ["x", "y", "z" , "vx" , "vy" , "vz"]

plt.figure(figsize=(12, 6))
for d in range(6):
    plt.subplot(2, 3, d + 1)
    plt.plot(ts[:time], q_total_pred[idx ,:time, d], label="Prediction")
    plt.plot(ts[:time], q_total_true[idx , :time, d], label="True value")
    plt.xlabel("time [s]")          # <-- real time
    plt.ylabel(labels[d])
    plt.title(labels[d])
    plt.grid(alpha=0.3)
    plt.legend()

plt.suptitle(f"Ball state prediction - sample {idx}", fontsize=14)
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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

t_window = jnp.arange(30)*1
plt.figure(figsize=(8, 4))
plt.plot(t_window, total_idx_pred[idx],"o-", label="Predicted")
plt.stem(t_window, total_idx_true[idx], "g-", label="True")
plt.xlabel("index")
plt.ylabel("Probability")
plt.title("Throw index prediction (physical time)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# %%

idx_high = jnp.where(error>0.2)[0]
print(idx_high)
throw_idx_high = jnp.array(total_idx_true[idx_high])
throw_pos = jnp.argmax(throw_idx_high == 1, axis=1)

plt.figure(figsize=(10, 5))
plt.hist(throw_pos , color="tab:blue", edgecolor="black", alpha=0.7, label="index of throw")
plt.xlabel("index of throw")
plt.ylabel("Number of samples")
plt.title("Distribution of index of throw for sample with large position error")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
# %%
