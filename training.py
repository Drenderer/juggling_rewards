#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import jax
import optax
import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, PyTree
import matplotlib.pyplot as plt
from dynax import ODESolver
import klax
from node import NODE
from helping_function import hitting_ground , find_throwing , ball_free_flight_trajecotry
from normalize import Normalization, coefficients

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = np.load('Data/prepared_data.npz')
robot_t = data['robot_t'][: , :20]
robot_y = data['robot_y'][: , :20]
ball_y = data['ball_y']
index = data['index']

print (robot_t.shape ,robot_y.shape , ball_y.shape , index.shape)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
ball_train  = ball_y[train_idx]
index_train = index[train_idx]

robot_time_test = robot_t[test_idx]
robot_test = robot_y[test_idx]
ball_test  = ball_y[test_idx]
index_test = index[test_idx]

print("Train:", robot_train.shape, ball_train.shape)
print("Test :", robot_test.shape, ball_test.shape)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Model (eqx.Module):
    node: NODE
    ode: ODESolver
    nn:eqx.nn.MLP
    latent_dim: int = eqx.field(static=True)

    def __init__ (self,node , ode , key , latent_dim):
        self.node = node
        self.ode = ode
        self.latent_dim = latent_dim
        self.nn = eqx.nn.MLP(
            in_size = latent_dim,
            out_size= 6,
            width_size = 32,
            depth=2,
            activation=jax.nn.softplus,
            key=key
        )

    def __call__(self , ts_robot , u_robot):
        h0 = jnp.zeros((self.latent_dim,))     
        h = self.ode(ts_robot, h0, us=u_robot)  # (50, latent_dim)
        h_last = h[-1]
        y_ball = self.nn(h_last)
        return y_ball

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def loss_function(model , data, batch_axis):
    robot_ts , robot_batch , ball_batch = data
    pred = jax.vmap(model ,in_axes=(0, 0))(robot_ts , robot_batch)
    loss = jnp.mean(jnp.square(pred - ball_batch))
    return loss   

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
latent_dim =16
key = jr.key(0)
encoder = NODE(state_size=latent_dim , input_size=12, width_sizes=[64,64], key=key)
ode = ODESolver(encoder)
model = Model(encoder, ode , latent_dim=latent_dim , key=key)

time_train = robot_time_train - robot_time_train[:, -1][:, None]  # [ ... , -2DT , -DT , 0.000] for each sample
time_test = robot_time_test - robot_time_test[:, -1][:, None]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model , hist = klax.fit(
    model,
    (time_train , robot_train , ball_train[: , 0 ,:]),
    validation_data=(time_test , robot_test , ball_test[: , 0 ,:]),
    batch_size=64,
    optimizer=optax.adam(3e-4),
    loss_fn=loss_function,
    steps=100000,
    key=jr.key(0)
)

hist.plot()
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_ = klax.finalize(model)
idx = 87
pred = model_(time_test[idx] ,robot_test[idx])
true = ball_test[idx , 0 , :]
print ("true inital value" , true)
print ("pred inital value" , pred)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DT = 0.002
ts = jnp.arange(1500) * DT

ball_q_true , true_time = ball_free_flight_trajecotry(true , ts)
ball_q_pred , pred_time = ball_free_flight_trajecotry(pred , ts)

print ("true hitting time",true_time)
print ("pred hitting time",pred_time)


labels = ["x", "y", "z" , "vx" , "vy" , "vz"]

plt.figure(figsize=(12, 6))
for d in range(6):
    plt.subplot(2, 3, d + 1)
    plt.plot(ts[:true_time], ball_q_pred[:true_time, d], label="Prediction")
    plt.plot(ts[:true_time], ball_q_true[:true_time, d], label="True value")
    plt.xlabel("time [s]")          # <-- real time
    plt.ylabel(labels[d])
    plt.title(labels[d])
    plt.grid(alpha=0.3)
    plt.legend()

plt.suptitle(f"Ball state prediction - sample {idx}", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
error_x = jnp.abs(ball_q_pred[true_time , 0] - ball_q_true[true_time , 0]) 
error_y = jnp.abs(ball_q_pred[true_time , 1] - ball_q_true[true_time , 1]) 
error_z = jnp.abs(ball_q_pred[true_time , 2] - ball_q_true[true_time , 2]) 
print ("error in x:" , error_x)
print ("error in y:" , error_y)
print ("error in z:" , error_z)

error = jnp.sqrt (error_x**2 + error_y**2 + error_z**2)
print ("distance error" , error)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
np.savez('Data/train_idx.npz' , train_idx = train_idx)
np.savez('Data/test_idx.npz' , test_idx = test_idx)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eqx.tree_serialise_leaves("Models/trained_model_norm.eqx", model_)

# %%
