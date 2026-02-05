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
data = np.load('Data/prepared_data_with_time.npz')
robot_t = data['robot_t']
robot_y = data['robot_y']
contact = data['contact']
ball_y = data['ball_y']
index = data['index']

print (robot_t.shape ,robot_y.shape , contact.shape, ball_y.shape , index.shape)

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
contact_train = contact[train_idx]
index_train = index[train_idx]

robot_time_test = robot_t[test_idx]
robot_test = robot_y[test_idx]
contact_test = contact[test_idx]
ball_test  = ball_y[test_idx]
index_test = index[test_idx]

print("Train:", robot_train.shape, ball_train.shape , contact_train.shape)
print("Test :", robot_test.shape, ball_test.shape , contact_test.shape)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
        h = self.ode(ts_robot, h0, us=u_robot)  # (window_time, latent_dim)

        
        scores = jnp.ravel(jax.vmap(self.scorer)(h))
        w = jax.nn.softmax(scores)
        h_final = jnp.sum(h * w[:, None], axis=0)
        y_ball = self.nn(h_final)
        return y_ball, w

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def loss_function(model , data, batch_axis):
    robot_ts , robot_batch , contact_batch , ball_batch = data
    pred , w_pred = jax.vmap(model ,in_axes=(0, 0))(robot_ts , robot_batch)
    idx = jnp.argmax(contact_batch, axis=1)                 # (B,)
    b = jnp.arange(w_pred.shape[0])
    eps = 1e-8
    loss_time = -jnp.mean(jnp.log(w_pred[b, idx] + eps))
    loss_position = jnp.mean(jnp.square(pred - ball_batch))
    #jax.debug.print("loss_position={lp:.6f} loss_time={lt:.6f}", lp=loss_position, lt=loss_time)
    landa = 0.1
    return (landa*loss_time + loss_position)   

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
latent_dim =16
key = jr.key(0)
encoder = NODE(state_size=latent_dim , input_size=12, width_sizes=[64,64], key=key)
ode = ODESolver(encoder)
model = Model(encoder, ode, latent_dim=latent_dim , key=key)


time_train = robot_time_train - robot_time_train[:, -1][:, None]  # [ ... , -2DT , -DT , 0.000] for each sample
time_test = robot_time_test - robot_time_test[:, -1][:, None]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model , hist = klax.fit(
    model,
    (time_train , robot_train , contact_train, ball_train),
    validation_data=(time_test , robot_test , contact_test, ball_test),
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
idx = 5000
pred_position , pred_idx = model_(time_test[idx] ,robot_test[idx])
true_position = ball_test[idx]
true_idx = contact_test[idx]
print ("true inital value" , true_position)
print ("pred inital value" , pred_position)

t_window = jnp.arange(30)*1
plt.figure(figsize=(8, 4))
plt.plot(t_window, pred_idx,"o-", label="Predicted")
plt.stem(t_window, true_idx, "g-", label="True")
plt.xlabel("Time [s]")
plt.ylabel("Probability")
plt.title("Throw time prediction (physical time)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DT = 0.002
ts = jnp.arange(1500) * DT

ball_q_true , true_time = ball_free_flight_trajecotry( true_position , ts)
ball_q_pred , pred_time = ball_free_flight_trajecotry(pred_position , ts)

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
eqx.tree_serialise_leaves("Models/trained_model_with_time3.eqx", model_)

# %%
