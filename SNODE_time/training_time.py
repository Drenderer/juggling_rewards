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
import sys
from pathlib import Path
sys.path.append("..") 
from node import NODE
from helping_function import hitting_ground , find_throwing , ball_free_flight_trajecotry
from normalize import Normalization, coefficients

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = np.load('prepared_samples/train_data.npz')
robot_time_train  = data['robot_t']
robot_train = data['robot_y']
contact_train = data['contact']
ball_train= data['ball_y']
index_train = data['index']

data = np.load('prepared_samples/test_data.npz')
robot_time_test  = data['robot_t']
robot_test = data['robot_y']
contact_test = data['contact']
ball_test= data['ball_y']
index_test = data['index']

print (robot_time_train.shape ,robot_train.shape , ball_train.shape , contact_train.shape , index_train.shape)




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Model (eqx.Module):
    node: NODE
    ode: ODESolver
    scorer : eqx.nn.MLP
    scorer_none: eqx.nn.MLP
    latent_dim: int = eqx.field(static=True)

    def __init__ (self, node  , ode , key , latent_dim):
        k1, k2 = jr.split(key, 2)
        self.node = node
        self.ode = ode
        self.latent_dim = latent_dim
        
        self.scorer = eqx.nn.MLP(
             in_size = latent_dim,
             out_size ='scalar',
             width_size= 32,
             depth=2,
             activation=jax.nn.softplus,
             key=k1
        )

        self.scorer_none = eqx.nn.MLP(
            in_size=latent_dim,
            out_size="scalar",
            width_size=32,
            depth=2,
            activation=jax.nn.softplus,
            key=k2,
        )


    def __call__(self , ts_robot , u_robot):
        h0 = jnp.zeros((self.latent_dim,))     
        h = self.ode(ts_robot, h0, us=u_robot)  # (window_time, latent_dim)

        time_scores = jnp.ravel(jax.vmap(self.scorer)(h))  #(window_time ,)
        h_pool = jnp.mean(h , axis=0)
        logit_scores = jnp.asarray(self.scorer_none(h_pool))   #scaler
        logit = jnp.concatenate([time_scores , logit_scores[None]] , axis = 0)  #(window +1 ,)
        w = jax.nn.softmax(logit)
        
        return w


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def loss_function(model , data, batch_axis):
    robot_ts , robot_batch , contact_batch = data
    w_pred = jax.vmap(model , in_axes=(0,0))(robot_ts , robot_batch)
    w_time = w_pred[:, :-1]
    w_none = w_pred[:, -1]

    idx = jnp.argmax(contact_batch, axis=1)                 # (B,)
    b = jnp.arange(w_time.shape[0])
    y_throw = (jnp.sum(contact_batch, axis=1) > 0).astype(jnp.float32)  #(B,)

    eps = 1e-8
    loss_time = -jnp.log(w_time[b, idx] + eps)   # (B,)
    loss_none  = -jnp.log(w_none + eps)          # (B,)
    loss = jnp.mean(y_throw * loss_time + (1.0 - y_throw) * loss_none)

    return loss


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
latent_dim =16
key = jr.key(0)
encoder = NODE(state_size=latent_dim , input_size=8, width_sizes=[64,64], key=key)
ode = ODESolver(encoder)
model = Model(encoder, ode, latent_dim=latent_dim , key=key)


time_train = robot_time_train - robot_time_train[:, -1][:, None]  # [ ... , -2DT , -DT , 0.000] for each sample
time_test = robot_time_test - robot_time_test[:, -1][:, None]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model , hist = klax.fit(
    model,
    (time_train , robot_train, contact_train),
    validation_data=(time_test , robot_test, contact_test),
    batch_size=64,
    optimizer=optax.adam(5e-4),
    loss_fn=loss_function,
    steps=10000,
    key=jr.key(0)
)

hist.plot()
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_ = klax.finalize(model)
idx = 181
pred_w = model_(time_test[idx] ,robot_test[idx])
true_w = contact_test[idx]

t_window = jnp.arange(30)*1
plt.figure(figsize=(8, 4))
plt.plot(t_window, pred_w,"o-", label="Predicted")
plt.stem(t_window, true_w, "g-", label="True")
plt.xlabel("Time [s]")
plt.ylabel("Probability")
plt.title("Throw time prediction (physical time)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print(pred_w[-1])
print (np.sum(pred_w))


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eqx.tree_serialise_leaves("trained_model_time.eqx", model_)

# %%
