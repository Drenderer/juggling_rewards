

from jax import numpy as jnp
from jaxtyping import Array, PyTree
import equinox as eqx
from dynax import normalization_coefficients



def coefficients (mean_q: Array, std_q: Array, std_u: Array,
                     std_v: Array, std_a: Array, tol: float = 1e-6):

    alpha_q, tau_q = normalization_coefficients(std_q, std_v, std_a, tol=1e-6)
    alpha_u, _ = normalization_coefficients(std_u, tol=1e-6)

    return alpha_q, tau_q , alpha_u


class Normalization(eqx.Module):
    mean_q: Array
    alpha_q: Array
    tau_q: Array
    mean_u: Array
    alpha_u: Array

    def transform_ts(self, ts: Array):
        return ts / self.tau_q

    def inverse_transform_ts(self, ts: Array):
        return ts * self.tau_q

    def transform_qs(self, qs: Array):
        return self.alpha_q * (qs - self.mean_q)

    def inverse_transform_qs(self, qs: Array):
        return qs / self.alpha_q + self.mean_q

    def transform_q_ts(self, q_ts: Array):
        return self.tau_q * self.alpha_q * q_ts

    def inverse_transform_q_ts(self, q_ts: Array):
        return q_ts / (self.tau_q * self.alpha_q)

    def transform_q_tts(self, q_tts: Array):
        return self.tau_q**2 * self.alpha_q * q_tts

    def inverse_transform_q_tts(self, q_tts: Array):
        return q_tts / (self.tau_q**2 * self.alpha_q)

    def transform_taus(self, taus: Array):
        return self.alpha_u * (taus - self.mean_u)

    def inverse_transform_taus(self, taus: Array):
        return taus / self.alpha_u + self.mean_u
