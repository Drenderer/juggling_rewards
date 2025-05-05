
import jax.random as jr
import jax.numpy as jnp

from jaxtyping import PRNGKeyArray, Array
from typing import Optional

def generate_aprbs(key: PRNGKeyArray, length: int, num_jumps: int, initial_value: Optional[float]=None) -> Array:
    """Generate an amplitude-modulated pseudo-random binary sequence (APRBS). The output sequence 
    contains numbers from [0, 1).

    Args:
        key: JAX PRNGKey.
        length: Number of samples in the sequence.
        num_jumps: Number of jumps in the sequence.
        initial_value: The inital value of the sequence. If None, then it is chosen randomly.

    Raises:
        ValueError: If the number of jumps exceeds the number of possible jumping points (length-2).

    Returns:
        Array with shape=(length,) describing the APRBS.
    """

    keys = jr.split(key, 3)
    ts = jnp.arange(length)

    if num_jumps > length - 2:
        raise ValueError('Number of jumps must be smaller than the length-2!')

    indices = jnp.sort(jr.choice(keys[1], ts[1:-1], shape=(num_jumps,), replace=False))
    if initial_value is None:
        values = jr.uniform(keys[2], shape=(num_jumps+1,))
    else:
        values = jr.uniform(keys[2], shape=(num_jumps,))
        values = jnp.concatenate([jnp.array([initial_value]), values])

    counts = jnp.diff(jnp.concatenate([jnp.array([0]), indices, jnp.array([length])]))

    return jnp.repeat(values, counts, total_repeat_length=length)
