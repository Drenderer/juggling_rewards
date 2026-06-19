from collections.abc import Callable

import equinox as eqx
import jax
from jaxtyping import Array, Float, Scalar

class contact_ISPHS (eqx.Module):

    hamiltonian: Callable[[Array], Scalar]
    structure_matrix: Callable[[Float[Array, "n"]], Float[Array, "n n"]]  
    dissipation_matrix: (
        Callable[[Float[Array, "n"]], Float[Array, "n n"]] | None
    )  
    input_matrix: Callable[[Float[Array, "n"]], Float[Array, "n m"]] | None  
    contact: Callable[[Array, Array], Scalar]

    def __init__ (
            self,
        hamiltonian: Callable[[Array], Scalar],
        structure_matrix: Callable[[Float[Array, "n"]], Float[Array, "n n"]],  # noqa: F722, F821
        dissipation_matrix: Callable[[Float[Array, "n"]], Float[Array, "n n"]]  # noqa: F722, F821
        | None = None,
        input_matrix: Callable[[Float[Array, "n"]], Float[Array, "n m"]]
        | None = None,  # noqa: F722, F821
        contact: Callable[[Array, Array], Scalar] | None = None
    ):
        
        self.hamiltonian = hamiltonian
        self.structure_matrix = structure_matrix
        self.dissipation_matrix = dissipation_matrix
        self.input_matrix = input_matrix
        self.contact = contact

    def __call__(self, t: Scalar, x: Array, u: Array | None = None) -> Array:
        
        if self.contact is None:
                c = 0.0
        else:
                c = self.contact(x, u)     # scalar in [0, 1]
        c= 0.0
        structure_matrix = self.structure_matrix

        if self.dissipation_matrix is not None:
            dissipation_matrix = self.dissipation_matrix(x)
            structure_matrix = structure_matrix - c * dissipation_matrix

        x_t = structure_matrix @ jax.grad(self.hamiltonian)(x)
        

        if self.input_matrix is not None:
            if u is None:
                raise ValueError(
                    "The ISPHS has an input matrix but no input u was provided."
                )

            input_matrix = self.input_matrix(x , u)

            #x_t = x_t + c * input_matrix
            #x_t = x_t.at[3:6].add(c*input_matrix)

        return x_t


