from collections.abc import Callable
from typing import Union
import odeintw
import numpy as np


def propagator(H: Callable,
               t0: float,
               num_time_steps: int = 1000,
               U0: Union[np.ndarray, None] = None,
               return_all: bool = False,
               rtol: float = 1e-13,
               atol: float = 1e-13):
    """
    Calculates the time-dependent propagator U(t) at t=t0 in the Heisenberg picture

    Parameters
    ----------
    H : function
        Function that returns the Hamiltonian at a time step `t`. 
    t0 : float
        Time to compute U on
    num_time_steps : int
        Number of timesteps for the solver
    U0 : np.ndarray
        Initial propagator
    """
    if U0 is None:
        U0 = np.eye(H(0).shape[0], dtype=complex)

    # Function for odeint solver
    def _propagate(U, t):
        _U = U.reshape(U0.shape)
        dU_dt = -1j * H(t) @ _U
        return dU_dt.flatten()
    
    # Time to solve!
    t = np.linspace(0, t0, num_time_steps)
    z = odeintw.odeintw(_propagate, U0.flatten(), t, h0=t0/num_time_steps, rtol=rtol, atol=atol)

    if return_all:
        return z.reshape((num_time_steps,U0.shape[0],U0.shape[1]))
    else:
        return z[-1,:].reshape(U0.shape)
    