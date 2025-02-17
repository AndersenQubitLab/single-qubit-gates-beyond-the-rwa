from collections.abc import Callable
import numpy as np
from sqgbrwa.pulse_envelopes import *
from sqgbrwa.time_solvers import *


k0 = np.array([1,0]).reshape((2,1))
k1 = np.array([0,1]).reshape((2,1))


def simulate_evolution_vs_carrier_phase(tg: float,
                                        hamiltonian: Callable,
                                        num_phases: int,
                                        num_ppps: int,
                                        num_levels: int | None,
                                        V0: float = 1,
                                        *args,
                                        **kwargs):
    """
    Simulates the unitary time-evolution for a range of carrier phases.
    Goal is to interpolate these time-evolution operators to obtain
    the time-evolution for an arbitrary carrier phase.

    Parameters
    ----------
    tg : float
        Gate duration
    hamiltonian : Callable
        Function that returns the Hamiltonian
    num_phases : int
        Number of carrier phases to simulate the Hamiltonian for
    num_ppps : int
        Number of PPP's to simulate the Hamiltonian for
    num_levels : int | None
        Number of levels in the Hamiltonian
    V0 : float
        Drive strength
    """
    pulse_envelope = CosinePulseEnvelope(tg=tg)
    V0 *= pulse_envelope.V0

    # Get number of levels
    if num_levels is None:
        _H = hamiltonian(tg=tg,
                         *args,
                         **kwargs)
        num_levels = _H.shape[0]

    phases = np.linspace(0, np.pi, num_phases)
    ppps = np.linspace(-1, 1, num_ppps)
    results = np.zeros((num_phases, num_ppps, num_levels, num_levels), dtype=complex)

    for i,phase in enumerate(phases):
        for j,ppp in enumerate(ppps):
            H = hamiltonian(tg=tg,
                            carrier_phase=phase,
                            ppp=ppp,
                            V0=V0,
                            *args,
                            **kwargs)

            results[i,j,:,:] = propagator(H=H, 
                                          t0=tg, 
                                          num_time_steps=10000, 
                                          U0=np.eye(num_levels, dtype=complex),
                                          atol=1e-10, 
                                          rtol=1e-10)
        
    return results
