import numpy as np
from sqgbrwa.time_solvers import *
from sqgbrwa.pulse_envelopes import *
from sqgbrwa.utils import *


"""
This file contains functions that calculate time evolution operators
for the 0th order and 1st order Magnus approximation
"""

def magnus_expansion_0(tg: float,
                       wd: float,
                       ppp: float,
                       t: np.ndarray | float,
                       V0: float | None = None,
                       detuning: float = 0,
                       phi: float = 0,
                       return_mode: str = 'dm'):
    """
    Calculates the dm's or unitaries for the 0th order Magnus term.
    `term_x` and `term_y` are computed using Mathematica.

    Parameters
    ----------
    tg : float
        Gate duration
    wd : float
        Drive frequency
    ppp : float
        Magnitude of out-of-phase drive component relative to the
        in-phase drive component
    t : float | np.ndarray
        Time or array of times to compute the time-evolution/density matrix at
    V0 : float | None
        Absolute magnitude of the in-phase component of the drive. If `None`, 
        it is set such that the integral of the pulse envelope is Pi.
    phi : float
        Phase of the carrier signal
    return_mode : str
        If set to `dm`, a density matrix is returned. Otherwise, the 
        unitary time-evolution operator is returned.
    """
    # Calculate V0 if not provided
    pulse_envelope = CosinePulseEnvelope(tg=tg)
    if V0 is None:
        V0 = pulse_envelope.V0

    # Calculate term ~sigma_x
    if np.isclose(tg, np.pi/wd, atol=1e-12):
        # If tg=tc, there is a divergence in the "normal" integral,
        # so we have to compute it separately
        term_x = lambda _t : 1/32/wd * (
            8 * V0 * _t * wd -
            4 * V0 * (1 - 2*ppp) * _t * wd * np.cos(2*phi) + 
            8 * V0 * np.cos(phi + 2*_t*wd) * np.sin(phi) + 
            V0 * (-3 + 2*ppp) * np.sin(2*phi) -
            V0 * (1 + 2*ppp) * np.sin(2*(phi + 2*_t*wd))
        )

        # Calculate term ~sigma_y
        term_y = lambda _t : 1/32/wd * (
            8 * V0 * ppp +
            V0 * (3 - 2*ppp) * np.cos(2*phi) -
            8 * V0 * ppp * np.cos(2*_t*wd) -
            4 * V0 * np.cos(2*(phi+_t*wd)) + 
            V0 * np.cos(2*(phi + 2*_t*wd)) +
            2 * V0 * ppp * np.cos(2*(phi+2*_t*wd)) -
            4 * V0 * (1 - 2*ppp) * _t * wd * np.sin(2*phi)
        )
    else:
        term_x = lambda _t : (
            1 / (16 * np.pi * wd * (np.pi - tg * wd) * (np.pi + tg * wd)) * (
                2 * V0 * tg * wd * (-np.pi**2 + tg**2 * wd**2) * np.sin((2 * np.pi * _t) / tg) +
                np.pi * (
                    -2 * np.pi * (V0 * np.pi - 2 * (V0*ppp) * tg * wd) * np.sin(2 * phi) +
                    2 * V0 * (np.pi**2 - tg**2 * wd**2) * np.sin(2 * (phi + _t * wd)) +
                    (V0 + 2 * (V0*ppp)) * tg * wd * (-np.pi + tg * wd) * np.sin(2 * ((np.pi * _t) / tg + phi + _t * wd)) +
                    wd * (np.pi + tg * wd) * (4 * V0 * _t * (np.pi - tg * wd) - (V0 - 2 * (V0*ppp)) * tg * np.sin((2 * np.pi * _t) / tg - 2 * (phi + _t * wd)))
                )
            )
        )

        # Calculate term ~sigma_y
        term_y = lambda _t : (
            1/8 * (
                (V0*ppp) * tg * (
                    2/np.pi - (2 * np.cos((2 * np.pi * _t) / tg)) / np.pi - 
                    (2 * np.pi * np.cos(2 * phi)) / ((np.pi - tg * wd) * (np.pi + tg * wd)) + 
                    np.cos(2 * (phi + _t * (-(np.pi / tg) + wd))) / (np.pi - tg * wd) +
                    np.cos(2 * (phi + _t * (np.pi / tg + wd))) / (np.pi + tg * wd)
                ) + 
                V0 * (
                    -np.pi**2 * np.cos(2 * phi) + 
                    (np.pi**2 - tg**2 * wd**2 + tg**2 * wd**2 * np.cos((2 * np.pi * _t) / tg)) * 
                    np.cos(2 * (phi + _t * wd)) + 
                    np.pi * tg * wd * np.sin((2 * np.pi * _t) / tg) * np.sin(2 * (phi + _t * wd))
                ) / (wd * (-np.pi + tg * wd) * (np.pi + tg * wd))
            ) 
        )


    if isinstance(t, NumberTypes):
        # Calculate Hamiltonian and unitary
        H = Hamiltonian(Rx=term_x(t),
                        Ry=term_y(t),
                        Rz=-t*detuning/2)
        U = H.unitary()

        if return_mode=='dm':
            # Calculate and return dm
            return U @ np.diag([1,0]) @ U.conj().T
        else:
            return U
    else:
        results = np.zeros((len(t),2,2), dtype=complex)

        for i,time_step in enumerate(t):
            # Calculate Hamiltonian and unitary
            H = Hamiltonian(Rx=term_x(time_step),
                            Ry=term_y(time_step),
                            Rz=-time_step*detuning/2)
            U = H.unitary()

            if return_mode=='dm':
                # Calculate and store dm
                results[i,:,:] = U @ np.diag([1,0]) @ U.conj().T
            else:
                # Store unitary
                results[i,:,:] = U

        return results


def magnus0_fidelity(tg: float, 
                     wd: float, 
                     ppp: float,
                     phi: float = 0,
                     t: float = None,
                     V0: float = 1,):
    """
    Computes the fidelity between the unitary evolutions of the
    RWA time-evolution and the 0th-order Magnus expansion of the
    non-RWA time-evolution.

    Parameters
    ----------
    tg : float
        Gate duration
    wd : float
        Drive frequency
    ppp : float
        Relative amplitude of out-of-phase and in-phase drive components.
    phi : float
        Phase of carrier signal
    t : float
        Time to compute Fidelity at
    """
    # Define pulse envelope
    pulse_envelope = CosinePulseEnvelope(tg=tg)

    # Calculate V0
    _V0 = pulse_envelope.V0

    # Hamiltonian for RWA evolution
    def H_rwa(t):
        return 1/2*_V0*pulse_envelope.envelope(t)*sigmax_np

    # Compute time-evolution of RWA Hamiltonian using an
    # ODE Solver. Note that `U_rwa` will represent a perfect
    # Xpi gate (up to numerical imprecisions)
    U_rwa = propagator(H_rwa, t0=tg, num_time_steps=1000)

    # Get time-evolution operator of 0th-order Magnus
    # expansion of non-RWA Hamiltonian
    U_nonrwa = magnus_expansion_0(tg=tg,
                                  wd=wd,
                                  ppp=ppp,
                                  t=tg,
                                  V0=V0*_V0,
                                  phi=phi,
                                  return_mode="unitary")

    # Calculate fidelity
    E = U_rwa.conj().T @ U_nonrwa
    F = (2+np.trace(E)*np.trace(E.conj().T)).real/6

    return F
    

def magnus_expansion_1_tg(tg: float,
                          wd: float,
                          ppp: float,
                          V0: float | None = None,
                          detuning: float = 0,
                          phi: float = 0,
                          return_mode: str = 'dm'):
    """
    Calculates the dm's or unitaries for the 1st order Magnus term.
    `term_x` and `term_y` are computed using Mathematica. It is only 
    possible to compute the 1st order Magnus term over the full 
    gate duration. It is possible, of course, to do it at an arbitary
    time `t`, but those expressions are significantly more involved.

    Parameters
    ----------
    tg : float
        Gate duration
    wd : float
        Drive frequency
    ppp : float
        Magnitude of out-of-phase drive component relative to the
        in-phase drive component
    t : float | np.ndarray
        Time or array of times to compute the time-evolution/density matrix at
    V0 : float | None
        Absolute magnitude of the in-phase component of the drive. If `None`, 
        it is set such that the integral of the pulse envelope is Pi.
    detuning : float
        Detuning of the drive frequency
    phi : float
        Phase of the carrier signal
    return_mode : str
        If set to `dm`, a density matrix is returned. Otherwise, the 
        unitary time-evolution operator is returned.
    """
    # Calculate V0 if not provided
    if V0 is None:
        pulse_envelope = CosinePulseEnvelope(tg=tg)
        V0 = pulse_envelope.V0

    term_x1 = 1/4 * (
        V0 * tg + (np.pi * V0 * (np.pi - 2 * ppp * tg * wd) * 
        np.cos(2 * phi + tg * wd) * np.sin(tg * wd)) / (wd * (np.pi - tg * wd) * (np.pi + tg * wd))
    )

    term_y1 = (
        np.pi * (-V0 * np.pi + 2 * (V0*ppp) * tg * wd) * np.sin(tg * wd) * np.sin(2 * phi + tg * wd)
    ) / (-4 * np.pi**2 * wd + 4 * tg**2 * wd**3)

    term_x2 = - (1/(8 * np.pi * wd**2 * (np.pi - tg * wd)**2 * (np.pi + tg * wd)**2)) * (
        4 * (V0*ppp) * np.pi**4 * tg**2 * wd**2 - 
        8 * (V0*ppp) * np.pi**2 * tg**4 * wd**4 + 
        4 * (V0*ppp) * tg**6 * wd**6 + 
        np.pi**2 * tg * wd * (V0 * np.pi - 2 * (V0*ppp) * tg * wd) * (np.pi**2 - tg**2 * wd**2) * np.cos(2 * phi) + 
        np.pi**2 * tg * wd * (V0 * np.pi - 2 * (V0*ppp) * tg * wd) * (np.pi**2 - tg**2 * wd**2) * np.cos(2 * (phi + tg * wd)) + 
        V0 * np.pi**5 * np.sin(2 * phi) - 
        3 * V0 * np.pi**3 * tg**2 * wd**2 * np.sin(2 * phi) + 
        4 * (V0*ppp) * np.pi**2 * tg**3 * wd**3 * np.sin(2 * phi) - 
        V0 * np.pi**5 * np.sin(2 * (phi + tg * wd)) + 
        3 * V0 * np.pi**3 * tg**2 * wd**2 * np.sin(2 * (phi + tg * wd)) - 
        4 * (V0*ppp) * np.pi**2 * tg**3 * wd**3 * np.sin(2 * (phi + tg * wd))
   )

    term_y2 = - (
        np.pi * (tg * wd * (V0 * np.pi - 2 * (V0*ppp) * tg * wd) * (np.pi**2 - tg**2 * wd**2) * np.cos(tg * wd) - 
        (4 * (V0*ppp) * tg**3 * wd**3 + V0 * (np.pi**3 - 3 * np.pi * tg**2 * wd**2)) * np.sin(tg * wd)) * np.sin(2 * phi + tg * wd)
    ) / (4 * wd**2 * (np.pi - tg * wd)**2 * (np.pi + tg * wd)**2)

    term_z = (1/(64 * wd**2 * (np.pi - tg * wd)**2 * (np.pi + tg * wd)**2 * (4 * np.pi**3 - np.pi * tg**2 * wd**2))) * (
        -8 * V0**2 * np.pi**7 * tg * wd + 
        32 * V0 * (V0*ppp) * np.pi**6 * tg**2 * wd**2 + 
        22 * V0**2 * np.pi**5 * tg**3 * wd**3 + 
        16 * (V0*ppp)**2 * np.pi**5 * tg**3 * wd**3 - 
        88 * V0 * (V0*ppp) * np.pi**4 * tg**4 * wd**4 - 
        17 * V0**2 * np.pi**3 * tg**5 * wd**5 - 
        20 * (V0*ppp)**2 * np.pi**3 * tg**5 * wd**5 + 
        68 * V0 * (V0*ppp) * np.pi**2 * tg**6 * wd**6 + 
        3 * V0**2 * np.pi * tg**7 * wd**7 + 4 * (V0*ppp)**2 * np.pi * tg**7 * wd**7 - 
        12 * V0 * (V0*ppp) * tg**8 * wd**8 + 
        2 * V0 * np.pi**2 * tg * wd * (V0 * np.pi - 2 * (V0*ppp) * tg * wd) * (
            4 * np.pi**4 - 
            5 * np.pi**2 * tg**2 * wd**2 + tg**4 * wd**4
        ) * np.cos(2 * phi) + 
        2 * V0 * np.pi**2 * tg * wd * (V0 * np.pi - 2 * (V0*ppp) * tg * wd) * (4 * np.pi**4 - 5 * np.pi**2 * tg**2 * wd**2 + tg**4 * wd**4) * np.cos(2 * (phi + tg * wd)) + 
        8 * V0**2 * np.pi**7 * np.sin(2 * phi) - 
        16 * V0 * (V0*ppp) * np.pi**6 * tg * wd * np.sin(2 * phi) - 
        20 * V0**2 * np.pi**5 * tg**2 * wd**2 * np.sin(2 * phi) + 
        24 * (V0*ppp)**2 * np.pi**5 * tg**2 * wd**2 * np.sin(2 * phi) + 
        40 * V0 * (V0*ppp) * np.pi**4 * tg**3 * wd**3 * np.sin(2 * phi) - 
        24 * (V0*ppp)**2 * np.pi**3 * tg**4 * wd**4 * np.sin(2 * phi) + 
        4 * V0**2 * np.pi**7 * np.sin(2 * tg * wd) - 
        16 * V0 * (V0*ppp) * np.pi**6 * tg * wd * np.sin(2 * tg * wd) - 
        V0**2 * np.pi**5 * tg**2 * wd**2 * np.sin(2 * tg * wd) + 
        16 * (V0*ppp)**2 * np.pi**5 * tg**2 * wd**2 * np.sin(2 * tg * wd) + 
        4 * V0 * (V0*ppp) * np.pi**4 * tg**3 * wd**3 * np.sin(2 * tg * wd) - 
        4 * (V0*ppp)**2 * np.pi**3 * tg**4 * wd**4 * np.sin(2 * tg * wd) - 
        8 * V0**2 * np.pi**7 * np.sin(2 * (phi + tg * wd)) + 
        16 * V0 * (V0*ppp) * np.pi**6 * tg * wd * np.sin(2 * (phi + tg * wd)) + 
        20 * V0**2 * np.pi**5 * tg**2 * wd**2 * np.sin(2 * (phi + tg * wd)) - 
        24 * (V0*ppp)**2 * np.pi**5 * tg**2 * wd**2 * np.sin(2 * (phi + tg * wd)) - 
        40 * V0 * (V0*ppp) * np.pi**4 * tg**3 * wd**3 * np.sin(2 * (phi + tg * wd)) + 
        24 * (V0*ppp)**2 * np.pi**3 * tg**4 * wd**4 * np.sin(2 * (phi + tg * wd))
    )

    H = Hamiltonian(Rx=term_x1 - 1j/2 * (-1j*detuning*term_x2),
                    Ry=term_y1 - 1j/2 * (-1j*detuning*term_y2),
                    Rz=-tg*detuning/2 - 1j/2 * (2*1j*term_z))
    U = H.unitary()

    if return_mode=='dm':
        # Calculate and return dm
        return U @ np.diag([1,0]) @ U.conj().T
    else:
        return U


def optimize_1st_order(tg: float,
                       w01: float,
                       phis: list[float] | np.ndarray = [0],
                       t: float | None = None,
                       V0: float | None = None,
                       V0_bound: float = 0.1,
                       ppp_bound: float = 1,
                       fd_bound: float = 5,
                       U_ideal: np.ndarray = sigmax_np,
                       tol=1e-8,
                       x0s: list[list] | None = None):
    """
    Optimizes the drive strength V0 for the 1st order Magnus
    approximation.
    """
    if V0 is None:
        pulse_envelope = CosinePulseEnvelope(tg=tg)
        V0 = pulse_envelope.V0

    if t is None:
        t = tg

    # Function passed to optimizer, essentially a wrapper
    # around `magnus_expansion_1`
    def _wrapper(x):
        cost = 0
        for phi in phis:
            U = magnus_expansion_1_tg(tg=tg,
                                      wd=w01-x[1]*1e6*2*np.pi,
                                      ppp=x[2],
                                      V0=x[0]*V0,
                                      detuning=x[1]*1e6*2*np.pi,
                                      phi=phi,
                                      return_mode="unitary")

            # Calculate 1-Fidelity
            E = U_ideal.conj().T @ U
            cost += 2 * (1 - (2+np.trace(E)*np.trace(E.conj().T)).real/6)

        return cost

    # Specify initial guess and bounds
    if x0s is None:
        x0s = []
        for x in [1]:
            for y in [0]:
                for z in [1/2/w01*np.pi/tg]:
                    x0s.append([x,y,z])

    delta_max = min([w01*1e-6/2/np.pi, fd_bound])
    bounds = ((1-V0_bound, 1+V0_bound),
              (-fd_bound, delta_max),
              (-ppp_bound, ppp_bound))

    # Optimize
    res = minimize_global(_wrapper, 
                          x0s, 
                          bounds=bounds,
                          method="Nelder-Mead",
                          tol=tol)
    
    fidelities = np.zeros(len(phis))
    for i,phi in enumerate(phis):
        U = magnus_expansion_1_tg(tg=tg,
                                  wd=w01-res["x"][1]*1e6*2*np.pi,
                                  ppp=res["x"][2],
                                  V0=res["x"][0]*V0,
                                  detuning=res["x"][1]*1e6*2*np.pi,
                                  phi=phi,
                                  return_mode="unitary")

        E = U_ideal.conj().T @ U
        fidelities[i] = (2+np.trace(E)*np.trace(E.conj().T)).real/6

    return res, fidelities
