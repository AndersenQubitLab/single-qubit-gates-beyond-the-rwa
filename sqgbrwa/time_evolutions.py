from collections.abc import Callable
import numpy as np
from sqgbrwa.time_solvers import *
from sqgbrwa.pulse_envelopes import *
from sqgbrwa.utils import *


gates_ideal = {
    "rx_pi": rx(np.pi),
    "ry_pi": ry(np.pi),
    "rx_pi2": rx(np.pi/2),
    "ry_pi2": ry(np.pi/2),
    "rx_mpi": rx(-np.pi),
    "rx_mpi2": rx(-np.pi/2),
    "ry_mpi": ry(-np.pi),
    "ry_mpi2": ry(-np.pi/2),
}


def mp_wrapper(fun,
               *args,
               **kwargs):
    return fun(*args, **kwargs)


def H_full_tls(tg: float,
               w01: float,
               wd: float,
               ppp: float,
               pulse_envelope: PulseEnvelope | None = None,
               V0: float | None = None,
               carrier_phase: float = 0):
    """
    Returns the Hamiltonian for a two-level system
    NOTE that this function does not support
    different envelope phases
    """
    if pulse_envelope is None:
        pulse_envelope = CosinePulseEnvelope(tg=tg)
    if V0 is None:
        V0 = pulse_envelope.V0

    delta = w01-wd
    H0 = -delta/2 *sigmaz_np

    def H(t):
        _cos = np.cos(2*wd*t + 2*carrier_phase)
        _sin = np.sin(2*wd*t + 2*carrier_phase)
        _envelope = pulse_envelope.envelope(t)
        _deriv    = pulse_envelope.derivative(t)
        HDrive = V0/2 * (
            sigmax_np * (
                (1 + _cos) * _envelope +
                ppp * _sin * _deriv
            ) + sigmay_np * (
                ppp * (1 - _cos) * _deriv +
                _sin * _envelope
            )
        )

        return H0 + HDrive

    return H


def H_full_4_levels(tg: float,
                    wd: float,
                    w01: float,
                    alpha2: float,
                    alpha3: float,
                    mat_elems: np.ndarray,
                    omega_delta: float = 0,
                    ppp: float = 0,
                    V0: None | float = None,
                    carrier_phase: float = 0,
                    envelope_phase: float = 0,
                    pulse_envelope: PulseEnvelope | None = None):
    """
    Returns the Hamiltonian for a 4 level system
    """
    # Define pulse envelope and drive strength
    if pulse_envelope is None:
        pulse_envelope = CosinePulseEnvelope(tg=tg)
    if V0 is None:
        V0 = pulse_envelope.V0

    # Delta(t) in lab frame
    def delta(t):
        wds = omega_delta * V0**2 / t * (
            pulse_envelope.envelope_square_int(t) +
            ppp**2 * pulse_envelope.derivative_square_int(t)
        )
        return np.nan_to_num(wds, nan=0)
    
    # Delta'(t) in rotating frame
    def delta_prime(t):
        wds = omega_delta * V0**2 * (
            pulse_envelope.envelope(t) +
            ppp**2 * pulse_envelope.derivative(t)
        )
        return np.nan_to_num(wds, nan=0)
    
    # Define pulse shape
    amp = V0*np.exp(1j*envelope_phase)
    wave = lambda t: amp * (
        pulse_envelope.envelope(t) + 
        1j * ppp * pulse_envelope.derivative(t) * wd/(wd - delta(t))
    )

    # Construct drive operators 
    drive_op_x = mat_elems * np.array([[0,1,0,0],
                                       [1,0,1,0],
                                       [0,1,0,1],
                                       [0,0,1,0]])
    drive_op_y = mat_elems * np.array([[0,-1j,0,0],
                                       [1j,0,-1j,0],
                                       [0,1j,0,-1j],
                                       [0,0,1j,0]])
    drive_op_x03 = mat_elems * np.array([[0,0,0,1],
                                         [0,0,0,0],
                                         [0,0,0,0],
                                         [1,0,0,0]])
    drive_op_y03 = mat_elems * np.array([[0,0,0,-1j],
                                         [0,0,0,0],
                                         [0,0,0,0],
                                         [1j,0,0,0]])

    def H(t):
        # Build Hamiltonian
        detuning = (w01-wd) + delta(t)
        detuning_prime = (w01-wd) + delta_prime(t)
        
        # H0
        H0 = np.diag([0, detuning_prime, 2*(detuning_prime)+alpha2, 3*(detuning_prime)+alpha3])

        # Construct drive terms
        _cos = np.cos(2*(w01-detuning)*t+2*carrier_phase)
        _sin = np.sin(2*(w01-detuning)*t+2*carrier_phase)
        w = wave(t)
        wx = 1/2 * (w.real*(1+_cos) + w.imag*_sin)
        wy = 1/2 * (w.imag*(1-_cos) + w.real*_sin)

        # Construct drive operator
        wxx = wx*_cos - wy*_sin
        wyy = wy*_cos + wx*_sin

        return (
            H0 +
            wx  * drive_op_x +
            wy  * drive_op_y +
            wxx * drive_op_x03 + 
            wyy * drive_op_y03
        )
    
    return H


def ideal_pulse_parameters_propagator_carrier_phase(tg: float,
                                                    w01: float = 2*np.pi*6e9,
                                                    gate: str = "rx_pi",
                                                    V0_bound: float = 0.1,
                                                    ppp_bound: float = 1,
                                                    fd_bound: float = 5,
                                                    carrier_phases: list[float] | np.ndarray = [0],
                                                    x0s: list | None = None,
                                                    num_time_steps: float = 10000,
                                                    tol: float = 1e-8,
                                                    propagator_func = H_full_tls,
                                                    *args,
                                                    **kwargs):
    """
    Optimizes pulse parameters using a range of carrier phases and using the
    propagator.
    """
    pulse_envelope = CosinePulseEnvelope(tg=tg)

    if not isinstance(gate, list):
        gate = [gate]

    def wrapper(x):
        cost = 0
        for phi in carrier_phases:
            for i,g in enumerate(gate):
                _V0 = pulse_envelope.V0 if g=="rx_pi" else pulse_envelope.V0/2
                H = propagator_func(tg=tg,
                                    w01=w01,
                                    wd=w01-x[0]*1e6*2*np.pi,
                                    ppp=x[1],
                                    V0=x[2+i]*_V0,
                                    pulse_envelope=pulse_envelope,
                                    carrier_phase=phi,
                                    *args,
                                    **kwargs)

                U = propagator(H, t0=tg, num_time_steps=num_time_steps, return_all=False)

                E = gates_ideal[g].conj().T @ U[:2,:2]
                cost += (2+np.trace(E)*np.trace(E.conj().T)).real/6

        return 3/2*(1-cost/len(carrier_phases)/len(gate))

    if x0s is None:
        x0s = [[0, 1/2/w01*np.pi/tg] + [1]*len(gate)]
    
    # Perform minimization with provided parameters
    delta_max = min([w01*1e-6/2/np.pi, fd_bound])
    res = minimize_global(
        wrapper,
        x0=x0s,
        bounds=(
            (-fd_bound, delta_max),
            (-ppp_bound, ppp_bound),
        ) + ((1-V0_bound, 1+V0_bound),)*len(gate),
        method="Nelder-Mead",
        options={
            'xatol': tol,
            'fatol': tol
        }
    )

    # Compute fidelities
    fidelities = np.zeros((len(carrier_phases), len(gate)))
    for i,phi in enumerate(carrier_phases):
        for j,g in enumerate(gate):
            _V0 = pulse_envelope.V0 if g=="rx_pi" else pulse_envelope.V0/2
            H = propagator_func(tg=tg,
                                w01=w01,
                                wd=w01-res["x"][0]*1e6*2*np.pi,
                                ppp=res["x"][1],
                                V0=res["x"][2+j]*_V0,
                                pulse_envelope=pulse_envelope,
                                carrier_phase=phi,
                                *args,
                                **kwargs)

            U = propagator(H, t0=tg, num_time_steps=num_time_steps, return_all=False)

            E = gates_ideal[g].conj().T @ U[:2,:2]
            fidelities[i,j] = (2+np.trace(E)*np.trace(E.conj().T)).real/6

    # Flatten if there is only one gate to optimize
    if len(gate)==1:
        fidelities = fidelities[:,0]

    return res, fidelities


def ideal_pulse_parameters_propagator_carrier_phase_protocol2(tg: float,
                                                              w01: float = 2*np.pi*6e9,
                                                              gate: str = "rx_pi",
                                                              V0_bound: float = 0.1,
                                                              ppp_bound: float = 1,
                                                              carrier_phases: list[float] | np.ndarray = [0],
                                                              x0s: list | None = None,
                                                              num_time_steps: float = 10000,
                                                              tol: float = 1e-8,
                                                              propagator_func = H_full_tls,
                                                              *args,
                                                              **kwargs):
    """
    Pulse optimization function tailored for the optimization of calibration protocol 2.
    Notice that this function fixes delta=0 and does not optimize the ppp either.
    """
    pulse_envelope = CosinePulseEnvelope(tg=tg)

    if not isinstance(gate, list):
        gate = [gate]

    def wrapper(x):
        cost = 0
        for phi in carrier_phases:
            for i,g in enumerate(gate):
                _V0 = pulse_envelope.V0 if g=="rx_pi" else pulse_envelope.V0/2
                H = propagator_func(tg=tg,
                                    w01=w01,
                                    wd=w01,
                                    ppp=x[0],
                                    V0=x[1+i]*_V0,
                                    pulse_envelope=pulse_envelope,
                                    carrier_phase=phi,
                                    *args,
                                    **kwargs)

                U = propagator(H, t0=tg, num_time_steps=num_time_steps, return_all=False)

                E = gates_ideal[g].conj().T @ U[:2,:2]
                cost += (2+np.trace(E)*np.trace(E.conj().T)).real/6

        return 3/2*(1-cost/len(carrier_phases)/len(gate))

    if x0s is None:
        x0s = [[1/2/w01*np.pi/tg] + [1]*len(gate)]
    
    # Perform minimization with provided parameters
    res = minimize_global(
        wrapper,
        x0=x0s,
        bounds=(
            (-ppp_bound, ppp_bound),
        ) + ((1-V0_bound, 1+V0_bound),)*len(gate),
        method="Nelder-Mead",
        options={
            'xatol': tol,
            'fatol': tol
        }
    )

    # Compute fidelities
    fidelities = np.zeros((len(carrier_phases), len(gate)))
    for i,phi in enumerate(carrier_phases):
        for j,g in enumerate(gate):
            _V0 = pulse_envelope.V0 if g=="rx_pi" else pulse_envelope.V0/2
            H = propagator_func(tg=tg,
                                w01=w01,
                                wd=w01,
                                ppp=res["x"][0],
                                V0=res["x"][1+j]*_V0,
                                pulse_envelope=pulse_envelope,
                                carrier_phase=phi,
                                *args,
                                **kwargs)

            U = propagator(H, t0=tg, num_time_steps=num_time_steps, return_all=False)

            E = gates_ideal[g].conj().T @ U[:2,:2]
            fidelities[i,j] = (2+np.trace(E)*np.trace(E.conj().T)).real/6

    # Flatten if there is only one gate to optimize
    if len(gate)==1:
        fidelities = fidelities[:,0]

    return res, fidelities


def ideal_pulse_parameters_propagator_carrier_phase_tdwd(tg: float,
                                                         alpha2: float,
                                                         alpha3: float,
                                                         mat_elems: np.ndarray,
                                                         w01: float = 2*np.pi*6e9,
                                                         gate: str = "rx_pi",
                                                         V0_bound: tuple | float = 0.1,
                                                         ppp_bound: tuple | float = 1,
                                                         fd_bound: float = 5,
                                                         tdwd_bounds: list = [0,0],
                                                         carrier_phases: list[float] | np.ndarray = [0],
                                                         x0s: list | None = None,
                                                         num_time_steps: float = 10000,
                                                         tol: float = 1e-8,
                                                         propagator_func = H_full_4_levels,
                                                         *args,
                                                         **kwargs):
    """
    Optimizes pulse parameters using a range of carrier phases and using the
    propagator.
    """
    pulse_envelope = CosinePulseEnvelope(tg=tg)

    if isinstance(V0_bound, NumberTypes):
        V0_bound = (1-V0_bound, 1+V0_bound)
    if isinstance(ppp_bound, NumberTypes):
        ppp_bound = (-ppp_bound, ppp_bound)

    # Units of omega_delta
    omega_unit = -1/2*(
        mat_elems[0,3]**2/alpha3 -
        mat_elems[1,2]**2/alpha2
    )

    if not isinstance(gate, list):
        gate = [gate]

    def wrapper(x):
        cost = 0
        for phi in carrier_phases:
            for i,g in enumerate(gate):
                _V0 = pulse_envelope.V0 if g=="rx_pi" else pulse_envelope.V0/2
                H = propagator_func(tg=tg,
                                    w01=w01,
                                    wd=w01-x[0]*1e6*2*np.pi,
                                    ppp=x[1],
                                    omega_delta=x[2]*omega_unit,
                                    V0=x[3+i]*_V0,
                                    pulse_envelope=pulse_envelope,
                                    carrier_phase=phi,
                                    alpha2=alpha2,
                                    alpha3=alpha3,
                                    mat_elems=mat_elems,
                                    *args,
                                    **kwargs)

                U = propagator(H, t0=tg, num_time_steps=num_time_steps, return_all=False)

                E = gates_ideal[g].conj().T @ U[:2,:2]
                cost += (2+np.trace(E)*np.trace(E.conj().T)).real/6

        return 3/2*(1-cost/len(carrier_phases)/len(gate))

    if x0s is None:
        x0s = [[0, 1/2/w01*np.pi/tg, 1] + [1]*len(gate)]
    
    # Perform minimization with provided parameters
    delta_max = min([w01*1e-6/2/np.pi, fd_bound])
    res = minimize_global(
        wrapper,
        x0=x0s,
        bounds=(
            (-fd_bound, delta_max),
            ppp_bound,
            (tdwd_bounds[0], tdwd_bounds[1]),
        ) + (V0_bound,)*len(gate),
        method="Nelder-Mead",
        options={
            'xatol': tol,
            'fatol': tol
        }
    )

    # Compute fidelities
    fidelities = np.zeros((len(carrier_phases), len(gate)))
    for i,phi in enumerate(carrier_phases):
        for j,g in enumerate(gate):
            _V0 = pulse_envelope.V0 if g=="rx_pi" else pulse_envelope.V0/2
            H = propagator_func(tg=tg,
                                w01=w01,
                                wd=w01-res["x"][0]*1e6*2*np.pi,
                                ppp=res["x"][1],
                                omega_delta=res["x"][2]*omega_unit,
                                V0=res["x"][3+j]*_V0,
                                pulse_envelope=pulse_envelope,
                                carrier_phase=phi,
                                alpha2=alpha2,
                                alpha3=alpha3,
                                mat_elems=mat_elems,
                                *args,
                                **kwargs)

            U = propagator(H, t0=tg, num_time_steps=num_time_steps, return_all=False)

            E = gates_ideal[g].conj().T @ U[:2,:2]
            fidelities[i,j] = (2+np.trace(E)*np.trace(E.conj().T)).real/6

    # Flatten if there is only one gate to optimize
    if len(gate)==1:
        fidelities = fidelities[:,0]

    return res, fidelities


def ideal_pulse_parameters_propagator_carrier_phase_tdwd_protocol3(tg: float,
                                                                   alpha2: float,
                                                                   alpha3: float,
                                                                   mat_elems: np.ndarray,
                                                                   w01: float = 2*np.pi*6e9,
                                                                   gate: str = "rx_pi",
                                                                   V0_bound: tuple | float = 0.1,
                                                                   tdwd_bounds: list = [0,0],
                                                                   ppp: float = 0,
                                                                   carrier_phases: list[float] | np.ndarray = [0],
                                                                   x0s: list | None = None,
                                                                   num_time_steps: float = 10000,
                                                                   tol: float = 1e-8,
                                                                   propagator_func = H_full_4_levels,
                                                                   *args,
                                                                   **kwargs):
    """
    Pulse optimization function tailored for the optimization of calibration protocol 3.
    Notice that this function fixes delta=0 and does not optimize the ppp either.
    """
    pulse_envelope = CosinePulseEnvelope(tg=tg)

    if isinstance(V0_bound, NumberTypes):
        V0_bound = (1-V0_bound, 1+V0_bound)

    # Units of omega_delta
    omega_unit = -1/2*(
        mat_elems[0,3]**2/alpha3 -
        mat_elems[1,2]**2/alpha2
    )

    if not isinstance(gate, list):
        gate = [gate]

    def wrapper(x):
        cost = 0
        for phi in carrier_phases:
            for i,g in enumerate(gate):
                _V0 = pulse_envelope.V0 if g=="rx_pi" else pulse_envelope.V0/2
                H = propagator_func(tg=tg,
                                    w01=w01,
                                    wd=w01,
                                    ppp=ppp,
                                    omega_delta=x[0]*omega_unit,
                                    V0=x[1+i]*_V0,
                                    pulse_envelope=pulse_envelope,
                                    carrier_phase=phi,
                                    alpha2=alpha2,
                                    alpha3=alpha3,
                                    mat_elems=mat_elems,
                                    *args,
                                    **kwargs)

                U = propagator(H, t0=tg, num_time_steps=num_time_steps, return_all=False)

                E = gates_ideal[g].conj().T @ U[:2,:2]
                cost += (2+np.trace(E)*np.trace(E.conj().T)).real/6

        return 3/2*(1-cost/len(carrier_phases)/len(gate))

    if x0s is None:
        x0s = [[1] + [1]*len(gate)]
    
    # Perform minimization with provided parameters
    res = minimize_global(
        wrapper,
        x0=x0s,
        bounds=(
            (tdwd_bounds[0], tdwd_bounds[1]),
        ) + (V0_bound,)*len(gate),
        method="Nelder-Mead",
        options={
            'xatol': tol,
            'fatol': tol
        }
    )

    # Compute fidelities
    fidelities = np.zeros((len(carrier_phases), len(gate)))
    for i,phi in enumerate(carrier_phases):
        for j,g in enumerate(gate):
            _V0 = pulse_envelope.V0 if g=="rx_pi" else pulse_envelope.V0/2
            H = propagator_func(tg=tg,
                                w01=w01,
                                wd=w01,
                                ppp=ppp,
                                omega_delta=res["x"][0]*omega_unit,
                                V0=res["x"][1+j]*_V0,
                                pulse_envelope=pulse_envelope,
                                carrier_phase=phi,
                                alpha2=alpha2,
                                alpha3=alpha3,
                                mat_elems=mat_elems,
                                *args,
                                **kwargs)

            U = propagator(H, t0=tg, num_time_steps=num_time_steps, return_all=False)

            E = gates_ideal[g].conj().T @ U[:2,:2]
            fidelities[i,j] = (2+np.trace(E)*np.trace(E.conj().T)).real/6

    # Flatten if there is only one gate to optimize
    if len(gate)==1:
        fidelities = fidelities[:,0]

    return res, fidelities


def H_full_tls_tdwd(tg: float,
                    wd: float,
                    w01: float,
                    omega_delta: float = 0,
                    ppp: float = 0,
                    V0: None | float = None,
                    carrier_phase: float = 0,
                    envelope_phase: float = 0,
                    pulse_envelope: PulseEnvelope | None = None):
    """
    Full time evolution for a TLS with a time-dependent
    drive frequency. The time dependent drive frequency (tdwd) is set
    to compensate from the influence of higher levels.
    """
    # Define pulse envelope and drive strength
    if pulse_envelope is None:
        pulse_envelope = CosinePulseEnvelope(tg=tg)
    if V0 is None:
        V0 = pulse_envelope.V0

    # Delta(t) in lab frame
    def delta(t):
        wds = omega_delta * V0**2 / t * (
            pulse_envelope.envelope_square_int(t) +
            ppp**2 * pulse_envelope.derivative_square_int(t)
        )
        return np.nan_to_num(wds, nan=0)
    
    # Define pulse shape
    wave = lambda t: np.exp(1j*envelope_phase) * V0 * (
        pulse_envelope.envelope(t) + 
        1j * ppp * pulse_envelope.derivative(t) * wd/(wd - delta(t))
    )

    # Build Hamiltonian
    def H(t):
        # Construct drive terms
        _cos = np.cos(2*(wd-delta(t))*t+2*carrier_phase)
        _sin = np.sin(2*(wd-delta(t))*t+2*carrier_phase)

        w = wave(t)

        wx = 1/2 * (w.real*(1+_cos) + w.imag*_sin)
        wy = 1/2 * (w.imag*(1-_cos) + w.real*_sin)

        return (
            np.diag([0, w01-wd]) +
            wx * sigmax_np +  
            wy * sigmay_np
        )

    return H


def optimize_omega_delta_for_tls(tg: float,
                                 wd: float,
                                 w01: float,
                                 alpha2: float,
                                 alpha3: float,
                                 mat_elems: np.ndarray,
                                 num_phases: int):
    """
    Optimizes the drive strength and $\Omega_\Delta$ to model
    a fluxonium 4-level system as a TLS
    """
    phis = [i*np.pi/num_phases for i in range(num_phases)]
    pulse_envelope = CosinePulseEnvelope(tg=tg)
    omega_unit = -1/2*(
        mat_elems[0,3]**2/alpha3 -
        mat_elems[1,2]**2/alpha2
    )

    # Fun for minimizer
    def fun(x):
        F = 0
        for phi in phis:
            # Simulate TLS
            H1 = H_full_tls_tdwd(tg=tg,
                                 wd=wd,
                                 w01=w01,
                                 omega_delta=x[1]*omega_unit,
                                 ppp=0,
                                 carrier_phase=phi,
                                 V0=pulse_envelope.V0,
                                 pulse_envelope=pulse_envelope)
            
            U1 = propagator(H1, t0=tg, num_time_steps=10000, U0=np.eye(2, dtype=complex), atol=1e-13, rtol=1e-13)

            # Simulate four-level system
            H2 = H_full_4_levels(tg=tg,
                                 wd=wd,
                                 w01=w01,
                                 alpha2=alpha2,
                                 alpha3=alpha3,
                                 mat_elems=mat_elems,
                                 omega_delta=x[1]*omega_unit,
                                 ppp=0,
                                 carrier_phase=phi,
                                 V0=x[0]*pulse_envelope.V0,
                                 pulse_envelope=pulse_envelope)
        
            U2 = propagator(H2, t0=tg, num_time_steps=10000, U0=np.eye(4, dtype=complex), atol=1e-13, rtol=1e-13)

            E = U1.conj().T @ U2[:2,:2]
            F += (2+np.trace(E)*np.trace(E.conj().T)).real/6/len(phis)
        return 3/2 * (1-F)
    
    # Calculate fidelity without correction
    no_correction = fun([1,0])

    # Minimize
    x0s = [[1,1]]
    bounds = ((0.3,1.5),(0,2))
    res = minimize_global(fun,
                          x0s,
                          bounds=bounds,
                          tol=1e-10,
                          method="Nelder-Mead")
    
    return res["x"], res["fun"], no_correction


def optimize_drive_strength(tg: float,
                            hamiltonian: Callable,
                            num_phases: int = 1,
                            target: float = 1,
                            *args,
                            **kwargs):
    pulse_envelope = CosinePulseEnvelope(tg=tg)
    phases = np.linspace(0,np.pi,num_phases)

    k0 = np.array([1,0,0,0]).reshape((4,1))
    k1 = np.array([0,1,0,0]).reshape((1,4))
    
    def fun(x):
        C = 0
        for phase in phases:
            H = hamiltonian(tg=tg,
                            carrier_phase=phase,
                            V0=x[0]*pulse_envelope.V0,
                            *args,
                            **kwargs)
            
            U = propagator(H, t0=tg, num_time_steps=10000, U0=np.eye(4, dtype=complex), atol=1e-13, rtol=1e-13)
            C += (target - np.abs((k1 @ U @ k0).item())**2)**2 / len(phases)

        return C
    
    x0s = [1*target]
    bounds = ((0.8*target,1.2*target),)

    res = minimize_global(fun,
                           x0s,
                           bounds=bounds,
                           tol=1e-11,
                           method="Nelder-Mead")
    
    return res["x"], res["fun"]


def optimize_drive_strength_fidelity(tg: float,
                                     hamiltonian: Callable,
                                     num_phases: int = 1,
                                     gate: str = "rx_pi",
                                     bound: float = 0.5,
                                     x0s: list = None,
                                     U0: np.ndarray = np.eye(2, dtype=complex),
                                     *args,
                                     **kwargs):
    pulse_envelope = CosinePulseEnvelope(tg=tg)
    _V0 = pulse_envelope.V0 if gate=="rx_pi" else pulse_envelope.V0/2
    phases = [i*np.pi/num_phases for i in range(num_phases)]
    
    def fun(x):
        cost = 0
        for phase in phases:
            H = hamiltonian(tg=tg,
                            carrier_phase=phase,
                            V0=x[0]*_V0,
                            *args,
                            **kwargs)
            
            U = propagator(H, t0=tg, num_time_steps=10000, U0=U0, atol=1e-13, rtol=1e-13)
            E = gates_ideal[gate].conj().T @ U[:2,:2]
            cost += (2+np.trace(E)*np.trace(E.conj().T)).real/6

        return 3/2*(1-cost/num_phases)
    
    if x0s is None:
        x0s = [[1]]
    bounds = ((1-bound, 1+bound),)

    res = minimize_global(fun,
                          x0s,
                          bounds=bounds,
                          tol=1e-12,
                          method="Nelder-Mead")
    
    return res["x"], res["fun"]
