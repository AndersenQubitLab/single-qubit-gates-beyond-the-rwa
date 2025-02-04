import numpy as np
import sympy
import scipy
import multiprocessing
from itertools import repeat
import pickle
from sqgbrwa.utils import *
from sqgbrwa.pulse_envelopes import *


"""
This file contains functions that calculate the ideal pulse parameters for the 
0th order or 1st order Magnus approximation. Both numerical and analytical ideal
pulse parameters can be found here.

For definitions regarding the 9 integrals that calculate the detuning,
see the notebook 'detuning-validation.ipynb'.
"""


def integral_lambda_numerical1(pulse_envelope: PulseEnvelope, 
                               t1: float, 
                               t2: float, 
                               wd: float,
                               phi: float):
    """
    Returns integrand of integral 9 for numerical integration
    """
    return 1/4 * (
        pulse_envelope.envelope(t1)-
        pulse_envelope.envelope(t2)
    )

def integral_lambda_numerical2(pulse_envelope: PulseEnvelope, 
                               t1: float, 
                               t2: float, 
                               wd: float,
                               phi: float):
    """
    Returns integrand of integral 9 for numerical integration
    """
    return 1/4 * (
        pulse_envelope.envelope(t1)*np.cos(2*wd*t1+2*phi)-
        pulse_envelope.envelope(t2)*np.cos(2*wd*t2+2*phi)
    )

def integral_lambda_numerical3(pulse_envelope: PulseEnvelope, 
                               t1: float, 
                               t2: float, 
                               wd: float,
                               phi: float):
    """
    Returns integrand of integral 9 for numerical integration
    """
    return 1/4 * (
        pulse_envelope.derivative(t1)*np.sin(2*wd*t1+2*phi)-
        pulse_envelope.derivative(t2)*np.sin(2*wd*t2+2*phi)
    )


def calculate_theory_pulse_parameters(tg: float,
                                      wd: float,
                                      beta: float = 0,
                                      t0: float = 0,
                                      K: int = 8,
                                      detuning: float = 0,
                                      ppp_theory: float = 0,
                                      sympy_integrals: list | None = None):
    """
    Calculates the theoretical pulse parameters for the 0th order
    Magnus expansion.

    Parameters
    ----------
    tg : float
        Gate duration
    wd : float
        Drive frequency in angular frequency
    beta : float
        Dimensionless parameter set by `phi` and `t0`
    t0 : float
        t0
    K : float
        Truncation number of correction terms
    detuning : float
        Detuning
    ppp_theory : float
        ppp to use in first order corrections (only used for
        non-zero detuning)
    sympy_integrals : list
        Integrals from `analytical_omega_lambda_sympy_functions`
    """
    tc = np.pi/wd
    N = np.floor((tg-t0)/tc).astype(int)

    # Define pulse envelope
    pulse_envelope = CosinePulseEnvelope(tg=tg)

    # Define init and final times
    tcm = t0
    tcp = t0 + N*tc

    # Evaluate equations (10a) and (10b) from the paper
    num = 0
    denum = pulse_envelope.envelope(tcp)-pulse_envelope.envelope(tcm)
    termx1 = 0
    termx2 = 0

    for k in range(K):
        fun1,fun2 = (np.cos,np.sin) if k%2==0 else (np.sin,np.cos)
        sign1 = (-1)**np.floor((k+1)/2).astype(int)
        sign2 = -(-1)**np.floor((k)/2).astype(int)

        num += (
            sign1 * (1/2/wd)**(k+1) * fun1(beta) * (dp(tcp,k,pulse_envelope) - dp(tcm,k,pulse_envelope))
        )
        denum += (
            sign2 * (1/2/wd)**(k+1) * fun2(beta) * (dp(tcp,k+1,pulse_envelope) - dp(tcm,k+1,pulse_envelope))
        )
        termx1 += (
            -1*sign2 * (1/2/wd)**(k+1) * fun2(beta) * (dp(tcp,k,pulse_envelope) - dp(tcm,k,pulse_envelope))
        )
        termx2 += (
            -1*sign1 * (1/2/wd)**(k+1) * fun1(beta) * (dp(tcp,k+1,pulse_envelope) - dp(tcm,k+1,pulse_envelope))
        )

    termx3 = 0
    if detuning != 0:
        # If the detuning is nonzero, corrections from the
        # first order Magnus approximation are calculated
        # and taken into account. However, as explained
        # in the paper, this does not necessarily make 
        # the theoretical pulse parameters more accurate.
        if sympy_integrals is None:
            raise ValueError(f"Nonzero detuning but no sympy integrals provided")

        termx3 = detuning*2 * (
            sympy_integrals[0](wd=wd,
                               tg=tg,
                               beta=beta,
                               ppp=ppp_theory,
                               b_min=tcm,
                               b_plus=tcp) + 
            sympy_integrals[1](wd=wd,
                               tg=tg,
                               beta=beta,
                               ppp=ppp_theory,
                               b_min=tcm,
                               b_plus=tcp) +
            sympy_integrals[2](wd=wd,
                               tg=tg,
                               beta=beta,
                               ppp=ppp_theory,
                               b_min=tcm,
                               b_plus=tcp)
        )

        t1 = detuning*2 * (
            sympy_integrals[4](wd=wd,
                               tg=tg,
                               beta=beta,
                               ppp=ppp_theory,
                               b_min=tcm,
                               b_plus=tcp)
        )
        t2 = detuning*2 * sympy_integrals[3](wd=wd,
                               tg=tg,
                               beta=beta,
                               ppp=ppp_theory,
                               b_min=tcm,
                               b_plus=tcp)

        t3 = detuning*2 * (
            sympy_integrals[5](wd=wd,
                               tg=tg,
                               beta=beta,
                               ppp=1,
                               b_min=tcm,
                               b_plus=tcp)
        )

        ## If you want, you can also use numerical integrals
        # f = lambda y,x: integral_lambda_numerical1(t1=x,t2=y,pulse_envelope=pulse_envelope,wd=wd,phi=(beta-2*wd*t0)/2)*2*detuning
        # t1,err = scipy.integrate.dblquad(f, 0, tg, 0, lambda x: x, epsabs=1e-13, epsrel=1e-13)
        # f = lambda y,x: integral_lambda_numerical2(t1=x,t2=y,pulse_envelope=pulse_envelope,wd=wd,phi=(beta-2*wd*t0)/2)*2*detuning
        # t2,err = scipy.integrate.dblquad(f, 0, tg, 0, lambda x: x, epsabs=1e-13, epsrel=1e-13)
        # f = lambda y,x: integral_lambda_numerical3(t1=x,t2=y,pulse_envelope=pulse_envelope,wd=wd,phi=(beta-2*wd*t0)/2)*2*detuning
        # t3,err = scipy.integrate.dblquad(f, 0, tg, 0, lambda x: x, epsabs=1e-13, epsrel=1e-13)
        # t3 *= np.pi/tg

        # if not np.abs((t1+t2)/num)>0.1:
        num -= (t1+t2)
        denum += t3

    # Evaluate ppp
    if np.isclose(denum,0):
        # This happens when pulses are commensurate, and we 
        # need to catch it since it will raise a division by
        # zero warning
        # Exact result for beta=0 and symmetric integration windows
        ppp_theory = 1/2/wd
    else:
        ppp_theory = num/denum

    # Drive amp
    d = 1/2*N*tc + tg/4/np.pi * (np.sin(2*np.pi*tcm/tg) - np.sin(2*np.pi*tcp/tg))
    V0_theory = d / (
        d + 
        # Ex terms
        termx1 +
        # Ey terms
        ppp_theory * termx2 +
        # Detuning term
        termx3
    )

    return ppp_theory, V0_theory


def integrated_envelope(pulse_envelope: PulseEnvelope,
                        t1: float,
                        t2: float):
    """Returns the integral of Ex(t)"""
    return (
        1/2 * (t2-t1) + pulse_envelope.tg/np.pi/4 * (
            -np.sin(2*np.pi/pulse_envelope.tg * t2) + np.sin(2*np.pi/pulse_envelope.tg * t1)
        )
    )


def integrated_envelope1(pulse_envelope: PulseEnvelope,
                         t1: float):
    """Returns the first anti-derivative of sx(t1)"""
    return t1/2 - pulse_envelope.tg * np.sin(2*np.pi*t1/pulse_envelope.tg)


def integrated_envelope2(pulse_envelope: PulseEnvelope,
                         t1: float):
    """Returns the second anti-derivative of sx(t1)"""
    return 1/8 * (
        2 * t1**2 + pulse_envelope.tg**2 / np.pi**2 * np.cos(2*np.pi*t1/pulse_envelope.tg) 
    )


def integrated_envelope_squared(pulse_envelope: PulseEnvelope,
                                t1: float,
                                t2: float):
    """Returns in the integral of (Ex(t))^2"""
    return 1/4 * (
        3/2 * (t2-t1) + pulse_envelope.tg/np.pi * (
            -np.sin(2*np.pi/pulse_envelope.tg * t2) + np.sin(2*np.pi/pulse_envelope.tg * t1)
        ) + pulse_envelope.tg/np.pi/8 * (
            np.sin(4*np.pi/pulse_envelope.tg * t2) - np.sin(4*np.pi/pulse_envelope.tg * t1)
        )
    )

def integrated_derivative_squared(pulse_envelope: PulseEnvelope,
                                  t1: float,
                                  t2: float):
    """Returns in the integral of (Ey(t))^2"""
    return 1/2 * (
        (t2-t1) + pulse_envelope.tg/np.pi/4 * (
            -np.sin(4*np.pi/pulse_envelope.tg * t2) + np.sin(4*np.pi/pulse_envelope.tg * t1)
        )
    )


# Define derivatives of pulse envelope
def dp(t: float,
       n: int,
       pulse_envelope: PulseEnvelope):
    """Returns the n-th derivative of the pulse envelope
    
    NOTE could've also used Sympy for this but you know,
         it is what it is.
    """
    if n<0:
        raise ValueError("n must be >=0")
    if n==0:
        # n=0 is an edge case
        return pulse_envelope.envelope(t)

    sign = (-1)**np.floor((n-1)/2).astype(int)
    prefac = 1/2 * (2*np.pi/pulse_envelope.tg)**n
    fun = np.sin if n%2==1 else np.cos

    return (
        sign * prefac * fun(2*np.pi*t/pulse_envelope.tg)
    )


def integral1_numerical(pulse_envelope: PulseEnvelope, 
                        t1: float, 
                        t2: float, 
                        wd: float,
                        phi: float,
                        lambda_hat: float):
    """
    Returns integrand of integral 1 for numerical integration
    """
    return 1/4 * (
        pulse_envelope.envelope(t1)*pulse_envelope.envelope(t2)*np.sin(2*wd*t2 + 2*phi) - 
        pulse_envelope.envelope(t2)*pulse_envelope.envelope(t1)*np.sin(2*wd*t1 + 2*phi)
    )


def integral1_analytical(pulse_envelope: PulseEnvelope,
                         t1: float,
                         t2: float,
                         wd: float,
                         beta: float,
                         lambda_hat: float):
    """
    Returns analytical result for integral 1, computed up
    to and including 4th order in tc/tg
    """
    return (
        np.cos(beta)/(8*wd) * (
            pulse_envelope.envelope(t2) + pulse_envelope.envelope(t1)
        ) * integrated_envelope(pulse_envelope=pulse_envelope,
                                t1=t1,
                                t2=t2)
    ) + np.sin(beta)/4/(2*wd)**2 * (
        -(dp(t2, 1, pulse_envelope) + dp(t1, 1, pulse_envelope)) * integrated_envelope(pulse_envelope=pulse_envelope,
                                                                                      t1=t1,
                                                                                      t2=t2)
        - dp(t2, 0, pulse_envelope)**2 + dp(t1, 0, pulse_envelope)**2
    ) + np.cos(beta)/4/(2*wd)**3 * (
        -(dp(t2, 2, pulse_envelope) + dp(t1, 2, pulse_envelope)) * integrated_envelope(pulse_envelope=pulse_envelope,
                                                                                       t1=t1,
                                                                                       t2=t2) -
        3 * (dp(t2, 1, pulse_envelope)*dp(t2, 0, pulse_envelope) - dp(t1, 1, pulse_envelope)*dp(t1, 0, pulse_envelope))
    ) + np.sin(beta)/4/(2*wd)**4 * (
        (dp(t1, 3, pulse_envelope)+dp(t2, 3, pulse_envelope)) * integrated_envelope(pulse_envelope=pulse_envelope,
                                                                                    t1=t1,
                                                                                    t2=t2)
        + 4 * (dp(t2, 2, pulse_envelope) - dp(t1, 2, pulse_envelope))
        + 3 * (dp(t2, 1, pulse_envelope)**2 - dp(t1, 1, pulse_envelope)**2)
    )


def integral2_numerical(pulse_envelope: PulseEnvelope, 
                        t1: float, 
                        t2: float, 
                        wd: float,
                        phi: float,
                        lambda_hat: float):
    """
    Returns integrand of integral 2 for numerical integration
    """
    return 1/4 * lambda_hat * (
        pulse_envelope.envelope(t1)*pulse_envelope.derivative(t2)- 
        pulse_envelope.envelope(t2)*pulse_envelope.derivative(t1)
    )


def integral2_analytical(pulse_envelope: PulseEnvelope,
                         t1: float,
                         t2: float,
                         wd: float,
                         beta: float,
                         lambda_hat: float):
    """
    Returns analytical result for integral 2, computed exactly,
    i.e. up to arbitrary order in tc/tg
    """
    return 2 * pulse_envelope.tg/np.pi * lambda_hat/4 * (
        integrated_envelope_squared(pulse_envelope=pulse_envelope,
                                    t1=t1,
                                    t2=t2)
    ) - pulse_envelope.tg/np.pi * lambda_hat/4 * (
        pulse_envelope.envelope(t2) + pulse_envelope.envelope(t1)
    ) * integrated_envelope(pulse_envelope=pulse_envelope,
                            t1=t1,
                            t2=t2)


def integral3_numerical(pulse_envelope: PulseEnvelope, 
                        t1: float, 
                        t2: float, 
                        wd: float,
                        phi: float,
                        lambda_hat: float):
    """
    Returns integrand of integral 3 for numerical integration
    """
    return 1/4 * (
        pulse_envelope.envelope(t1)*np.cos(2*wd*t1+2*phi)*pulse_envelope.envelope(t2)*np.sin(2*wd*t2+2*phi) -
        pulse_envelope.envelope(t2)*np.cos(2*wd*t2+2*phi)*pulse_envelope.envelope(t1)*np.sin(2*wd*t1+2*phi)
    )


def integral3_analytical(pulse_envelope: PulseEnvelope,
                         t1: float,
                         t2: float,
                         wd: float,
                         beta: float,
                         lambda_hat: float):
    """
    Returns analytical result for integral 3, computed up
    to and including 1st order in tc/tg
    """
    return (
        -1/(8*wd) * integrated_envelope_squared(pulse_envelope=pulse_envelope,
                                                t1=t1,
                                                t2=t2)
    )


def integral4_numerical(pulse_envelope: PulseEnvelope, 
                        t1: float, 
                        t2: float, 
                        wd: float,
                        phi: float,
                        lambda_hat: float):
    """
    Returns integrand of integral 4 for numerical integration
    """
    return 1/4 * lambda_hat * (
        -pulse_envelope.envelope(t1)*pulse_envelope.derivative(t2)*np.cos(2*wd*t2+2*phi)+
         pulse_envelope.envelope(t2)*pulse_envelope.derivative(t1)*np.cos(2*wd*t1+2*phi)
    )


def integral4_analytical(pulse_envelope: PulseEnvelope,
                         t1: float,
                         t2: float,
                         wd: float,
                         beta: float,
                         lambda_hat: float):
    """
    Returns analytical result for integral 4, computed up
    to and including 5th order in tc/tg
    """
    return (
        np.sin(beta) * lambda_hat / (8*wd) * (
        pulse_envelope.derivative(t2) + pulse_envelope.derivative(t1)
    ) * integrated_envelope(pulse_envelope=pulse_envelope,
                            t1=t1,
                            t2=t2) 
    ) + np.cos(beta)/4/(2*wd)**2 * lambda_hat * pulse_envelope.tg/np.pi * (
        (dp(t2, 2, pulse_envelope) + dp(t1, 2, pulse_envelope)) * integrated_envelope(pulse_envelope=pulse_envelope,
                                                                                      t1=t1,
                                                                                      t2=t2)
        + dp(t2, 1, pulse_envelope)*dp(t2, 0, pulse_envelope)
        - dp(t1, 1, pulse_envelope)*dp(t1, 0, pulse_envelope)
    ) + np.sin(beta)/4/(2*wd)**3 * lambda_hat * pulse_envelope.tg/np.pi * (
        - (dp(t1, 3, pulse_envelope) + dp(t2, 3, pulse_envelope)) * integrated_envelope(pulse_envelope=pulse_envelope,
                                                                                        t1=t1,
                                                                                        t2=t2)
        + 2*(dp(t1, 2, pulse_envelope)*dp(t1, 0, pulse_envelope) - dp(t2, 2, pulse_envelope)*dp(t2, 0, pulse_envelope))
        +   (dp(t1, 1, pulse_envelope)**2 - dp(t2, 1, pulse_envelope)**2)
    ) + np.cos(beta)/4/(2*wd)**4 * lambda_hat * pulse_envelope.tg/np.pi * (
        - (dp(t2, 4, pulse_envelope) + dp(t2, 4, pulse_envelope)) * integrated_envelope(pulse_envelope=pulse_envelope,
                                                                                        t1=t1,
                                                                                        t2=t2)
        + 3 * (dp(t1, 3, pulse_envelope)*dp(t1, 0, pulse_envelope) - dp(t2, 3, pulse_envelope)*dp(t2, 0, pulse_envelope))
        + 3 * (dp(t1, 2, pulse_envelope)*dp(t1, 1, pulse_envelope) - dp(t2, 2, pulse_envelope)*dp(t2, 1, pulse_envelope))
        +     (dp(t1, 1, pulse_envelope)*dp(t1, 2, pulse_envelope) - dp(t2, 1, pulse_envelope)*dp(t2, 2, pulse_envelope))
    ) + np.sin(beta)/4/(2*wd)**5 * lambda_hat * pulse_envelope.tg/np.pi * (
        (dp(t1, 5, pulse_envelope) + dp(t2, 5, pulse_envelope)) * integrated_envelope(pulse_envelope=pulse_envelope,
                                                                                      t1=t1,
                                                                                      t2=t2)
        + 3 * (dp(t2, 4, pulse_envelope)*dp(t2, 0, pulse_envelope) - dp(t1, 4, pulse_envelope)*dp(t1, 0, pulse_envelope))
        + 6 * (dp(t2, 3, pulse_envelope)*dp(t2, 1, pulse_envelope) - dp(t1, 3, pulse_envelope)*dp(t1, 1, pulse_envelope))
        + 4 * (dp(t2, 2, pulse_envelope)*dp(t2, 2, pulse_envelope) - dp(t1, 2, pulse_envelope)*dp(t1, 2, pulse_envelope))
        +     (dp(t2, 1, pulse_envelope)*dp(t2, 3, pulse_envelope) - dp(t1, 1, pulse_envelope)*dp(t1, 3, pulse_envelope))
    )


def integral5_numerical(pulse_envelope: PulseEnvelope, 
                        t1: float, 
                        t2: float, 
                        wd: float,
                        phi: float,
                        lambda_hat: float):
    """
    Returns integrand of integral 5 for numerical integration
    """
    return 1/4 * lambda_hat * (
        pulse_envelope.envelope(t1)*np.cos(2*wd*t1+2*phi)*pulse_envelope.derivative(t2)-
        pulse_envelope.envelope(t2)*np.cos(2*wd*t2+2*phi)*pulse_envelope.derivative(t1)
    )


def integral5_analytical(pulse_envelope: PulseEnvelope,
                         t1: float,
                         t2: float,
                         wd: float,
                         beta: float,
                         lambda_hat: float):
    """
    Returns analytical result for integral 5, computed up
    to and including 3rd order in tc/tg
    """
    return np.sin(beta) * lambda_hat * pulse_envelope.tg/np.pi / (8*wd) * (
        pulse_envelope.envelope(t2)**2 - pulse_envelope.envelope(t1)**2
    ) + np.cos(beta) * lambda_hat * pulse_envelope.tg/np.pi / 4 / (2*wd)**2 * (
        2 * (dp(t2, 1, pulse_envelope)*dp(t2, 0, pulse_envelope) - dp(t1, 1, pulse_envelope)*dp(t1, 0, pulse_envelope)) +
        2 * (dp(t1, 1, pulse_envelope)*dp(t2, 0, pulse_envelope) - dp(t1, 0, pulse_envelope)*dp(t2, 1, pulse_envelope))
    ) + np.sin(beta) * lambda_hat * pulse_envelope.tg/np.pi / 4 / (2*wd)**3 * (
        2 * (dp(t1, 2, pulse_envelope)*dp(t1, 0, pulse_envelope) - dp(t2, 2, pulse_envelope)*dp(t2, 0, pulse_envelope)) +
        2 * (dp(t1, 1, pulse_envelope)**2 - dp(t2, 1, pulse_envelope)**2) +
            (dp(t1, 0, pulse_envelope)*dp(t1, 2, pulse_envelope) - dp(t1, 2, pulse_envelope)*dp(t2, 0, pulse_envelope))
    )


def integral6_numerical(pulse_envelope: PulseEnvelope, 
                        t1: float, 
                        t2: float, 
                        wd: float,
                        phi: float,
                        lambda_hat: float):
    """
    Returns integrand of integral 6 for numerical integration
    """
    return 1/4 * lambda_hat * (
        -pulse_envelope.envelope(t1)*np.cos(2*wd*t1+2*phi)*pulse_envelope.derivative(t2)*np.cos(2*wd*t2+2*phi)+
         pulse_envelope.envelope(t2)*np.cos(2*wd*t2+2*phi)*pulse_envelope.derivative(t1)*np.cos(2*wd*t1+2*phi)
    )


def integral6_analytical(pulse_envelope: PulseEnvelope,
                         t1: float,
                         t2: float,
                         wd: float,
                         beta: float,
                         lambda_hat: float):
    """
    Returns analytical result for integral 6, computed up
    to and including 2nd order in tc/tg
    """
    return (
        lambda_hat * pulse_envelope.tg / np.pi / 4 / (2*wd)**2 * (
            (np.pi/pulse_envelope.tg)**2 * integrated_derivative_squared(pulse_envelope, t1, t2) -
            1/2 * dp(t2, 0, pulse_envelope)*dp(t2, 1, pulse_envelope) + 
            1/2 * dp(t1, 0, pulse_envelope)*dp(t1, 1, pulse_envelope)
        )
    )


def integral7_numerical(pulse_envelope: PulseEnvelope, 
                        t1: float, 
                        t2: float, 
                        wd: float,
                        phi: float,
                        lambda_hat: float):
    """
    Returns integrand of integral 7 for numerical integration
    """
    return 1/4 * lambda_hat**2 * (
        pulse_envelope.derivative(t1)*np.sin(2*wd*t1+2*phi)*pulse_envelope.derivative(t2)-
        pulse_envelope.derivative(t2)*np.sin(2*wd*t2+2*phi)*pulse_envelope.derivative(t1)
    )


def integral7_analytical(pulse_envelope: PulseEnvelope,
                         t1: float,
                         t2: float,
                         wd: float,
                         beta: float,
                         lambda_hat: float):
    """
    Returns analytical result for integral 7, computed up
    to and including 1st order in tc/tg
    """
    return np.cos(beta) * lambda_hat**2 * pulse_envelope.tg/np.pi / (8*wd) * (
        pulse_envelope.derivative(t2) + pulse_envelope.derivative(t1)
    ) * (
        pulse_envelope.envelope(t1) - pulse_envelope.envelope(t2)
    )


def integral8_numerical(pulse_envelope: PulseEnvelope, 
                        t1: float, 
                        t2: float, 
                        wd: float,
                        phi: float,
                        lambda_hat: float):
    """
    Returns integrand of integral 8 for numerical integration
    """
    return 1/4 * lambda_hat**2 * (
        -pulse_envelope.derivative(t1)*np.sin(2*wd*t1+2*phi)*pulse_envelope.derivative(t2)*np.cos(2*wd*t2+2*phi)+
         pulse_envelope.derivative(t2)*np.sin(2*wd*t2+2*phi)*pulse_envelope.derivative(t1)*np.cos(2*wd*t1+2*phi)
    )


def integral8_analytical(pulse_envelope: PulseEnvelope,
                         t1: float,
                         t2: float,
                         wd: float,
                         beta: float,
                         lambda_hat: float):
    """
    Returns analytical result for integral 8, computed up
    to and including 1st order in tc/tg
    """
    return -1/(8*wd) * lambda_hat**2 * integrated_derivative_squared(pulse_envelope=pulse_envelope,
                                                                     t1=t1,
                                                                     t2=t2)


def integral9_numerical(pulse_envelope: PulseEnvelope, 
                        t1: float, 
                        t2: float, 
                        wd: float,
                        phi: float,
                        lambda_hat: float):
    """
    Returns integrand of integral 9 for numerical integration
    """
    return 1/4 * lambda_hat * (
        pulse_envelope.derivative(t1)*np.sin(2*wd*t1+2*phi)*pulse_envelope.envelope(t2)*np.sin(2*wd*t2+2*phi)-
        pulse_envelope.derivative(t2)*np.sin(2*wd*t2+2*phi)*pulse_envelope.envelope(t1)*np.sin(2*wd*t1+2*phi)
    )


def integral9_analytical(pulse_envelope: PulseEnvelope,
                         t1: float,
                         t2: float,
                         wd: float,
                         beta: float,
                         lambda_hat: float):
    """
    Returns analytical result for integral 9, computed up
    to and including 2nd order in tc/tg
    """
    return (
        lambda_hat * pulse_envelope.tg / np.pi / 4 / (2*wd)**2 * (
            (np.pi/pulse_envelope.tg)**2 * integrated_derivative_squared(pulse_envelope, t1, t2) -
            1/2 * dp(t2, 0, pulse_envelope)*dp(t2, 1, pulse_envelope) + 
            1/2 * dp(t1, 0, pulse_envelope)*dp(t1, 1, pulse_envelope)
        )
    )


analytical_integrals = [
    integral1_analytical,
    integral2_analytical,
    integral3_analytical,
    integral4_analytical,
    integral5_analytical,
    integral6_analytical,
    integral7_analytical,
    integral8_analytical,
    integral9_analytical,
]


def chi_plus(k):
    return 1 if k%2==0 else 0

def chi_min(k):
    return 1 if k%2==1 else 0

def gamma1(k,beta):
    sign = (-1)**np.floor(k/2).astype(int)
    fun = sympy.sin if k%2==0 else sympy.cos
    return fun(beta)*sign

def gamma2(k,beta):
    sign = (-1)**np.floor((k+1)/2).astype(int)
    fun = sympy.cos if k%2==0 else sympy.sin
    return fun(beta)*sign

def kderiv(expr, k):
    t = sympy.symbols('t')
    for _ in range(k):
        expr = sympy.diff(expr, t)
    return expr

def expr_integral(f, osc, K):
    if osc not in ["cos", "sin"]:
        raise ValueError(f"osc should be cos or sin, not {osc}")
    
    # Init sympy symbols
    t, wd, b_min, b_plus, beta = sympy.symbols('t, wd, b_min, b_plus, beta')

    expr = 0

    if osc=="sin":
        gamma = gamma2
    else:
        gamma = gamma1

    for k in range(K):
        deriv = kderiv(f, k)
        expr += (1/2/wd)**(k+1) * gamma(k,beta) * (deriv.subs(t,b_plus)-deriv.subs(t,b_min))
    
    if osc=="sin":
        expr *= -1

    return expr


def expr_integral_form_1(f, g, osc, K):
    """
    Returns a Sympy expression for integrals of the form:
    $\int_{b_-}^{b_+}dt_1 \int_{b_-}^{t_1}dt_2 (f(t_1)g(t_2)osc(2\omega_dt_2 + 2\phi) - g(t_1)osc(2\omega_dt_1 + 2\phi)f(t_2))
    Where `osc` is either 'cos' or 'sin', and `f` and `g` are Sympy expressions.
    `K` is the order in (2*t_c/t_g) up to which the integral is computed
    """
    if osc not in ["cos", "sin"]:
        raise ValueError(f"osc should be cos or sin, not {osc}")
    
    # Init sympy symbols
    t, wd, b_min, b_plus, beta = sympy.symbols('t, wd, b_min, b_plus, beta')

    expr = 0
    intef = sympy.integrate(f, t)

    # Define gamma based on oscillating function
    if osc=="sin":
        gamma_i = gamma2
        chi_1 = chi_plus
        chi_2 = chi_min
    else:
        gamma_i = gamma1
        chi_1 = chi_min
        chi_2 = chi_plus

    for k in range(K):
        term2 = 0
        if k>0:
            N = k-1
            # All indices k1,k2 such that k1+k2+2=k
            pairs = [(i, N - i) for i in range(N + 1)]
            for k1,k2 in pairs:
                # Calculate k1/k2-th derivatives
                dderiv = kderiv(g, k2)*f
                dderiv = kderiv(dderiv, k1)

                # Calculate sign based on oscillator
                if osc=="sin":
                    sign = (-1)**np.floor((k2+1)/2).astype(int)
                else:
                    sign = (-1)**np.floor(k2/2).astype(int)

                term2 += sign*(1/2/wd)**(2+k2+k1) * (chi_1(k2)*gamma1(k1,beta) - chi_2(k2)*gamma2(k1,beta)) * (
                    dderiv.subs(t, b_plus) - dderiv.subs(t, b_min)
                )

        # Calculate kth derivatives
        derivg = kderiv(g, k)
        derivf2 = kderiv(g*intef, k)

        # Calculate prefactor
        prefac = (1/2/wd)**(k+1) * gamma_i(k,beta)

        # Add terms to expression
        expr += term2 - prefac * (
            derivg.subs(t, b_min) * (intef.subs(t,b_plus) - intef.subs(t,b_min))
        )
        expr -= prefac * (
            derivf2.subs(t, b_plus) - derivf2.subs(t, b_min) -
            intef.subs(t, b_min) * (derivg.subs(t, b_plus) - derivg.subs(t, b_min))
        )

    if osc=="sin":
        # Sin picks-up a total minus sign
        expr *= -1
    
    return expr


def expr_integral_form_2(f, g, osc1, osc2, K):
    """
    Returns a Sympy expression for integrals of the form:
    $\int_{b_-}^{b_+}dt_1 \int_{b_-}^{t_1}dt_2 f(t_1)*osc(2\omega_dt_1+2\phi)g(t_2)osc(2\omega_dt_2 + 2\phi)
    Where `osc` is either 'cos' or 'sin', and `f` and `g` are Sympy expressions.
    `K` is the order in (2*t_c/t_g) up to which the integral is computed
    """
    if osc1 not in ["cos", "sin"]:
        raise ValueError(f"osc should be cos or sin, not {osc1}")
    if osc2 not in ["cos", "sin"]:
        raise ValueError(f"osc should be cos or sin, not {osc2}")

    b = (osc1=="cos" and osc2=="sin") or (osc1=="sin" and osc2=="cos")

    # Init sympy symbols
    t, wd, b_min, b_plus, beta = sympy.symbols('t, wd, b_min, b_plus, beta')

    gamma_i = gamma2 if osc1=="sin" else gamma1
    gamma_j = gamma2 if osc2=="sin" else gamma1

    expr = 0
    for k in range(K):
        term1 = 0
        term2 = 0
        if k>0:
            N = k-1
            # Calculate prefac
            prefac2 = (1/2/wd)**(2+N)
            if b:
                prefac = -prefac2
            else:
                prefac = prefac2

            # All indices k1,k2 such that k1+k2+2=k
            pairs = [(i, N - i) for i in range(N + 1)]
            for k1,k2 in pairs:
                # Calculate term1
                gderiv = kderiv(g,k2)
                fderiv = kderiv(f,k1)

                term1 += gamma_i(k1,beta)*gamma_j(k2,beta) * gderiv.subs(t, b_min) * (
                    fderiv.subs(t, b_plus) - fderiv.subs(t, b_min)
                )

                # Calculate term2
                if osc2=="sin":
                    sign = (-1)**np.floor((k2+1)/2).astype(int)
                else:
                    sign = (-1)**np.floor(k2/2).astype(int)

                k1k2deriv = kderiv(f*gderiv, k1)

                if osc1=="sin" and osc2=="sin":
                    mp = chi_min(k2) * gamma1(k1,2*beta) + chi_plus(k2) * gamma2(k1,2*beta) 
                elif osc1=="sin" and osc2=="cos":
                    mp = -chi_plus(k2) * gamma1(k1,2*beta) - chi_min(k2) * gamma2(k1,2*beta)
                elif osc1=="cos" and osc2=="sin":
                    mp = -chi_plus(k2) * gamma1(k1,2*beta) + chi_min(k2) * gamma2(k1,2*beta)
                elif osc1=="cos" and osc2=="cos":
                    mp = chi_min(k2) * gamma1(k1,2*beta) - chi_plus(k2) * gamma2(k1,2*beta)

                term2 += sign * (1/2)**(2+k1) * (mp) * (k1k2deriv.subs(t, b_plus) - k1k2deriv.subs(t, b_min))

            term1 *= prefac
            term2 *= prefac2

        # Simplifying the to-be-integrated expression here
        # significantly speeds-up the integration
        arg = sympy.FU["TR0"](f*kderiv(g,k))
        term3 = 1/2*(1/2/wd)**(k+1) * (
            sympy.integrate(arg, (t,b_min,b_plus))
        )

        if osc2=="sin":
            sign = -(-1)**np.floor((k+1)/2).astype(int)
        elif osc2=="cos":
            sign = (-1)**np.floor(k/2).astype(int)

        if osc1==osc2:
            chi = chi_min
        else:
            chi = chi_plus
        
        expr += (chi(k)*sign*term3 + term2 - term1)

    return expr


def analytical_omega_lambda_sympy_functions(K: int):
    """
    Calculates and returns a list of 6 double integrals
    that need to be solved to compute the drive strength
    and ppp in the first oder Magnus approximation
    """
    t, wd, tg, b_min, b_plus, ppp, beta = sympy.symbols('t, wd, tg, b_min, b_plus, ppp, beta')
    EI = 1/2 * (1 - sympy.cos(2*sympy.pi*t/tg))
    EQ = ppp * sympy.sin(2*sympy.pi*t/tg)

    inte = sympy.integrate(EI, t)
        
    integrals = [
        2 * ppp * sympy.integrate(EI, (t, b_min, b_plus)) - ppp * (b_plus - b_min) * (EI.subs(t, b_plus) + EI.subs(t, b_min)),
        -ppp * sympy.pi/tg * expr_integral_form_1(f=1, g=EQ, osc="cos", K=K),
        expr_integral_form_1(f=1, g=EI, osc="sin", K=K),
        sympy.integrate(((t-b_min)*EI - inte + inte.subs(t,b_min)), (t, b_min, b_plus)),
        -expr_integral_form_1(f=1, g=EI, osc="cos", K=K),
        -ppp * sympy.pi/tg * expr_integral_form_1(f=1, g=EQ, osc="sin", K=K)
    ]

    integrals = [expr/4 for expr in integrals]
    integrals = [sympy.utilities.lambdify([wd, tg, beta, ppp, b_min, b_plus], expr) for expr in integrals]

    return integrals


def analytical_comm_sympy_functions(K: int):
    """
    If it works I will write documentation
    """
    t, wd, wd_tilde, tg, b_min, b_plus, ppp, beta, phi_tilde = sympy.symbols('t, wd, wd_tilde, tg, b_min, b_plus, ppp, beta, phi_tilde')
    # phi_tilde = -wd_tilde*tg/2

    EI = 1/2 * (1 - sympy.cos(2*sympy.pi*t/tg))
    EQ = ppp * sympy.pi/tg * sympy.sin(2*sympy.pi*t/tg)

    cos_match = sympy.cos(2*wd_tilde*t + 2*phi_tilde)
    sin_match = sympy.sin(2*wd_tilde*t + 2*phi_tilde)

    inte = sympy.integrate(EI, t)

    integrals = [
        # First order ~sigma_x
        sympy.integrate(EI, (t,b_min,b_plus)),
        expr_integral(EI*cos_match, "cos", K=K) - expr_integral(EI*sin_match, "sin", K=K),
        expr_integral(EQ*cos_match, "sin", K=K) + expr_integral(EQ*sin_match, "cos", K=K),
        # First order ~sigma_y
        sympy.integrate(EQ, (t,b_min,b_plus)),
        -(expr_integral(EQ*cos_match, "cos", K=K) - expr_integral(EQ*sin_match, "sin", K=K)),
        expr_integral(EI*cos_match, "sin", K=K) + expr_integral(EI*sin_match, "cos", K=K),
        # Second order ~sigma_x
        1/4 * (2 * sympy.integrate(EI, (t, b_min, b_plus)) - (b_plus - b_min) * (EI.subs(t, b_plus) + EI.subs(t, b_min))),
        -1/4 * (expr_integral_form_1(f=1, g=EQ*cos_match, osc="cos", K=K) - expr_integral_form_1(f=1, g=EQ*sin_match, osc="sin", K=K)),
        1/4 * (expr_integral_form_1(f=1, g=EI*cos_match, osc="sin", K=K) + expr_integral_form_1(f=1, g=EI*sin_match, osc="cos", K=K)),
        # Second order ~sigma_y
        1/4 * (sympy.integrate(((t-b_min)*EI - inte + inte.subs(t,b_min)), (t, b_min, b_plus))),
        -1/4 * (expr_integral_form_1(f=1, g=EI*cos_match, osc="cos", K=K) - expr_integral_form_1(f=1, g=EI*sin_match, osc="sin", K=K)),
        -1/4 * (expr_integral_form_1(f=1, g=EQ*cos_match, osc="sin", K=K) + expr_integral_form_1(f=1, g=EQ*sin_match, osc="cos", K=K))
    ]

    integrals = [sympy.utilities.lambdify([wd, wd_tilde, tg, beta, ppp, b_min, b_plus, phi_tilde], expr) for expr in integrals]
    return integrals


def analytical_detuning_sympy_functions(K: int):
    """
    Calculates and returns a list of nine functions
    that calculate "the" 9 integrals for the detuning
    using Sympy
    """
    t, wd, tg, b_min, b_plus, ppp, beta = sympy.symbols('t, wd, tg, b_min, b_plus, ppp, beta')
    EI = 1/2 * (1 - sympy.cos(2*sympy.pi*t/tg))
    EQ = sympy.sin(2*sympy.pi*t/tg)
    
    integrals = [
        expr_integral_form_1(f=EI, g=EI, osc="sin", K=K),
        ppp*tg/sympy.pi * (2*sympy.integrate(EI**2, (t,b_min,b_plus)) - (EI.subs(t,b_plus) + EI.subs(t,b_min)) * sympy.integrate(EI, (t,b_min,b_plus))),
        expr_integral_form_2(f=EI, g=EI, osc1="cos", osc2="sin", K=K) - expr_integral_form_2(f=EI, g=EI, osc1="sin", osc2="cos", K=K),
        -ppp*expr_integral_form_1(f=EI, g=EQ, osc="cos", K=K),
        -ppp*expr_integral_form_1(f=EQ, g=EI, osc="cos", K=K),
        ppp*(expr_integral_form_2(f=EQ, g=EI, osc1="cos", osc2="cos", K=K) - expr_integral_form_2(f=EI, g=EQ, osc1="cos", osc2="cos", K=K)),
        -ppp**2*expr_integral_form_1(f=EQ, g=EQ, osc="sin", K=K),
        ppp**2*(expr_integral_form_2(f=EQ, g=EQ, osc1="cos", osc2="sin", K=K) - expr_integral_form_2(f=EQ, g=EQ, osc1="sin", osc2="cos", K=K)),
        ppp*(expr_integral_form_2(f=EQ, g=EI, osc1="sin", osc2="sin", K=K) - expr_integral_form_2(f=EI, g=EQ, osc1="sin", osc2="sin", K=K))
    ]

    integrals = [expr/4 for expr in integrals]
    integrals = [sympy.utilities.lambdify([wd, tg, beta, ppp, b_min, b_plus], expr) for expr in integrals]

    return integrals


def analytical_detuning_comm_sympy_functions(K: int,
                                             with_mp = True,
                                             processes: int = 1):
    """
    Calculates and returns a list of nine functions
    that calculate "the" 9 integrals for the detuning
    using Sympy
    """
    t, wd, wd_tilde, tg, b_min, b_plus, ppp, beta = sympy.symbols('t, wd, wd_tilde, tg, b_min, b_plus, ppp, beta')
    phi_tilde = -wd_tilde*tg/2

    EI = 1/2 * (1 - sympy.cos(2*sympy.pi*t/tg))
    EQ = sympy.sin(2*sympy.pi*t/tg)

    cos_match = sympy.cos(2*wd_tilde*t + 2*phi_tilde)
    sin_match = sympy.sin(2*wd_tilde*t + 2*phi_tilde)

    if with_mp:
        # Computing `expr_integral_form_2(...)` takes quite some time,
        # so we quickly toss them into a MP pool
        args_iter = repeat([])
        base_arguments = dict(K=K)
        args_f = [EI*cos_match, EI*cos_match, EI*sin_match, EI*sin_match, EI*cos_match, EI*sin_match, EI*sin_match, EI*cos_match,
                  EQ*cos_match, EQ*sin_match, EQ*cos_match, EQ*sin_match, EI*cos_match, EI*sin_match, EI*cos_match, EI*sin_match, 
                  EQ*cos_match, EQ*sin_match, EQ*sin_match, EQ*cos_match, EQ*cos_match, EQ*cos_match, EQ*sin_match, EQ*sin_match, 
                  EQ*cos_match, EQ*sin_match, EQ*cos_match, EQ*sin_match, EI*cos_match, EI*sin_match, EI*cos_match, EI*sin_match]
        args_g = [EI*cos_match, EI*sin_match, EI*sin_match, EI*cos_match, EI*cos_match, EI*cos_match, EI*sin_match, EI*sin_match, 
                  EI*cos_match, EI*cos_match, EI*sin_match, EI*sin_match, EQ*cos_match, EQ*cos_match, EQ*sin_match, EQ*sin_match, 
                  EQ*cos_match, EQ*cos_match, EQ*sin_match, EQ*sin_match, EQ*cos_match, EQ*sin_match, EQ*sin_match, EQ*cos_match, 
                  EI*cos_match, EI*cos_match, EI*sin_match, EI*sin_match, EQ*cos_match, EQ*cos_match, EQ*sin_match, EQ*sin_match]
        args_osc1 = ["cos", "cos", "sin", "sin", "sin", "cos", "cos", "sin",
                     "cos", "sin", "cos", "sin", "cos", "sin", "cos", "sin",
                     "cos", "sin", "sin", "cos", "sin", "sin", "cos", "cos",
                     "sin", "cos", "sin", "cos", "sin", "cos", "sin", "cos"]
        args_osc2 = ["sin", "cos", "cos", "sin", "cos", "cos", "sin", "sin",
                     "cos", "cos", "sin", "sin", "cos", "cos", "sin", "sin",
                     "sin", "sin", "cos", "cos", "cos", "sin", "sin", "cos",
                     "sin", "sin", "cos", "cos", "sin", "sin", "cos", "cos"]
        
        iter_obj = zip(args_f, args_g, args_osc1, args_osc2)
        kwargs_iter = [
            {**base_arguments,
            'f': f,
            'g': g,
            'osc1': osc1,
            'osc2': osc2} for f,g,osc1,osc2 in iter_obj
        ]

        # Execute Pool
        pool = multiprocessing.Pool(processes=processes)
        results = starmap_with_kwargs(pool, expr_integral_form_2, args_iter, kwargs_iter)
        pool.close()
        pool.join()

        # Build integrals
        integrals = [
            (expr_integral_form_1(f=EI, g=EI*cos_match, osc="sin", K=K) + expr_integral_form_1(f=EI, g=EI*sin_match, osc="cos", K=K)),
            ppp*tg/sympy.pi * (2*sympy.integrate(EI**2, (t,b_min,b_plus)) - (EI.subs(t,b_plus) + EI.subs(t,b_min)) * sympy.integrate(EI, (t,b_min,b_plus))),
            (results[0] + results[1] - results[2] - results[3]) - (results[4] + results[5] - results[6] - results[7]),
            -ppp*(expr_integral_form_1(f=EI, g=EQ*cos_match, osc="cos", K=K) - expr_integral_form_1(f=EI, g=EQ*sin_match, osc="sin", K=K)),
            -ppp*(expr_integral_form_1(f=EQ, g=EI*cos_match, osc="cos", K=K) - expr_integral_form_1(f=EQ, g=EI*sin_match, osc="sin", K=K)),
            ppp*((results[8] - results[9] - results[10] + results[11]) - (results[12] - results[13] - results[14] + results[15])),
            -ppp**2*(expr_integral_form_1(f=EQ, g=EQ*cos_match, osc="sin", K=K) + expr_integral_form_1(f=EQ, g=EQ*cos_match, osc="sin", K=K)),
            ppp**2*((results[16] - results[17] - results[18] + results[19]) - (results[20] - results[21] - results[22] + results[23])),
            ppp*((results[24] + results[25] + results[26] + results[27]) - (results[28] + results[29] + results[30] + results[31]))
        ]
    else:
        integrals = [
            (expr_integral_form_1(f=EI, g=EI*cos_match, osc="sin", K=K) + expr_integral_form_1(f=EI, g=EI*sin_match, osc="cos", K=K)),
            ppp*tg/sympy.pi * (2*sympy.integrate(EI**2, (t,b_min,b_plus)) - (EI.subs(t,b_plus) + EI.subs(t,b_min)) * sympy.integrate(EI, (t,b_min,b_plus))),
            (expr_integral_form_2(f=EI*cos_match, g=EI*cos_match, osc1="cos", osc2="sin", K=K) + expr_integral_form_2(f=EI*cos_match, g=EI*sin_match, osc1="cos", osc2="cos", K=K) - expr_integral_form_2(f=EI*sin_match, g=EI*sin_match, osc1="sin", osc2="cos", K=K) - expr_integral_form_2(f=EI*sin_match, g=EI*cos_match, osc1="sin", osc2="sin", K=K)) - (expr_integral_form_2(f=EI*cos_match, g=EI*cos_match, osc1="sin", osc2="cos", K=K) + expr_integral_form_2(f=EI*sin_match, g=EI*cos_match, osc1="cos", osc2="cos", K=K) - expr_integral_form_2(f=EI*sin_match, g=EI*sin_match, osc1="cos", osc2="sin", K=K) - expr_integral_form_2(f=EI*cos_match, g=EI*sin_match, osc1="sin", osc2="sin", K=K)),
            -ppp*(expr_integral_form_1(f=EI, g=EQ*cos_match, osc="cos", K=K) - expr_integral_form_1(f=EI, g=EQ*sin_match, osc="sin", K=K)),
            -ppp*(expr_integral_form_1(f=EQ, g=EI*cos_match, osc="cos", K=K) - expr_integral_form_1(f=EQ, g=EI*sin_match, osc="sin", K=K)),
            ppp*((expr_integral_form_2(f=EQ*cos_match, g=EI*cos_match, osc1="cos", osc2="cos", K=K) - expr_integral_form_2(f=EQ*sin_match, g=EI*cos_match, osc1="sin", osc2="cos", K=K) - expr_integral_form_2(f=EQ*cos_match, g=EI*sin_match, osc1="cos", osc2="sin", K=K) + expr_integral_form_2(f=EQ*sin_match, g=EI*sin_match, osc1="sin", osc2="sin", K=K)) - (expr_integral_form_2(f=EI*cos_match, g=EQ*cos_match, osc1="cos", osc2="cos", K=K) - expr_integral_form_2(f=EI*sin_match, g=EQ*cos_match, osc1="sin", osc2="cos", K=K) - expr_integral_form_2(f=EI*cos_match, g=EQ*sin_match, osc1="cos", osc2="sin", K=K) + expr_integral_form_2(f=EI*sin_match, g=EQ*sin_match, osc1="sin", osc2="sin", K=K))),
            -ppp**2*(expr_integral_form_1(f=EQ, g=EQ*cos_match, osc="sin", K=K) + expr_integral_form_1(f=EQ, g=EQ*cos_match, osc="sin", K=K)),
            ppp**2*((expr_integral_form_2(f=EQ*cos_match, g=EQ*cos_match, osc1="cos", osc2="sin", K=K) - expr_integral_form_2(f=EQ*sin_match, g=EQ*cos_match, osc1="sin", osc2="sin", K=K) - expr_integral_form_2(f=EQ*sin_match, g=EQ*sin_match, osc1="sin", osc2="cos", K=K) + expr_integral_form_2(f=EQ*cos_match, g=EQ*sin_match, osc1="cos", osc2="cos", K=K)) - (expr_integral_form_2(f=EQ*cos_match, g=EQ*cos_match, osc1="sin", osc2="cos", K=K) - expr_integral_form_2(f=EQ*cos_match, g=EQ*sin_match, osc1="sin", osc2="sin", K=K) - expr_integral_form_2(f=EQ*sin_match, g=EQ*sin_match, osc1="cos", osc2="sin", K=K) + expr_integral_form_2(f=EQ*sin_match, g=EQ*cos_match, osc1="cos", osc2="cos", K=K))),
            ppp*((expr_integral_form_2(f=EQ*cos_match, g=EI*cos_match, osc1="sin", osc2="sin", K=K) + expr_integral_form_2(f=EQ*sin_match, g=EI*cos_match, osc1="cos", osc2="sin", K=K) + expr_integral_form_2(f=EQ*cos_match, g=EI*sin_match, osc1="sin", osc2="cos", K=K) + expr_integral_form_2(f=EQ*sin_match, g=EI*sin_match, osc1="cos", osc2="cos", K=K)) - (expr_integral_form_2(f=EI*cos_match, g=EQ*cos_match, osc1="sin", osc2="sin", K=K) + expr_integral_form_2(f=EI*sin_match, g=EQ*cos_match, osc1="cos", osc2="sin", K=K) + expr_integral_form_2(f=EI*cos_match, g=EQ*sin_match, osc1="sin", osc2="cos", K=K) + expr_integral_form_2(f=EI*sin_match, g=EQ*sin_match, osc1="cos", osc2="cos", K=K)))
        ]

    integrals = [expr/4 for expr in integrals]
    integrals = [sympy.utilities.lambdify([wd, wd_tilde, tg, beta, ppp, b_min, b_plus], expr) for expr in integrals]

    return integrals


def calculate_theoretical_detuning(wd: float,
                                   tg: float,
                                   ppp: float,
                                   V0: float,
                                   beta: float = 0,
                                   t0: float = 0,
                                   sympy_integrals: list | None = None):
    if sympy_integrals is None:
        raise ValueError("Please provide the sympy integrals")
    
    detuning = 0
    tc = np.pi/wd

    # Define time constants
    Nc = np.floor((tg-t0) / tc).astype(int)
    t1 = t0
    t2 = Nc*tc+t0

    # Calculate lambda_hat
    lambda_hat = ppp * np.pi/tg
    for i in range(9):
        detuning += sympy_integrals[i](wd=wd,
                                       tg=tg,
                                       beta=beta,
                                       ppp=lambda_hat,
                                       b_min=t1,
                                       b_plus=t2)
            
    detuning *= 2 * V0**2/tg
    return detuning


def numerical_integration(fun,
                          t1: float,
                          t2: float,
                          *args,
                          **kwargs):
    """Wrapper for numerical integration for double integrals"""
    f = lambda y,x: fun(t1=x,t2=y,*args,**kwargs)
    val,err = scipy.integrate.dblquad(f, t1, t2, t1, lambda x: x, epsabs=1e-15, epsrel=1e-15)
    return val


# List of all numerical integrals
integrals_numerical_det = [
    integral1_numerical,
    integral2_numerical,
    integral3_numerical,
    integral4_numerical,
    integral5_numerical,
    integral6_numerical,
    integral7_numerical,
    integral8_numerical,
    integral9_numerical,
]


def calculate_theoretical_pulse_parameters_comm(w01: float,
                                                tg: float,
                                                sympy_integrals: list | None = None,
                                                beta: float = 0,
                                                V0_frac: float = 1,
                                                step_size: tuple[float] = (0.05, 0.05),
                                                Z: int = 1000,
                                                method: list[str] = "numerical",
                                                sympy_det_integrals: list | None = None):
    """
    Calculates theoretical pulse parameters in the first-order 
    Magnus approximation using the "commensurate" technique outlined
    in appendix C.

    Parameters
    ----------
    w01 : float
        Drive frequency
    tg : float
        Gate duration
    sympy_integrals : list
        List of symbolic integrals computed using `analytical_comm_sympy_functions`
    beta : float
        Dimensionless parameter `beta`
    V0_frac : float
        Fraction of desired rotation angle w.r.t. a pi pulse
    step_size : tuple[float]
        Step sizes for fixed point iteration
    Z : int
        Maximum number of iterations for fixed point iteration
    method : list[str]
        Refers to the integration method of the detuning integrals. Valid choices 
        are `numerical` or `analytical`. These can be solved analytically (symbolically), but 
        that can take very long. Additionally, Sympy seems to be a bit buggy for very long 
        expressions, or maybe it just eats-up lot of memory. If `comm`, `sympy_det_integrals`
        should be specified.
    """
    # Load sympy integrals from file if not specified
    if sympy_integrals is None:
        with open("sympy-integrals/sympy-integrals-1.pickle", "rb") as f:
            sympy_integrals = pickle.load(f)

    if sympy_det_integrals is None:
        with open("sympy-integrals/sympy-integrals-3.pickle", "rb") as f:
            sympy_det_integrals = pickle.load(f)

    # Define pulse envelope
    pulse_envelope = CosinePulseEnvelope(tg=tg)

    # Initial params
    detuning = 0
    ppp_theory = 1/4/w01
    V0_theory = V0_frac

    # Init params
    sidx = 0
    test_xx = np.array([])
    test_yy = np.array([])

    # Specify start/end of integration windows
    t0 = 0
    t1 = tg
    
    for z in range(Z):
        # Calculate params
        _wd = w01 - detuning
        tc = np.pi/_wd

        # Number of Magnus periods rounded
        N_hat = np.round(tg/tc).astype(int)

        wd_hat = N_hat * np.pi/tg
        wd_tilde = _wd - wd_hat

        # PPP and drive strength
        params = {
            'wd': wd_hat,
            'wd_tilde': wd_tilde,
            'tg': tg,
            'beta': beta + wd_tilde*tg,
            'phi_tilde': -wd_tilde*tg/2,
            'b_min': t0,
            'b_plus': t1,
            'ppp': 1 # This looks wrong but it's not
        }
        integrals = np.array([k(**params) for k in sympy_integrals])
        integrals[6:] *= 2*detuning

        V0_theory = V0_frac * integrals[0] / (sum(integrals[:2]) + ppp_theory*integrals[2] + ppp_theory*integrals[6] + ppp_theory*integrals[7] + integrals[8])
        denum = (integrals[3]+integrals[4]+integrals[11])
        if np.isclose(denum,0):
            ppp_theory_new = 1/2/_wd
        else:
            ppp_theory_new = (-integrals[5]-integrals[9]-integrals[10])/denum

        # Detuning
        if method=="numerical":
            params_det = dict(
                t1=t0,
                t2=t1,
                pulse_envelope=pulse_envelope,
                wd=_wd,
                phi=beta/2 + wd_tilde*tg/2,
                lambda_hat=ppp_theory*np.pi/tg
            )
            detuning_new = sum([numerical_integration(fun=integrals_numerical_det[i], **params_det) for i in range(9)])
        else:
            params_det = dict(
                b_min=t0,
                b_plus=t1,
                wd=wd_hat,
                wd_tilde=wd_tilde,
                beta=beta + wd_tilde*tg,
                ppp=ppp_theory*np.pi/tg,
                tg=tg
            )
            detuning_new = sum([sympy_det_integrals[i](**params_det) for i in range(9)])
        detuning_new *= 2 * (pulse_envelope.V0*V0_theory)**2 / (t1-t0)

        # Break if converged
        if np.abs(detuning_new - detuning)*1e-6/2/np.pi < 0.0001 and \
            np.abs(ppp_theory - ppp_theory_new)*2*w01 < 0.0001 and \
            z!=sidx:
            break
        
        # Append params to arrays
        test_xx = np.append(test_xx, ppp_theory)
        test_yy = np.append(test_yy, detuning)

        # Calculate new detuning and ppp
        detuning = detuning + step_size[0] * (detuning_new - detuning)
        ppp_theory = ppp_theory + step_size[1] * (ppp_theory_new - ppp_theory)

        # Define points
        points = test_xx*2*w01 + 1j*test_yy*1e-6/2/np.pi

        # Here, there is some trickery to improve the speed at which 
        # fixed point iteration converges. If you don't do this trickery,
        # the, the "optimizer" will spiral to the optimum. This will converge at
        # some point as long as the step size is small enough. This spiralling
        # technically happens in 3D, but here we will only use \lambda and \Delta.
        # The trickery here jumps a little bit towards its best guess of the center
        # of the spiral every time the spiral completes a 90 degree turn.
        if z>=sidx + 3 and \
            np.abs(np.angle(points[sidx+1]-points[sidx+0]) - np.angle(points[-1]-points[-2])) > np.pi/2:
            
            # Obtain relevant points and vectors
            points_x = np.append(test_xx, ppp_theory)*2*w01
            points_y = np.append(test_yy, detuning)*1e-6/2/np.pi
            points = points_x + 1j*points_y
            v3 = points[-1]-points[-3]

            # Calculate angle of vector to new point 
            # Can be off by 180 degrees, this is fixed later
            _ang = np.angle(v3) + np.pi/2

            _diffs = points[(sidx+1):] - points[sidx:-1]
            mean_r = np.mean(np.abs(_diffs.real))
            mean_i = np.mean(np.abs(_diffs.imag))
            _abs = np.abs(np.cos(_ang))*mean_r + np.abs(np.sin(_ang))*mean_i

            m = points[-3] + v3/2
            n = points[-2] + _abs*np.exp(1j*_ang)

            # Make sure we rotate the correct way
            if np.abs(n - points[-2]) < np.abs(n - m):
                _ang += np.pi
                n = points[-2] + _abs*np.exp(1j*_ang)

            # Calculate new point
            detuning = n.imag*2*np.pi*1e6
            ppp_theory = n.real/2/w01
            # Update idx of last "jumped" point
            sidx = z+1

    return V0_theory, ppp_theory, detuning


def calculate_theoretical_pulse_parameters(w01: float,
                                           tg: float,
                                           sympy_integrals: list | None = None,
                                           beta: float = 0,
                                           V0_frac: float = 1,
                                           step_size: tuple[float] = (0.05, 0.05),
                                           Z: int = 1000,
                                           method: list[str] = "numerical",
                                           sympy_det_integrals: list | None = None):
    """
    Calculates theoretical pulse parameters in the first-order 
    Magnus approximation without using the "commensurate" technique outlined
    in appendix C.

    Parameters
    ----------
    w01 : float
        Drive frequency
    tg : float
        Gate duration
    sympy_integrals : list
        List of symbolic integrals computed using `analytical_comm_sympy_functions`
    beta : float
        Dimensionless parameter `beta`
    V0_frac : float
        Fraction of desired rotation angle w.r.t. a pi pulse
    step_size : tuple[float]
        Step sizes for fixed point iteration
    Z : int
        Maximum number of iterations for fixed point iteration
    method : list[str]
        Refers to the integration method of the detuning integrals. Valid choices 
        are `numerical` or `analytical`. These can be solved analytically (symbolically), but 
        that can take very long. Additionally, Sympy seems to be a bit buggy for very long 
        expressions, or maybe it just eats-up lot of memory. If `comm`, `sympy_det_integrals`
        should be specified.
    """
    # Load sympy integrals from file if not specified
    if sympy_integrals is None:
        with open("sympy-integrals/sympy-integrals-1.pickle", "rb") as f:
            sympy_integrals = pickle.load(f)

    if sympy_det_integrals is None:
        with open("sympy-integrals/sympy-integrals-2.pickle", "rb") as f:
            sympy_det_integrals = pickle.load(f)

    # Define pulse envelope
    pulse_envelope = CosinePulseEnvelope(tg=tg)

    # Initial params
    detuning = 0
    ppp_theory = 1/4/w01
    V0_theory = V0_frac

    # Init params
    sidx = 0
    test_xx = np.array([])
    test_yy = np.array([])

    # Specify start/end of integration windows
    t0 = 0
    t1 = tg
    
    for z in range(Z):
        # Calculate params
        _wd = w01 - detuning
        tc = np.pi/_wd

        # Number of Magnus periods
        Nc = np.floor(tg / tc).astype(int)
        t0 = 0
        t1 = t0 + Nc*tc

        # PPP and drive strength
        params = {
            'wd': _wd,
            'wd_tilde': 0,
            'tg': tg,
            'beta': beta,
            'phi_tilde': 0,
            'b_min': t0,
            'b_plus': t1,
            'ppp': 1 # This looks wrong but it's not
        }
        integrals = np.array([k(**params) for k in sympy_integrals])
        integrals[6:] *= 2*detuning

        V0_theory = V0_frac * integrals[0] / (sum(integrals[:2]) + ppp_theory*integrals[2] + ppp_theory*integrals[6] + ppp_theory*integrals[7] + integrals[8])
        denum = (integrals[3]+integrals[4]+integrals[11])
        if np.isclose(denum,0):
            ppp_theory_new = 1/2/_wd
        else:
            ppp_theory_new = (-integrals[5]-integrals[9]-integrals[10])/denum

        # Detuning
        if method=="numerical":
            params_det = dict(
                t1=t0,
                t2=t1,
                pulse_envelope=pulse_envelope,
                wd=_wd,
                phi=(beta-2*_wd*t0)/2,
                lambda_hat=ppp_theory*np.pi/tg
            )
            detuning_new = sum([numerical_integration(fun=integrals_numerical_det[i], **params_det) for i in range(9)])
        else:
            params_det = dict(
                b_min=t0,
                b_plus=t1,
                wd=_wd,
                beta=beta,
                ppp=ppp_theory*np.pi/tg,
                tg=tg
            )
            detuning_new = sum([sympy_det_integrals[i](**params_det) for i in range(9)])
        detuning_new *= 2 * (pulse_envelope.V0*V0_theory)**2 / (t1-t0)

        # Break if converged
        if np.abs(detuning_new - detuning)*1e-6/2/np.pi < 0.0001 and \
            np.abs(ppp_theory - ppp_theory_new)*2*w01 < 0.0001 and \
            z!=sidx:
            break
        
        # Append params to arrays
        test_xx = np.append(test_xx, ppp_theory)
        test_yy = np.append(test_yy, detuning)

        # Calculate new detuning and ppp
        detuning = detuning + step_size[0] * (detuning_new - detuning)
        ppp_theory = ppp_theory + step_size[1] * (ppp_theory_new - ppp_theory)

        # Define points
        points = test_xx*2*w01 + 1j*test_yy*1e-6/2/np.pi

        # Here, there is some trickery to improve the speed at which 
        # fixed point iteration converges. If you don't do this trickery,
        # the, the "optimizer" will spiral to the optimum. This will converge at
        # some point as long as the step size is small enough. This spiralling
        # technically happens in 3D, but here we will only use \lambda and \Delta.
        # The trickery here jumps a little bit towards its best guess of the center
        # of the spiral every time the spiral completes a 90 degree turn.
        if z>=sidx + 3 and \
            np.abs(np.angle(points[sidx+1]-points[sidx+0]) - np.angle(points[-1]-points[-2])) > np.pi/2:
            
            # Obtain relevant points and vectors
            points_x = np.append(test_xx, ppp_theory)*2*w01
            points_y = np.append(test_yy, detuning)*1e-6/2/np.pi
            points = points_x + 1j*points_y
            v3 = points[-1]-points[-3]

            # Calculate angle of vector to new point 
            # Can be off by 180 degrees, this is fixed later
            _ang = np.angle(v3) + np.pi/2

            _diffs = points[(sidx+1):] - points[sidx:-1]
            mean_r = np.mean(np.abs(_diffs.real))
            mean_i = np.mean(np.abs(_diffs.imag))
            _abs = np.abs(np.cos(_ang))*mean_r + np.abs(np.sin(_ang))*mean_i

            m = points[-3] + v3/2
            n = points[-2] + _abs*np.exp(1j*_ang)

            # Make sure we rotate the correct way
            if np.abs(n - points[-2]) < np.abs(n - m):
                _ang += np.pi
                n = points[-2] + _abs*np.exp(1j*_ang)

            # Calculate new point
            detuning = n.imag*2*np.pi*1e6
            ppp_theory = n.real/2/w01
            # Update idx of last "jumped" point
            sidx = z+1

    return V0_theory, ppp_theory, detuning
