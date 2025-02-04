from typing import Union
import numpy as np


class PulseEnvelope:
    pass


class CosinePulseEnvelope(PulseEnvelope):
    def __init__(self, tg) -> None:
        self.tg = tg

        # Calculate the amplitude required for a Pi pulse
        self.V0 = 2*np.pi/tg
    
    def envelope(self, t, *args) -> Union[np.ndarray, float]:
        return 1/2 * (1 - np.cos(2*np.pi*t/self.tg))

    def derivative(self, t) -> Union[np.ndarray, float]:
        return np.sin(2*np.pi*t/self.tg)
    
    def envelope_square_int(self, t) -> Union[np.ndarray, float]:
        """Returns int_0^t (envelope^2)"""
        return (
            (self.tg * (-8*np.sin(2*np.pi*t/self.tg) + np.sin(4*np.pi*t/self.tg)) + 12*np.pi*t) / (32*np.pi)
        )
    
    def derivative_square_int(self, t) -> Union[np.ndarray, float]:
        """Returns int_0^t (derivative^2)"""
        return (
            (-self.tg * np.sin(4*np.pi*t/self.tg) + 4*np.pi*t) / (8*np.pi)
        )
    