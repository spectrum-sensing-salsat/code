import numpy as np
from scipy import constants


class DopplerSignal():
    def __init__(self, t_sec, f_sample=1e6, f_carrier=450e6, steps=1e3):

        self.t_sec = t_sec
        self.f_sample = f_sample
        self.f_carrier = f_carrier
        self.steps = steps

        # prepare time array that starts at 0 and goes to whatever time you want (equally spaced)
        self.t = np.linspace(0, int(self.t_sec), self.steps, endpoint=False)

    def get(self, f_center=0., power=0., distance=1000.0, velocity=7660.0):
        '''
        distance : orthogonal distance of observer and object path in meters
        velocity : velocity of object on its path in m/s

        'object'   : thing that creates the signal (moving)
        'observer' : we / thing that receives the signal (stationary)
        '''

        # calculate relative velocity and doppler shift
        v, d, max_d = self.relative_speed(distance, velocity)
        freq = v / constants.speed_of_light * self.f_carrier
        powr = -np.abs(d / max_d)**3

        # upscale to 'self.t * self.f_sample' resolution
        t = np.linspace(0,
                        int(self.t_sec),
                        int(self.t_sec * self.f_sample),
                        endpoint=False)
        freq = np.repeat(freq, int(self.t_sec * self.f_sample / self.steps))
        powr = np.repeat(powr, int(self.t_sec * self.f_sample / self.steps))

        # add frequency offset in baseband
        freq += f_center

        # convert normalized power (0 to 1) to dB
        powr = powr * 100. + power

        return t, freq, powr

    def relative_speed(self, distance, velocity):

        # we will assume that the object will be closest to the oberserver at half the time
        max_d = self.t[-1] * .5 * velocity

        # calculate objects position on its path at every time step
        x = max_d - self.t * velocity

        # calculate relative angle of observer and object at every time step
        # we will assume that the object is moving from right to left above the observer
        a = np.arctan2(x, distance)

        # calculate realtive velocity of object
        v = a / np.pi * 2. * velocity

        # calculate distance between object and observer
        d = np.sqrt(distance**2. + x**2.)

        return v, d, max_d
