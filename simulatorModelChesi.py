import numpy as np
from math import cos, sin, tan

from scipy.integrate import odeint
class SimulatorModel_Chesi():
    """ Based on:
        Chesi, Simone, et al. 
        "Automatic mass balancing of a spacecraft three-axis simulator: Analysis and experimentation."
        Journal of Guidance, Control, and Dynamics 37.1 (2014): 197-206.
    """


    def __init__(self, J0, m, g, R_true):
        # inertia matrix
        self.J0 = J0 
        # mass of sat + sat sim
        self.m = m
        # earth acceleration
        self.g = g
        # true offset CoR to CoM offset vector
        self.R_true = R_true
        # g in body frame
        self.gb = np.zeros((3,))
        

    def calculate_omegadot(self, omega, J, Rcm, gb):
        """ Chesi, Eq. (2)
        """
        m = self.m
        
        cross_product1 = np.cross(omega, np.dot(J, omega))[0]
        cross_product2 = np.cross(Rcm, gb * m)

        return np.linalg.solve(J, -cross_product1 + cross_product2)

    def eom(self, y:np.array, t):
        """ Equations of Motion to update system state/dynamics
        """

        ydot = np.zeros_like(y)
            
        # get Rcm estimation from last iteration
        Rcm = np.array([y[6], y[7], y[8]])
        m = self.m
        g = self.g

        # Eq (4) adapted with m_p = m and no mmus
        J_aug = np.matrix([[m*(Rcm[1]**2 + Rcm[2]**2), -m*Rcm[0]*Rcm[1], -m*Rcm[0]*Rcm[2]],
                        [-m*Rcm[0]*Rcm[1], m*(Rcm[0]**2 + Rcm[2]**2), -m*Rcm[1]*Rcm[2]],
                        [-m*Rcm[0]*Rcm[2], -m*Rcm[1]*Rcm[2], m*(Rcm[0]**2 + Rcm[1]**2)]
                        ])

        J = self.J0 + J_aug

        omega = np.array([y[3], y[4], y[5]])

        sphi = sin(y[0])
        cphi = cos(y[0])
        sth = sin(y[1])
        cth = cos(y[1])
        tth = tan(y[1])

        # Euler angles dot
        eulerdot = np.matrix([[1, tth*sphi, tth*cphi],
                            [0, cphi, -sphi],
                            [0, sphi/cth, cphi/cth]]).dot(omega)
        
        eulerdot = np.ndarray.flatten(eulerdot)

        # Gravity vector in body frame
        gb = np.array([g*sth, -g*sphi*cth, -g*cphi*cth])
        self.gb = gb

        # Dynamics equation
        omegadot = self.calculate_omegadot(omega=omega, J=J, Rcm=Rcm, gb=gb)

        ydot[0:3] = eulerdot

        ydot[3:6] = omegadot

        # ydot[6:8] will not be updated as Rcm is considered static

        return ydot

    def step(self, y, dt):
            ydot = odeint(self.eom, y, [0, dt])

            y = ydot[-1, :]

            return y
    