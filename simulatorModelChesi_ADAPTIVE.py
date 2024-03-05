from math import sin, cos, tan

from scipy.integrate import odeint

import numpy as np
import matplotlib.pyplot as plt

class SimulatorModel_Chesi_ADAPTIVE():
    """ Based on:
            Chesi, Simone, et al. 
            "Automatic mass balancing of a spacecraft three-axis simulator: Analysis and experimentation."
            Journal of Guidance, Control, and Dynamics 37.1 (2014): 197-206.
        and adaptions from
            Silva, Rodrigo Cardoso da. 
            "Filtering and adaptive control for balancing a nanosatellite testbed." (2018).
    """

    def __init__(self, kp, dt, v_mmu, m_mmu, m, J0, g, tf, Rcm, Rcm_guess, mmu_min, mmu_max):
        self.kp = kp
        self.dt = dt
        self.tf = tf

        self.v_mmu = v_mmu
        self.m_mmu = m_mmu
        self.mmu_min = mmu_min
        self.mmu_max = mmu_max

        self.m = m
        self.J0 = J0
        self.Rcm = Rcm
        self.Rcm_guess = Rcm_guess
        self.g = g

    def eom_chesi(self, y, t):
        """ The equations of motion used for simulating the satellite simulator in the transversal approximation phase.
        """
        # y = [q1, q2, q3, q4, wx, wy, wz, rx, ry, rz, rx_est, ry_est, rz_est, r_mmux, r_mmuy, r_mmuz]

        ydot = np.zeros(16)

        q = y[0:4]
        omega = y[4:7]
        Rcm = y[7:10]
        r_est = y[10:13]
        MMUx = np.array([y[13], -0.22, 0])
        MMUy = np.array([0.22, y[14], 0])
        MMUz = np.array([-0.22, 0, y[15]])

        # Testbed characteristics
        J0 = self.J0

        m = self.m  # mass of the testbed
        m_mmu = self.m_mmu  # mass of movable parts of an MMU
        g = self.g  # in m/s^2, local gravity

        # Chesi, Eq (4) I/II
        Jaug = np.array([[m*(Rcm[1]**2 + Rcm[2]**2), -m*Rcm[0]*Rcm[1], -m*Rcm[0]*Rcm[2]],
                           [-m*Rcm[0]*Rcm[1], m * (Rcm[0]**2 + Rcm[2]**2), -m*Rcm[1]*Rcm[2]],
                           [-m*Rcm[0]*Rcm[2], -m*Rcm[1]*Rcm[2],m*(Rcm[0]**2 + Rcm[1]**2)]
                           ])

        J = J0 + Jaug

        # Update of inertia tensor
        MMUxCross = np.array([[0, -MMUx[2], MMUx[1]],
                            [MMUx[2], 0, -MMUx[0]],
                            [-MMUx[1], MMUx[0], 0]])

        MMUyCross = np.array([[0, -MMUy[2], MMUy[1]],
                            [MMUy[2], 0, -MMUy[0]],
                            [-MMUy[1], MMUy[0], 0]])

        MMUzCross = np.array([[0, -MMUz[2], MMUz[1]],
                            [MMUz[2], 0, -MMUz[0]],
                            [-MMUz[1], MMUz[0], 0]])

        # Chesi, Eq (4) II/II
        J = J - m_mmu * (np.dot(MMUxCross, MMUxCross) + np.dot(MMUyCross, MMUyCross) + np.dot(MMUzCross, MMUzCross))

        ## Model equations
        ## Kinematic equation
        # Quaternion rates
        # DaSilva, Eq (4.117)
        qdot = 0.5 * np.dot([[0, -omega[0], -omega[1], -omega[2]],
                            [omega[0], 0, omega[2], -omega[1]],
                            [omega[1], -omega[2], 0, omega[0]],
                            [omega[2], omega[1], -omega[0], 0]], q)

        # Gravity vector in body-frame
        # DaSilva, Eq (4.116)
        Rbi = np.array([[2 * q[0] ** 2 - 1 + 2 * q[1] ** 2, 2 * q[1] * q[2] - 2 * q[0] * q[3], 2 * q[1] * q[3] + 2 * q[0] * q[2]],
                        [2 * q[1] * q[2] + 2 * q[0] * q[3], 2 * q[0] ** 2 - 1 +
                            2 * q[2] ** 2, 2 * q[2] * q[3] - 2 * q[0] * q[1]],
                        [2 * q[1] * q[3] - 2 * q[0] * q[2], 2 * q[2] * q[3] + 2 * q[0] * q[1], 2 * q[0] ** 2 - 1 + 2 * q[3] ** 2]])

        gb = np.dot(Rbi.T, np.array([0, 0, -g]))

        # Projection operator
        # # DaSilva, Eq (4.124)
        P = np.eye(3) - np.outer(gb, gb) / np.linalg.norm(gb) ** 2

        # DaSilva, Eq (4.123)
        omegap = np.dot(P, omega)

        # DaSilva, Eq (4.113)
        gbcross = np.array([[0, -gb[2], gb[1]],
                            [gb[2], 0, -gb[0]],
                            [-gb[1], gb[0], 0]])

        # DaSilva, Eq (4.121)
        Phi = -m * gbcross

        # DaSilva, Eq (4.141)
        control_torque = -np.dot(Phi, r_est) - self.kp * omegap

        # DaSilva, Eq (4.143)
        r_mmu = np.dot(gbcross, control_torque) / (np.linalg.norm(gb) ** 2 * m_mmu)
        # Limits the mmus positions to [self.mmu_min, self.mmu_max]
        np.clip(r_mmu, self.mmu_min, self.mmu_max)

        # DaSilva, Eq (4.142)
        control_torque = m_mmu * np.cross(r_mmu, gb)

        # Adaptive control law for the estimated unbalance vector
        # DaSilva, Eq (4.133)
        r_est_dot = np.dot(Phi.T, omega)

        # Dynamics equation
        # DaSilva, Eq (4.146)
        omegadot = np.linalg.solve(J, (-np.cross(omega, np.dot(J, omega)) + np.cross(Rcm, gb * m) + control_torque))

        # Derivatives (xdot)
        ydot[0:4] = qdot
        ydot[4:7] = omegadot
        ydot[7:10] = 0
        ydot[10:13] = r_est_dot
        ydot[13:16] = 0

        return ydot

    def transversalCorrection(self):
        """ First phase of the adaptive control.
            Estimation and correction of the x-y offset vector using the law of conversion of angular momentum and 
            the equation of motions of the testbed.
        """

        tf = 50.0  # simulation length
        dt = self.dt  # simulation step size

        Rcm = self.Rcm  # CM offset vector
        m = self.m  # mass of the testbed, in kg
        m_mmu = self.m_mmu  # mass of one Movable Mass Unit (MMU), in kg
        g = self.g  # in m/s^2, local gravity

        # Initialize lists for later plotting
        tArray = [0.0]
        omegaArray = [[0.0, 0.0, 0.0]]
        r_mmusArray = [[0.0, 0.0, 0.0]]
        ctrltorqueArray = [[0.0, 0.0, 0.0]]
        r_estArray = [self.Rcm_guess]

        # Initialization of model initial conditions
        # y = [q1, q2, q3, q4, wx, wy, wz, rx, ry, rz, rx_est, ry_est, rz_est, r_mmux, r_mmuy, r_mmuz]
        y = np.hstack([[1, 0, 0, 0], [0, 0, 0], Rcm, np.zeros(3), np.zeros(3)])

        # step factor is to model the time it takes the mmus to move, approximation only
        step_factor = 6
        for t in np.arange(dt, tf + dt, dt*step_factor):

            # Simulation of the system model
            y_dt = odeint(self.eom_chesi, y, [0, dt])[1]
            y = y_dt

            # MMUs operation
            omega = y[4:7]
            q = y[0:4]

            # DaSilva, Eq (4.116)
            Rbi = np.array([[2 * q[0] ** 2 - 1 + 2 * q[1] ** 2, 2 * q[1] * q[2] - 2 * q[0] * q[3], 2 * q[1] * q[3] + 2 * q[0] * q[2]],
                            [2 * q[1] * q[2] + 2 * q[0] * q[3], 2 * q[0] ** 2 -
                                1 + 2 * q[2] ** 2, 2 * q[2] * q[3] - 2 * q[0] * q[1]],
                            [2 * q[1] * q[3] - 2 * q[0] * q[2], 2 * q[2] * q[3] + 2 * q[0] * q[1], 2 * q[0] ** 2 - 1 + 2 * q[3] ** 2]])
            gb = np.dot(Rbi.T, np.array([0, 0, -g]))
            # DaSilva, Eq (4.133)
            gbcross = np.array([[0, -gb[2], gb[1]],
                                [gb[2], 0, -gb[0]],
                                [-gb[1], gb[0], 0]])
            
            r_est = y[10:13]

            if t == 0.05:
                # Approximation for first timestep would be set directly to 0 otherwise 
                # which would make the estimate converge "too fast" as the true values are very close to 0 
                r_est = self.Rcm_guess

            # Same as in eom_chesi
            Phi = -m * gbcross
            P = np.eye(3) - np.outer(gb, gb) / np.linalg.norm(gb) ** 2
            omegap = np.dot(P, omega)
            control_torque = -np.dot(Phi, r_est) - self.kp * omegap

            # Determine the new r_mmus vector. Dynamics of the MMUs are not considered,
            # which means r_mmus changes instantaneously
            # DaSilva, Eq (4.143)
            r_mmus = np.dot(gbcross, control_torque) / (np.linalg.norm(gb) ** 2 * m_mmu)

            # Limits the mmus positions to [self.mmu_min, self.mmu_max], which are the 
            # physical limitations
            r_mmus = np.clip(r_mmus, self.mmu_min, self.mmu_max)

            y[13:16] = r_mmus

            # Save data for plotting.
            tArray.append(t)
            omegaArray.append(omega)
            r_mmusArray.append(r_mmus)
            r_estArray.append(r_est)
            ctrltorqueArray.append(control_torque)

        return [tArray, omegaArray, r_mmusArray, r_estArray, ctrltorqueArray]
    
    def eom_ukf(self, y, dt):
        """ The equations of motion that are used to simulate the system for the z component offset vector estimation 
            in the vertical correction phase. Different to eom_chesi as the state vector is now 
            y = [phi, theta, psi, w_x, w_y, w_z, rz] but same to eom in simulatorModelChesi.py.
        """
        
        ydot = np.zeros(7)

        Rcm = np.array((0.0, 0.0, y[6]))
        J0 = np.diag(np.diag(self.J0))
        m = self.m
        g = self.g

        Jaug = np.array([[m*(Rcm[1]**2 + Rcm[2]**2), -m*Rcm[0]*Rcm[1], -m*Rcm[0]*Rcm[2]],
                           [-m*Rcm[0]*Rcm[1], m * (Rcm[0]**2 + Rcm[2]**2), -m*Rcm[1]*Rcm[2]],
                           [-m*Rcm[0]*Rcm[2], -m*Rcm[1]*Rcm[2],m*(Rcm[0]**2 + Rcm[1]**2)]
                           ])

        J = J0 + Jaug

        omega = np.array(([y[3], y[4], y[5]]))

        sphi = sin(y[0])
        cphi = cos(y[0])
        sth = sin(y[1])
        cth = cos(y[1])
        tth = tan(y[1])

        # Euler angles dot
        eulerdot = np.matrix([[1, tth*sphi, tth*cphi],
                            [0, cphi, -sphi],
                            [0, sphi/cth, cphi/cth]]).dot(omega)

        # omegadot = self.calculate_omegadot(omega=omega, J=J, y=y, sphi=sphi, sth=sth, cth=cth)
        omegadot = np.dot(np.linalg.inv(J), -np.cross(omega, np.dot(J, omega)) + [g * sphi * cth * m * y[6], g * sth * m * y[6], 0])
        ydot[0:3] = eulerdot[0]
        ydot[3:6] = omegadot
        ydot[6] = 0.0

        return ydot

    def calculate_omegadot(self, omega, J, y, sphi, sth, cth, ):
        """ Same as in simulatorModelChesi.py.
        """
        m = self.m
        g = self.g

        vec = np.array(([g*sphi*cth*m*y[6], g*sth*m*y[6], 0]))
        cross_product = np.cross(omega, np.dot(J, omega))[0] + vec


        omegadot = np.linalg.solve(J, -cross_product + vec)

        return omegadot

    def verticalCorrection(self):
        """ Second phase of the adaptive control.
            Estimation and correction of the z component of the offset vector using an UKF.
            Taken from DaSilva, Eq (4.163 - 4.168)
        """            
        # State vector x = [w_x, w_y, w_z, r_z]

        # In the UKF only the diagonal elements of inertia matrix are used.
        J = np.diag(np.diag(self.J0))

        # Number of states
        n = 4

        m = self.m
        # only z component used
        Rcm = np.array((0, 0, self.Rcm[2]))
        g = self.g

        # Filter matrices
        R = np.array([[0.005**2, 0.0, 0.0], [0.0, 0.005**2, 0.0], [0.0, 0.0, 0.005**2]])
        H = np.hstack((np.eye(3), np.zeros((3,1))))
        
        # Initial state
        xstate = np.array(([0, 0, 0, self.Rcm_guess[2]]))

        # Initial UKF state estimate, 2*Rcm[2] as initial guess
        xhatukf = np.array(2*xstate)

        z = np.dot(H, xstate) + np.sqrt(R).dot(np.array([np.random.random(), np.random.random(), np.random.random()]))
        # UKF weights
        W = np.ones((2*n, 1))/(2*n)

        Q = np.diag((0.00005, 0.00005, 0.00005, (0.1*Rcm[2])**2))

        # pi/6 so that we dont have roll/pitch as 0 deg/rad which would get a singularity
        y = np.array(([np.pi/6, np.pi/6, 0, 0, 0, 0, Rcm[2]]))
        
        P = 0.00001*np.eye(4) # scale to mm
        Pukf = P

        # For later plotting
        yArray = list(y)
        zArray = list(z)
        tArray = []
        zhatArray = list(z)
        xArray = list(xstate)
        xhatukfArray = list()
        PArray = list(P)
        PukfArray = list(Pukf)
        attitudeArray = list()
        angular_velArray = list()
        z_estimate = list()

        for t in np.arange(self.dt, self.tf + self.dt, self.dt):
            ## Simulation of timestep of the system model
            y_dt = odeint(self.eom_ukf, y, [0, self.dt])[1]
            
            #y = [phi, theta, psi, w_x, w_y, w_z, r_z]
            y = y_dt

            attitudeArray.append(y[0:3])
            angular_velArray.append(y[3:6])

            xstate[:4] = np.transpose(y[3:7]) + np.sqrt(Q).dot(np.random.rand(4))

            # Simulating sensor with noise by adding random noise
            z = np.dot(H, xstate) + np.sqrt(R).dot(np.random.rand(3))

            ## 0. UKF outline
            ## x(k+1) = f[x(k), u(k), t(k))] + w(k)
            ## y(k) = h[x(k), t(k)] + v(k)

            ## 1. UKF initialization
            root = np.linalg.cholesky(n*Pukf)

            # Initialize sigma as a 4 x (2*n) matrix
            sigma = np.zeros((4, 2*n))

            # Loop to generate columns of sigma
            for i in range(n):
                sigma[:, i] = xhatukf + root[i, :]
                sigma[:, i+n] = xhatukf - root[i, :]

            # 4x8
            xbreve = sigma

            ## 2. Transform the sigma points into xhat_k(i) vectors using the f() function
            for i in range(0, 2*n):
                state_vec = np.hstack((y[0:3], xbreve[:4, i]))
                xbrevedot = self.eom_ukf(dt=t, y=state_vec)

                # Update xbreve by integrating change over timestep
                xbreve[:, i] = xbreve[:, i] + np.hstack((xbrevedot[3:6], 0)) * self.dt

            ## 3.1 Combine xhat_k(i) vectors to receive the a priori state
            xhatukf = np.zeros((n,))

            # Update xhatukf using the weighted sum of xbreve columns
            for i in range(0, 2*n):
                xhatukf = xhatukf + W[i] * xbreve[:, i]

            ## 3.2 Estimate a priori error covariance
            Pukf = np.zeros((n,n))

            for i in range(0, 2*n):
                # Update Pukf using the weighted sum of outer products

                diff = xbreve[:, i] - xhatukf
                Pukf = Pukf + W[i]*(np.outer(diff, diff))

            Pukf = Pukf + Q

            ## 4. Measurement Update
            ## 4.1 Apply linear measurement equation to sigma points
            
            zukf = np.zeros((3, n * 2))

            # Generate zukf measurements
            for i in range(0, n * 2):
                zukf[:, i] = np.dot(H, xbreve[:, i])

            ## 4.2 Combine yhat_k(i) vectors to obtain predicted measurement at time k
            zhat = np.zeros((3))

            # Weighted sum of zukf columns
            for i in range(0, 2 * n):
                zhat = zhat + W[i] * zukf[:, i]
            
            ## 4.3 Estimate covariance of predicted measurement
            Py = np.zeros((3,3))
            Pxy = np.zeros((n,3))

            # Weighted sum for Py and Pxy calculations
            for i in range(0, 2 * n):
                diff_zukf = zukf[:, i] - zhat
                diff_xbreve = xbreve[:, i] - xhatukf

                Py = Py + W[i] * np.outer(diff_zukf, diff_zukf)
                Pxy = Pxy + W[i] * np.outer(diff_xbreve, diff_zukf)

            Py = Py + R

            ## 4.4 Measurement Update using standard Kalman filter equations
            Kukf =  np.dot(Pxy, np.linalg.inv(Py))
            xhatukf = xhatukf + np.dot(Kukf, z - zhat)
            Pukf = Pukf - np.dot(Kukf, np.dot(Py,  np.transpose(Kukf)))

            ## Save data for plotting
            yArray.append(y)
            PArray.append(Pukf)
            xArray.append(xstate)
            zArray.append(z)
            zhatArray.append(zhat)
            xhatukfArray.append(xhatukf)
            PukfArray.append(np.diag(Pukf))
            tArray.append(t)
            z_estimate.append(xhatukf[3])

        return [tArray, attitudeArray, angular_velArray, z_estimate]



        


