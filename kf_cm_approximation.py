import numpy as np

from math import sin, cos

class CM_approximation_KF():
    """ Implements the CM approximation using a Kalman Filter
        state vector x: [wx_k, wy_k, wz_k, rx_k, ry_k, rz_k]

        _k denotes time = k
        _km1 denotes time = k-1

        _est denotes estimated value

        captial letters denote matrices

        Mp, Kp, ... denotes M posteriori, K posteriori
        Mm, Km, ... denotes M priori, K priori

        example:
        Pp_k = Matrix p posteriori at time=k
        xm_k_est = estimated value of state vector x at time=k 
    """

    def __init__(self, R_true):
        ## Matrices setup
        # Process noise matrix R
        # self.R = np.zeros((3,3))
        self.R = np.matrix([[0.01**2, 0, 0],
                            [0.0, 0.01**2, 0.0],
                            [0.0, 0.0, 0.01**2]])
        
        # Measurement noise matrix Q
        self.Q = np.matrix([[0.004, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.004, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.004, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, (R_true[0])**2, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, (R_true[1])**2, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, (R_true[2])**2]])
        
        self.H = np.hstack((np.eye(3), np.zeros((3,3))))

        ## Initial values
        self.P_post = np.zeros((6,6))

        # initial true state
        self.x = np.array((0, 0, 0, R_true[0], R_true[1], R_true[2]))

        # initial state estimate
        self.xhat_post  = np.hstack((np.zeros(3), self.x[3:]))

        # Generate random noise
        noise = np.sqrt(self.R) * np.random.randn(3, 1) * 0.1

        # initial measurement
        self.z = np.dot(self.H, self.x) + noise

    def calc_phi_matrix(self, dT, m:float, g:float, J:np.ndarray, attitude_km1:np.ndarray, attitude_k:np.ndarray):
        """ DaSilva (4.5) p. 42
        Args:
            dT:                 sampling size
            m:                  mass
            g:                  grav. accel.
            J:                  inertia matrix
            attitude_k:         roll, pitch, yaw at t=k
            attitude_km1:       roll, pitch, yaw at t=k-1
        """

        phi_km1 = attitude_km1[0]
        theta_km1 = attitude_km1[1]

        phi_k = attitude_k[0]
        theta_k = attitude_k[1]

        phi12_k = -1*m*g*dT*0.5/J[0,0]*(cos(phi_k)*cos(theta_k) + cos(phi_km1)*cos(theta_km1))
        phi13_k =  m*g*dT*0.5/J[0,0]*(sin(phi_k)*cos(theta_k) + (sin(phi_km1)*cos(theta_km1)))
        phi21_k =  m*g*dT*0.5/J[1,1]*(cos(phi_k)*cos(theta_k) + (cos(phi_km1)*cos(theta_km1)))
        phi23_k =  m*g*dT*0.5/J[1,1]*(sin(theta_k) + sin(theta_km1))
        phi31_k = -1*m*g*dT*0.5/J[2,2]*(sin(phi_k)*cos(phi_k) + (sin(phi_km1)*cos(theta_km1)))
        phi32_k = -1*m*g*dT*0.5/J[2,2]*(sin(theta_k) + sin(theta_km1))

        phi_mat_k = np.array([[0, phi12_k, phi13_k],
                            [phi21_k, 0, phi23_k],
                            [phi31_k, phi32_k, 0]])

        return phi_mat_k        

    def calc_F_k(self, phi_mat):
        # (4.27)
        phi_mat_extended = np.vstack([phi_mat, np.identity(3)])
        extender = np.vstack([np.identity(3), np.zeros((3,3))])
        
        return np.hstack([extender, phi_mat_extended])
    
    def calc_R_avg(self, data:np.ndarray, samples:float, array_access_offset=3):
        """ Calculates the avg of a vector that is stored in a list.
        """
        r_avg = np.zeros((3,)) 

        for i in range(1, int(samples)):
            r_avg[0] = r_avg[0] + data[len(data) - i][0 + array_access_offset]
            r_avg[1] = r_avg[1] + data[len(data) - i][1 + array_access_offset]
            r_avg[2] = r_avg[2] + data[len(data) - i][2 + array_access_offset]

        r_avg[0] = r_avg[0]/samples
        r_avg[1] = r_avg[1]/samples
        r_avg[2] = r_avg[2]/samples

        return r_avg
