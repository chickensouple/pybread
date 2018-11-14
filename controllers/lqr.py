import numpy as np
import scipy.linalg

class LQR(object):
    """
    Computes LQR with recursive solution for discrete time systems

    For shorthand,
    state_dim is m
    control_dim is l

    The problem that is solved is

    for x_{k+1} = Ax + Bu
    find K for
    u_k = K_k x_k in order
    min x_n'Px_n + sum{x_k' Q x_k + u_k' R u_k} from k = 0 to N-1
    subject to x_{k+1} = A x_{k} + b u_{k}

    You can approximate the infinite horizon LQR problem
    by using a large enough N
    """
    def __init__(self, A, B, P, Q, R):
        """
        Initializes LQR solver

        Args:
            A (numpy array): size (m, m) matrix
            B (numpy array): size (m, l) matrix
            P (numpy array): size (m, m) matrix
            Q (numpy array): size (m, m) matrix
            R (numpy array): size (l, l) matrix
        """
        self.state_dim = A.shape[1]
        self.control_dim = B.shape[1]

        if A.shape != (self.state_dim, self.state_dim):
            raise Exception('A must be m by m')
        if B.shape != (self.state_dim, self.control_dim):
            raise Exception('B must be m by l')
        if P.shape != (self.state_dim, self.state_dim):
            raise Exception('P must be m by m')
        if Q.shape != (self.state_dim, self.state_dim):
            raise Exception('Q must be m by m')
        if R.shape != (self.control_dim, self.control_dim):
            raise Exception('R must be l by l')

        if np.max(np.abs(P - P.T)) >= 1e-5:
            raise Exception('P must be symmetric')
        if not np.all(np.linalg.eigvals(P) >= 0):
            raise Exception('P must be symmetric positive semdefinite')

        if np.max(np.abs(Q - Q.T)) >= 1e-5:
            raise Exception('Q must be symmetric')
        if not np.all(np.linalg.eigvals(Q) >= 0):
            raise Exception('Q must be symmetric positive semdefinite')

        if np.max(np.abs(R - R.T)) >= 1e-5:
            raise Exception('R must be symmetric')
        if not np.all(np.linalg.eigvals(R) > 0):
            raise Exception('R must be symmetric positive definite')

        self.A = A
        self.B = B
        self.P = P
        self.Q = Q
        self.R = R


    def solve(self, N, keep_intermediate=False):
        """
        Solves LQR problem with recursive approach.
        This will return the gain matrices for optimal control and the cost matrices

        if keep_intermediate is false, this will only return the K 
        at the final step of the algrithm (K_1). 
        set it to false to get approximate solution to infinite horizon case


        Args:
            N (int): time horizon
            keep_intermediate (bool, optional): set to true
            to return all intermediate control gain matrices (K_k)
        Returns:
            if keep_intermediate is true, returns list of (m, l) gain matrices (K) 
            where the first index has the gain matrix for the last time step
            and the last index has the gain matrix for the first time step
            as well as a list of (m, m) cost matrices (P)

            if keep_intermediate is false, returns a single (m, l) gain matrix and (m, m) cost matrix
        """
        P_k = np.copy(self.P)

        if keep_intermediate:
            K_list = []
            P_list = []
        for i in range(N-1):
            K_1 = np.dot(self.B.T, np.dot(P_k, self.B)) + self.R
            K_2 = -np.dot(self.B.T, np.dot(P_k, self.A))
            K = np.linalg.solve(K_1, K_2)

            P_k = np.dot(self.A.T, np.dot(P_k, self.A)) + self.Q + \
                np.dot(np.dot(self.A.T, np.dot(P_k, self.B)), K)

            if keep_intermediate:
                K_list.append(K)
                P_list.append(P_k)

        if keep_intermediate:
            return K_list, P_list
        else:
            return K, P_k

    def solve_batch(self, N, x0):
        """
        Solves LQR problem with batch approach. This will only return open loop control inputs.

        Args:
            N (int): time horizon
            x0 (numpy array): intial state, (m, 1) matrix

        Returns:
            returns a (N-1, l) matrix of optimal controls where
            the first row is the control input at the first time step
            Also returns the optimal cost of being in state x0
        """

        # S_x * x0 + X_u * u = X
        # TODO: cite morari book

        # S_x is (N*m, m)
        S_x = np.zeros((N*self.state_dim, self.state_dim))

        A_pow = np.eye(self.state_dim)
        for i in range(N):
            S_x[i*self.state_dim:(i+1)*self.state_dim, :] = A_pow
            A_pow = np.dot(self.A, A_pow)

        # S_u is (N*m, (N-1)*l)
        S_u = np.zeros((N*self.state_dim, (N-1)*self.control_dim))
        B_mat = np.copy(self.B)
        for i in range(1, N):
            row_1 = i*self.state_dim
            row_2 = (i+1)*self.state_dim
            col_1 = 0
            col_2 = self.control_dim
            for j in range(N-i):
                S_u[row_1:row_2, col_1:col_2] = B_mat
                row_1 += self.state_dim
                row_2 += self.state_dim
                col_1 += self.control_dim
                col_2 += self.control_dim

            B_mat = np.dot(self.A, B_mat)

        Q_list = [self.Q for _ in range(N-1)]
        Q_list.append(self.P)

        # Q_mat is (N*m, N*m)
        Q_mat = scipy.linalg.block_diag(*Q_list)

        # R_mat is ((N-1)*l, (N-1)*l)
        R_mat = scipy.linalg.block_diag(*[self.R for _ in range(N-1)])


        # H is ((N-1)*l, (N-1)*l)
        H = np.dot(S_u.T, np.dot(Q_mat, S_u)) + R_mat
        # F is (m, (N-1)*l)
        F = np.dot(S_x.T, np.dot(Q_mat, S_u))
        # Y is (m, m)
        Y = np.dot(S_x.T, np.dot(Q_mat, S_x))

        # controller of form u = K*x0
        K = -np.linalg.solve(H, F.T)
        u = np.dot(K, x0)

        cost = np.dot(x0.T, np.dot(F, u)) + np.dot(x0.T, np.dot(Y, x0))
        cost = np.asscalar(cost)

        # reshape (m * N-1 by 1) into (N-1 by m)
        u = np.reshape(u, (-1, self.control_dim))

        return u, cost

