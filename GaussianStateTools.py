""" 
Arbitrary Gaussian State Evolution Toolkit v1.4
Created by Ryan P. Marchildon 2017/09/30

Last updated on 2018/15/07

Follows the recipes outlined in:
(1) Stefano Olivares, "Quantum optics in the phase space", ArXiv:1111.0786v2 (2011)
(2) Alessandro Ferraro, Stefano Olivares, and Matteo G. A. Paris, "Gaussian states
    in continuous variable quantum information"

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from numpy.linalg import det
from numpy.linalg import inv
from numpy import dot


" GAUSSIAN STATE OBJECTS "


class GaussianState:
    def __init__(self, n_modes=1):
        """
        Initializes an n-mode Gaussian state in the vacuum state.

        :param n_modes: number of modes (e.g. waveguides) of Gaussian state

        :attrib R_vec: 1-by-2n array, n-mode first-moments vector
                       (<q1>, <p1>, ..., <qn>, <pn>)^T

        :attrib sigma: 2n-by-2n array, n-mode covariance matrix.
        
        Note: we use the convention X = (a + adag)/sqrt(2), p = (a - adag)/isqrt(2),
              hence the diagonal terms of the covariance matrix, which are the only non-zero
              elements in the case of vacuum, are the variances <x^2> and <p^2>, which it can be
              shown DOES evaluate to 1/2 and 1/2 owing to the sqrt(2) term in the x and p definitions.
        
        """
        self.n_modes = n_modes
        self.R_vec = np.zeros(2 * n_modes)[:, np.newaxis]
        self.sigma = 0.5 * np.identity(2 * n_modes)

    def reset(self):
        """
        Resets the current state to N-mode vacuum. 
        
        :return: self, object. 
        
        """
        n_modes_ = self.n_modes
        self.R_vec = np.zeros(2 * n_modes_)[:, np.newaxis]
        self.sigma = 0.5 * np.identity(2 * n_modes_)

        return self

    def copy(self):
        """
        Creates a new object that is a duplicate of the state.

        Necessary because psi_new = psi merely creates a pointer, and if psi gets modified,
        psi_new will be modified likewise.

        :return: psi_new, a new object that is a copy of this state.
        """
        psi_new = GaussianState(n_modes=self.n_modes)
        psi_new.R_vec = self.R_vec
        psi_new.sigma = self.sigma

        return psi_new

    def displace(self, mode_id=1, alpha_mag=0.0, alpha_phase=0.0):
        """
        Applies a displacement operation to the Gaussian state on the specified mode. 

        :param mode_id: int, The mode we are displacing (note: mode_id for first mode is 1,
                        mode_id for second mode is 2)

        :param alpha_mag: float, Magnitude of the displacement.

        :param alpha_phase: float, Phase of the displacement in radians. (e.g. np.pi)

        :return: self: object
        
        Note: we use the convention X = (a + adag)/sqrt(2), p = (a - adag)/isqrt(2),
              which means that for a displacement of D(alpha), which evolves the mode
              operators as Ddag a D = a + alpha, we get the transformations
              x' = x + sqrt(2)Re{alpha}, p' = p + sqrt(2)Im{alpha}.

        """
        self.R_vec[(2 * mode_id - 1) - 1, :] = self.R_vec[(2 * mode_id - 1) - 1, :] \
                                               + np.sqrt(2)*alpha_mag * np.cos(alpha_phase)
        self.R_vec[(2 * mode_id - 0) - 1, :] = self.R_vec[(2 * mode_id - 0) - 1, :] \
                                               + np.sqrt(2)*alpha_mag * np.sin(alpha_phase)

        return self

    def rotate(self, mode_id=1, phase=0.0):
        """
        Applies a phase rotation to the specified mode. 

        :param mode_id: int, the mode we are squeezing. 

        :param phase: float, rotation phase.

        :return self: object
        
        Note: The Wigner function evolves in a CW (NOT a CCW) rotation. So even though
              displacement angles are treated in a positive (i.e. CCW) sense, for phase rotations, 
              a rotation by angle theta is in a negative (i.e. CW) sense with respect to the 
              quadrature axes. 

        """
        n_modes_ = self.n_modes
        theta = phase

        # Define the symplectic transformation in the local subspace
        r_symp_local = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

        # Extend the transformation into the total Hilbert space, via direct summation
        # Note: direct sum in phase space is equal to a tensor product in state space
        r_symp_global = np.zeros((2 * n_modes_, 2 * n_modes_))  # initialize
        for mode_num in range(n_modes_):  # note, mode_num begins at zero

            x_ = 2 * mode_num
            y_ = x_ + 1

            if (mode_num + 1) == mode_id:
                local_mat = r_symp_local
            else:
                local_mat = np.identity(2)

            r_symp_global[x_, x_] = local_mat[0, 0]
            r_symp_global[x_, y_] = local_mat[0, 1]
            r_symp_global[y_, x_] = local_mat[1, 0]
            r_symp_global[y_, y_] = local_mat[1, 1]

        # Apply the symplectic transformation
        self.sigma = dot(r_symp_global, dot(self.sigma, np.transpose(r_symp_global)))
        self.R_vec = dot(r_symp_global, self.R_vec)

        return self

    def dissipate(self, mode_id=1, transmission=1.0):
        """
        Accounts for the impact of a lumped loss on the Gaussian state by applying appropriate
        transformations to the mean vector and covariance matrix.

        :param mode_id: int, the mode we are dissipating.

        :param transmission: float, effective power transmission of optical element (e.g. efficiency).

        :return self: object

        *** Notes on Computation ***: we refer to equations (4.11) and (4.15) found in
        "Gaussian states in continuous variable quantum information"
        by Alessandro Ferraro, Stefano Olivares, and Matteo G. A. Paris.

        Evolution of mean vector:
        -------------------------
        Mean photon number |alpha|^2 = [mean(X)^2 + mean(P)^2]/2
        So you obtain T|alpha|^2 via:  X-> X*sqrt(T), P -> P*sqrt(T)

        Evolution of covariance matrix
        ------------------------------
        Covariance matrix is in block form, e.g. for three modes:
        [11][12][13]
        [21][22][21]
        [31][32][33]
        where [11] is: 0.5*[[<x1x1 + x1x1>, <x1p1 + p1x1>],
                            [<p1x1 + x1p1>, <p1p1 + p1p1>]]
              [21] is: 0.5*[[<x2x1 + x2x1>, <x2p1 + p121>],
                            [<p2x1 + x1p1>, <p2p1 + p1p2>]]
              etc.
        Let's call this the initial covariance matrix sigma_0.

        In the asymptotic limit, if mode i is dissipated:
        1) block [ii] becomes 0.5*[[1, 0], [0, 1]]
        2) blocks [ni] or [in] where n != i become [[0, 0], [0, 0]]
        Let's call this the asymptotic (or dissipation) matrix sigma_inf

        The evolved (dissipated) covariance matrix is merely
        sigma = T*sigma_0 + (1-T)*sigma_inf

        """
        sigma_0 = np.copy(self.sigma)  # initial covariance matrix
        tt = transmission

        # *** Evolution of mean vector ****
        self.R_vec[(2 * mode_id - 1) - 1, :] = self.R_vec[(2 * mode_id - 1) - 1, :]*np.sqrt(tt)
        self.R_vec[(2 * mode_id - 0) - 1, :] = self.R_vec[(2 * mode_id - 0) - 1, :]*np.sqrt(tt)

        # *** Define covariance matrix in the asymptotic limit ***
        sigma_inf = np.copy(self.sigma)  # initialize
        block_id = 2*mode_id - 2  # index of the matrix block [ii] that is dissipated

        # delete all elements that share a row or column with the dissipated matrix block
        dissipated_ids = [block_id, block_id+1]
        for i in range(2*self.n_modes):
            for j in range(2*self.n_modes):
                if i in dissipated_ids or j in dissipated_ids:
                    sigma_inf[i, j] = 0.0

        # now set the elements of the dissipated matrix block to the vacuum covariance matrix
        sigma_inf[block_id, block_id] = 0.5
        sigma_inf[block_id, block_id + 1] = 0.0
        sigma_inf[block_id + 1, block_id] = 0.0
        sigma_inf[block_id + 1, block_id + 1] = 0.5

        # *** Evolve the covariance matrix ***
        self.sigma = tt*sigma_0 + (1 - tt)*sigma_inf

        return self

    def single_mode_squeeze(self, mode_id=1, beta_mag=0.0, beta_phase=0.0):
        """
        Applies a single-mode squeezing operation on the specified mode

        :param mode_id: int, the mode we are squeezing. 

        :param beta_mag: float, squeezing magnitude (equal to r, the squeezing parameter).

        :param beta_phase: float, squeezing phase.
        
        :return self: object
                
        """
        n_modes_ = self.n_modes
        r = beta_mag  # squeezing parameter
        phi = beta_phase

        # Define the symplectic transformation in the local subspace
        s_symp_local = np.cosh(r) * np.identity(2) \
                       + np.sinh(r) * np.array([[np.cos(phi), np.sin(phi)],
                                                [np.sin(phi), (-1) * np.cos(phi)]])

        # Extend the transformation into the total Hilbert space, via direct summation
        # Note: direct sum in phase space is equal to a tensor product in state space
        s_symp_global = np.zeros((2*n_modes_, 2*n_modes_))  # initialize
        for mode_num in range(n_modes_):  # note, mode_num begins at zero

            x_ = 2*mode_num
            y_ = x_ + 1

            if (mode_num + 1) == mode_id:
                local_mat = s_symp_local
            else:
                local_mat = np.identity(2)

            s_symp_global[x_, x_] = local_mat[0, 0]
            s_symp_global[x_, y_] = local_mat[0, 1]
            s_symp_global[y_, x_] = local_mat[1, 0]
            s_symp_global[y_, y_] = local_mat[1, 1]

        # Apply the symplectic transformation
        self.sigma = dot(s_symp_global, dot(self.sigma, np.transpose(s_symp_global)))
        self.R_vec = dot(s_symp_global, self.R_vec)

        return self

    def two_mode_squeeze(self, mode_1_id=1, mode_2_id=2, beta_mag=0.0, beta_phase=0.0):
        """
        Applies a two-mode squeezing operation on the specified modes.
        Mode index counting begins at 1. 

        :param mode_1_id: int, the index of the first mode involved in the two-mode squeezing.

        :param mode_2_id: int, the index of the second mode involved in the two-mode squeezing.

        :param beta_mag: float, squeezing magnitude (equal to r, the squeezing parameter).

        :param beta_phase: float, squeezing phase.

        :return self: object

        """
        n_modes_ = self.n_modes
        r = beta_mag  # squeezing parameter
        phi = beta_phase

        # Define the global symplectic transformation:
        s_symp_global = np.identity(2 * n_modes_)  # initialize

        ind_ref_1 = 2 * (mode_1_id - 1)
        ind_ref_2 = 2 * (mode_2_id - 1)

        # Matrix block corresponding to mode_id_1 local subspace covariance
        s_symp_global[ind_ref_1, ind_ref_1] = np.cosh(r)
        s_symp_global[ind_ref_1 + 1, ind_ref_1 + 1] = np.cosh(r)

        # Matrix block corresponding to mode_id_1 <--> mode_id_2 correlations
        s_symp_global[ind_ref_1, ind_ref_2] = np.sinh(r) * np.cos(phi)
        s_symp_global[ind_ref_1, ind_ref_2 + 1] = np.sinh(r) * np.sin(phi)
        s_symp_global[ind_ref_1 + 1, ind_ref_2] = np.sinh(r) * np.sin(phi)
        s_symp_global[ind_ref_1 + 1, ind_ref_2 + 1] = (-1) * np.sinh(r) * np.cos(phi)

        # Matrix block corresponding to mode_id_2 <--> mode_id_1 correlations
        s_symp_global[ind_ref_2, ind_ref_1] = np.sinh(r) * np.cos(phi)
        s_symp_global[ind_ref_2, ind_ref_1 + 1] = np.sinh(r) * np.sin(phi)
        s_symp_global[ind_ref_2 + 1, ind_ref_1] = np.sinh(r) * np.sin(phi)
        s_symp_global[ind_ref_2 + 1, ind_ref_1 + 1] = (-1) * np.sinh(r) * np.cos(phi)

        # Matrix block corresponding to mode_id_2 local subspace covariance
        s_symp_global[ind_ref_2, ind_ref_2] = np.cosh(r)
        s_symp_global[ind_ref_2 + 1, ind_ref_2 + 1] = np.cosh(r)

        # Apply the symplectic transformation
        self.sigma = dot(s_symp_global, dot(self.sigma, np.transpose(s_symp_global)))
        self.R_vec = dot(s_symp_global, self.R_vec)

        return self

    def two_mode_mix(self, mode_1_id=1, mode_2_id=2, R0 = 0.5, mixing_angle=np.pi/2):
        """
        Applies a mode mixing operation between the two selected modes.
        Default mixing angle is pi/2, which corresponds to a beamsplitter. 
        R0 is the beamsplitter 'reflectivity'. (R0 + T0 = 1)
        The complex mixing amplitude is  arcsin(sqrt(R0))exp(i*mixing_angle).
        
        :param mode_1_id: int, the first mode involved in the interaction, > 0. (numbering starts at 1)

        :param mode_2_id: int, the second mode involved in the interaction, > 0. (numbering starts at 1)

        :param R0: float, beamsplitter reflectivity

        :param mixing_angle: float, given in radians, where pi/2 is for a beamsplitter

        :return self: object
        
        Note: The Wigner function evolves in a CW (NOT a CCW) rotation. So even though
              displacement angles are treated in a positive (i.e. CCW) sense, for 'reflections' we
              pick up a -90 degree rotation (corresponding to a +90 degree phase shift), i.e. a 
              rotation in phase space that is CW, with respect to the quadrature axes. Therefore
              +x -> -p, +p -> +x. 

        """
        n_modes_ = self.n_modes
        theta = mixing_angle
        gamma = np.arcsin(np.sqrt(R0))

        # Define the global symplectic transformation:
        m_symp_global = np.identity(2*n_modes_)  # initialize

        ind_ref_1 = 2 * (mode_1_id - 1)
        ind_ref_2 = 2 * (mode_2_id - 1)

        # Matrix block corresponding to mode_id_1 local subspace covariance
        m_symp_global[ind_ref_1, ind_ref_1] = np.cos(gamma)
        m_symp_global[ind_ref_1 + 1, ind_ref_1 + 1] = np.cos(gamma)

        # Matrix block corresponding to mode_id_1 <--> mode_id_2 correlations
        m_symp_global[ind_ref_1, ind_ref_2] = np.sin(gamma)*np.cos(theta)
        m_symp_global[ind_ref_1, ind_ref_2 + 1] = np.sin(gamma)*np.sin(theta)
        m_symp_global[ind_ref_1 + 1, ind_ref_2] = -np.sin(gamma)*np.sin(theta)
        m_symp_global[ind_ref_1 + 1, ind_ref_2 + 1] = np.sin(gamma)*np.cos(theta)

        # Matrix block corresponding to mode_id_2 <--> mode_id_1 correlations
        m_symp_global[ind_ref_2, ind_ref_1] = -np.sin(gamma)*np.cos(theta)
        m_symp_global[ind_ref_2, ind_ref_1 + 1] = np.sin(gamma)*np.sin(theta)
        m_symp_global[ind_ref_2 + 1, ind_ref_1] = -np.sin(gamma)*np.sin(theta)
        m_symp_global[ind_ref_2 + 1, ind_ref_1 + 1] = -np.sin(gamma)*np.cos(theta)

        # Matrix block corresponding to mode_id_2 local subspace covariance
        m_symp_global[ind_ref_2, ind_ref_2] = np.cos(gamma)
        m_symp_global[ind_ref_2 + 1, ind_ref_2 + 1] = np.cos(gamma)

        # Apply the symplectic transformation
        self.sigma = dot(m_symp_global, dot(self.sigma, np.transpose(m_symp_global)))
        self.R_vec = dot(m_symp_global, self.R_vec)

        # print('The Symplectic Two Mode Mixing Transformation Is:')
        # print(m_symp_global)
        # print('\n')

        return self

    def get_squeezing(self):
        """
        Finds the eigenvalues of the covariance matrix, which provide the variance along
        each principal axis.

        :return: Returns the amount of squeezing, in dB, along each principal axis.

        """
        eig_val, eig_vec = np.linalg.eig(self.sigma)
        return 10*np.log10(eig_val/0.5)


" WIGNER FUNCTION OBJECTS "


class OneModeGaussianWignerFunction:
    def __init__(self, state, range_min=-5, range_max=5, range_num_steps=40):
        """
        Numerically computes the Wigner function of a one-mode state, and returns this as an object.  

        :param state: object of class GaussianState()

        :param range_min: start of quadrature calculation range

        :param range_max: end of quadrature calculation range

        :param range_num_steps: number of calculation steps along each dimension

        :attrib values:  2N-dimensional array containing the Wigner function evaluated at a grid of points 

        :attrib spacing: float, the sampling spacing (stepsize, dx) between grid points

        :attrib x_mesh: 2N-dimensional meshgrid array containing the x-coordinates the wigner function was evaluated at

        :attrib p_mesh: 2N-dimensional meshgrid array containing the p-coordinates the wigner function was evaluated at

        """
        n_modes_ = state.n_modes
        R_vec_ = state.R_vec
        sigma_inv = inv(state.sigma)
        sigma_det = det(state.sigma)

        x_ = np.linspace(range_min, range_max, range_num_steps)
        spacing = np.absolute(x_[1] - x_[0])
        x_mesh, p_mesh = np.meshgrid(x_, x_)

        X_vec = np.array([[x, p] for x, p in zip(np.ravel(x_mesh), np.ravel(p_mesh))])
        X_vec = X_vec[:, :, np.newaxis]

        wigner = []  # initialize as a list, convert to array later
        for i in range(X_vec.shape[0]):
            v1 = X_vec[i, :, :] - R_vec_
            v2 = np.transpose(v1)
            v3 = dot(sigma_inv, v1)
            arg_exp = -0.5 * dot(v2, v3)
            wigner.append(expm(arg_exp) / (np.pi * np.sqrt(sigma_det)))
        wigner = np.array(wigner)
        wigner = wigner.reshape(x_mesh.shape)

        # Ensure Wigner function is normalized to unity
        wigner_trace_x = np.trapz(wigner, x=None, dx=spacing, axis=0)
        wigner_trace_xp = np.trapz(wigner_trace_x, x=None, dx=spacing, axis=0)
        wigner = wigner / wigner_trace_xp

        # Assign attributes
        self.x_mesh = x_mesh
        self.p_mesh = p_mesh
        self.spacing = spacing
        self.values = wigner

    def plot(self):
        """
        Generates a countour plot of a 1-mode Wigner function. 

        :param self: object of class OneModeGaussianWignerFunction()

        :return: none

        """
        x_mesh_ = self.x_mesh  # 2D meshgrid array, sampling coordinates for x
        p_mesh_ = self.p_mesh  # 2D meshgrid array, sampling coordinates for y
        values_ = self.values  # wigner function evaluated at sampling coordinates

        plt.figure()
        cp = plt.contourf(x_mesh_, p_mesh_, values_)
        plt.colorbar(cp)
        plt.axhline(0, color='white')
        plt.axvline(0, color='white')
        plt.title('Wigner Function')
        plt.xlabel('x')
        plt.ylabel('p')
        plt.show()

        return


class TwoModeGaussianWignerFunction:
    def __init__(self, state, range_min=-5, range_max=5, range_num_steps=40):
        """
        Numerically computes the Wigner function of a two-mode state, and returns this as an object. 
    
        :param state: object of class GaussianState()

        :param range_min: start of quadrature calculation range

        :param range_max: end of quadrature calculation range

        :param range_num_steps: number of calculation steps along each dimension
    
        :attrib values: 2N-dimensional array, the Wigner function evaluated at a grid of points

        :attrib spacing: float, the sampling spacing (stepsize, dx) between grid points

        :attrib x1_mesh: 2N-dimensional meshgrid array of x1-coordinates sampling the wigner function

        :attrib p1_mesh: 2N-dimensional meshgrid array of p1-coordinates sampling the wigner function

        :attrib x2_mesh: 2N-dimensional meshgrid array of x2-coordinates sampling the wigner function

        :attrib p2_mesh: 2N-dimensional meshgrid array of p2-coordinates sampling the wigner function
    
        """
        n_modes_ = state.n_modes
        R_vec_ = state.R_vec
        sigma_inv = inv(state.sigma)
        sigma_det = det(state.sigma)

        x_ = np.linspace(range_min, range_max, num=range_num_steps)
        spacing = np.absolute(x_[1] - x_[0])
        x1_mesh, p1_mesh, x2_mesh, p2_mesh = np.meshgrid(x_, x_, x_, x_)

        X_vec = np.array([[x1, p1, x2, p2] for x1, p1, x2, p2 in zip(np.ravel(x1_mesh), np.ravel(p1_mesh),
                                                                     np.ravel(x2_mesh), np.ravel(p2_mesh))])
        X_vec = X_vec[:, :, np.newaxis]

        wigner = []  # initialize as a list, convert to array later
        for i in range(X_vec.shape[0]):
            v1 = X_vec[i, :, :] - R_vec_
            v2 = np.transpose(v1)
            v3 = dot(sigma_inv, v1)
            arg_exp = -0.5 * dot(v2, v3)
            wigner.append(expm(arg_exp) / ((np.pi**n_modes_) * np.sqrt(sigma_det)))
        wigner = np.array(wigner)
        wigner = wigner.reshape(x1_mesh.shape)

        # Ensure Wigner function is normalized to unity
        num_axes = 2 * n_modes_
        normalization = wigner  # initialize
        for _ in range(num_axes):  # repeat over each axis
            normalization = np.trapz(normalization, x=None, dx=spacing, axis=0)
        wigner = wigner / normalization

        # Assign Attributes
        self.x1_mesh = x1_mesh
        self.p1_mesh = p1_mesh
        self.x2_mesh = x2_mesh
        self.p2_mesh = p2_mesh
        self.spacing = spacing
        self.values = wigner

    def plot(self, axis_1='x1', axis_2='p1', plot_all=False):
        """
        Generates a countour plot of a 2-mode Wigner function along the specified axes. 

        :param self: object of class TwoModeGaussianWignerFunction().

        :param axis_1: specified plotting axis 1 (options are 'x1', 'p1', 'x2', or 'p2')

        :param axis_2: specified plotting axis 1 (options are 'x1', 'p1', 'x2', or 'p2')

        :return: none

        """
        spacing_ = self.spacing  # float, sampling spacing used to create the wigner functions
        x1_mesh_ = self.x1_mesh  # 2N-dim meshgrid containing the x1-coordinates the wigner function was evaluated at
        values_ = self.values  # wigner function evaluated at sampling coordinates

        # Note, the relevant axes in the meshgrids are like this:
        # print(x1_mesh[0, :, 0, 0]) --> axis 1
        # print(p1_mesh[:, 0, 0, 0]) --> axis 0
        # print(x2_mesh[0, 0, :, 0]) --> axis 2
        # print(p2_mesh[0, 0, 0, :]) --> axis 3

        # Recover the original x_vec used to generate the meshgrids
        x_vec = x1_mesh_[0, :, 0, 0]
        # Create new meshgrids
        mesh_1, mesh_2 = np.meshgrid(x_vec, x_vec)

        # Integrate over all unused axes, and recover 2D meshgrids.
        # Note, the axis labels are 1, 0, 2, 3 for x1, p1, x2, p2 respectively.
        if (axis_1 == 'x1' and axis_2 == 'p1') or (axis_1 == 'p1' and axis_2 == 'x1'):
            # Integrate over axes 2 and 3
            values_ = np.trapz(values_, x=None, dx=spacing_, axis=3)
            values_ = np.trapz(values_, x=None, dx=spacing_, axis=2)
            if axis_1 == 'p1':
                axis_1_mesh = mesh_2
                axis_2_mesh = mesh_1
            else:
                axis_1_mesh = mesh_1
                axis_2_mesh = mesh_2
        elif (axis_1 == 'x2' and axis_2 == 'p2') or (axis_1 == 'p2' and axis_2 == 'x2'):
            # Integrate over axes 0 and 1
            values_ = np.trapz(values_, x=None, dx=spacing_, axis=0)
            values_ = np.trapz(values_, x=None, dx=spacing_, axis=0)  # axis 1 (after reduction)
            if axis_1 == 'p2':
                axis_1_mesh = mesh_1
                axis_2_mesh = mesh_2
            else:
                axis_1_mesh = mesh_2
                axis_2_mesh = mesh_1
        elif (axis_1 == 'x1' and axis_2 == 'x2') or (axis_1 == 'x2' and axis_2 == 'x1'):
            # Integrate over axes 0 and 3
            values_ = np.trapz(values_, x=None, dx=spacing_, axis=3)
            values_ = np.trapz(values_, x=None, dx=spacing_, axis=0)
            if axis_1 == 'x2':
                axis_1_mesh = mesh_1
                axis_2_mesh = mesh_2
            else:
                axis_1_mesh = mesh_2
                axis_2_mesh = mesh_1
        elif (axis_1 == 'p1' and axis_2 == 'p2') or (axis_1 == 'p2' and axis_2 == 'p1'):
            # Integrate over axes 1 and 2
            values_ = np.trapz(values_, x=None, dx=spacing_, axis=1)
            values_ = np.trapz(values_, x=None, dx=spacing_, axis=1)  # axis 2 (after reduction)
            if axis_1 == 'p2':
                axis_1_mesh = mesh_1
                axis_2_mesh = mesh_2
            else:
                axis_1_mesh = mesh_2
                axis_2_mesh = mesh_1
        elif (axis_1 == 'x1' and axis_2 == 'p2') or (axis_1 == 'p2' and axis_2 == 'x1'):
            # Integrate over axes 0 and 2
            values_ = np.trapz(values_, x=None, dx=spacing_, axis=0)
            values_ = np.trapz(values_, x=None, dx=spacing_, axis=1)  # axis 2 (after reduction)
            if axis_1 == 'p2':
                axis_1_mesh = mesh_2
                axis_2_mesh = mesh_1
            else:
                axis_1_mesh = mesh_1
                axis_2_mesh = mesh_2
        elif (axis_1 == 'p1' and axis_2 == 'x2') or (axis_1 == 'x2' and axis_2 == 'p1'):
            # Integrate over axes 1 and 3
            values_ = np.trapz(values_, x=None, dx=spacing_, axis=3)
            values_ = np.trapz(values_, x=None, dx=spacing_, axis=1)
            if axis_1 == 'x2':
                axis_1_mesh = mesh_1
                axis_2_mesh = mesh_2
            else:
                axis_1_mesh = mesh_2
                axis_2_mesh = mesh_1
        else:
            raise Exception('Check axis_1 and axis_2 inputs; options are \'x1\',\'p1\',\'x2\', or \'p2\'.')

        if plot_all is False:
            plt.figure()
            cp = plt.contourf(axis_1_mesh, axis_2_mesh, values_)
            plt.colorbar(cp)
            plt.axhline(0, color='white')
            plt.axvline(0, color='white')
            plt.title('Wigner Function Slice')
            plt.xlabel(axis_1)
            plt.ylabel(axis_2)
            plt.show()
            return
        elif plot_all is True:
            return axis_1_mesh, axis_2_mesh, values_

    def plot_all(self, figsize_=10):

        axis_1 = ['x1', 'x2', 'x1', 'p1']
        axis_2 = ['p1', 'p2', 'x2', 'p2']

        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row',
                                                   figsize=(1.3*figsize_, figsize_))

        for a1, a2, this_plot in zip(axis_1, axis_2, [ax1, ax2, ax3, ax4]):
            axis_1_mesh, axis_2_mesh, values_ = self.plot(axis_1=a1, axis_2=a2, plot_all=True)
            z = this_plot.contourf(axis_1_mesh, axis_2_mesh, values_)
            plt.colorbar(z, ax=this_plot)
            this_plot.axhline(0, color='white')
            this_plot.axvline(0, color='white')
            this_plot.set_xlabel(a1)
            this_plot.set_ylabel(a2)

        plt.show()

        return

    def plot_all_export(self, figsize_=10):
        """
        Exports wigner function values along with plot generation.
        
        """

        axis_1 = ['x1', 'x2', 'x1', 'p1']
        axis_2 = ['p1', 'p2', 'x2', 'p2']

        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(1.3*figsize_, figsize_))

        a1_mesh = []
        a2_mesh = []
        z_vals = []
        for a1, a2, this_plot in zip(axis_1, axis_2, [ax1, ax2, ax3, ax4]):
            axis_1_mesh, axis_2_mesh, values_ = self.plot(axis_1=a1, axis_2=a2, plot_all=True)
            z = this_plot.contourf(axis_1_mesh, axis_2_mesh, values_)
            plt.colorbar(z, ax=this_plot)
            this_plot.axhline(0, color='white')
            this_plot.axvline(0, color='white')
            this_plot.set_xlabel(a1)
            this_plot.set_ylabel(a2)

            a1_mesh.append(axis_1_mesh)
            a2_mesh.append(axis_2_mesh)
            z_vals.append(values_)

        plt.show()

        return a1_mesh, a2_mesh, z_vals


" METRICS AND HELPER FUNCTIONS "


def WignerOverlap(wigner_state_1, wigner_state_2):
    """
    Computes the overlap between two N-mode Wigner functions. 
    
    :param wigner_state_1: object of class [N]ModeGaussianWignerFunction() for state 1

    :param wigner_state_2: object of class [N]ModeGaussianWignerFunction() for state 2
    
    :return: float, overlap factor between the two wigner functions
    
    """
    spacing = wigner_state_1.spacing  # float, sampling spacing used to create the wigner functions
    wigner_1 = wigner_state_1.values  # 2N-dimensional array, sampled wigner function 1
    wigner_2 = wigner_state_2.values  # 2N-dimensional array, sampled wigner function 2

    num_axes = np.size(wigner_1.shape)

    # first get normalization from case where we have |W(x,p)|^2
    normalization = np.multiply(wigner_1, wigner_1)  # initialize
    for _ in range(num_axes):  # repeat over each axis
        normalization = np.trapz(normalization, x=None, dx=spacing, axis=0)

    # now compute the overlap between the two Wigner functions
    integral = np.multiply(wigner_1, wigner_2)  # initialize
    for _ in range(num_axes):  # repeat over each axis
        integral = np.trapz(integral, x=None, dx=spacing, axis=0)

    return integral/normalization


def OneModeFidelity(state_1, state_2):
    """
    Returns the Uhlmann's fidelity of two single-mode Gaussian states.
    
    :param state_1: object of class GuassianState()

    :param state_2: object of class GaussianState()
    
    :return: Uhlmann's fidelity
    
    See equation (103) in 
    [Stefano Olivares, "Quantum optics in the phase space", ArXiv:1111.0786v2 (2011)]
    
    """
    r1, s1 = (state_1.R_vec, state_1.sigma)
    r2, s2 = (state_2.R_vec, state_2.sigma)

    det_s1 = det(s1)
    det_s2 = det(s2)
    det_s_sum = det(s1 + s2)
    inv_s_sum = inv(s1 + s2)

    delta = 4*(det_s1 - 0.25)*(det_s2 - 0.25)

    arg_exp = -0.5 * dot(np.transpose(r1 - r2), dot(inv_s_sum, (r1 - r2)))

    if delta < 0 and np.absolute(delta) < 0.001:  # prevents NaN error in sqrt()
        delta = 0                                 # by ensuring no small negative numbers are passed.

    fidelity = expm(arg_exp) / (np.sqrt(det_s_sum + delta) - np.sqrt(delta))

    return np.asscalar(fidelity)


def TwoModeFidelity(state_1, state_2):
    """
    Returns the Uhlmann's fidelity of two 2-mode Gaussian states.

    :param state_1: object of class GuassianState()

    :param state_2: object of class GaussianState()

    :return: Uhlmann's fidelity

    See equations (104)-(106) in 
    [Stefano Olivares, "Quantum optics in the phase space", ArXiv:1111.0786v2 (2011)]

    """
    r1, s1 = (state_1.R_vec, state_1.sigma)
    r2, s2 = (state_2.R_vec, state_2.sigma)

    epsilon = 0.0000001 * np.identity(4)  # this padding helps prevent overflow in the determinant
    # calculation, which can raise the following error:
    # "RuntimeWarning: invalid value encountered in slogdet sign,
    # logdet = _umath_linalg.slogdet(a, signature=signature)"

    det_s_sum = det(s1 + s2)
    inv_s_sum = inv(s1 + s2)

    omega = np.array([[0, 1, 0, 0],
                      [-1, 0, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, -1, 0]])  # note, this is formed through a DIRECT SUM, not KRONECKER SUM

    A = det(dot(omega, dot(s1, dot(omega, s2))) - 0.25*np.identity(4))/det_s_sum
    B = (det(s1 + 0.5j*omega + epsilon))*(det(s2 + 0.5j*omega + epsilon))/det_s_sum

    arg_exp = -0.5 * dot(np.transpose(r1 - r2), dot(inv_s_sum, (r1 - r2)))

    traceRho1Rho2 = expm(arg_exp) / (np.sqrt(det_s_sum))

    gamma = 2*np.sqrt(A) + 2*np.sqrt(B) + 0.5

    fidelity = traceRho1Rho2*((np.sqrt(gamma) + np.sqrt(gamma - 1))**2)

    return np.asscalar(np.absolute(fidelity))


def delete_mode(psi_, deleted_mode_id=3):
    """
    Traces out the specified mode and returns the reduced state.
    ***IMPORTANT***: Assumes input state of 3 Gaussian modes.
    TODO: Generalize to N modes.

    :param psi_: object of class GaussianState() for which we want to 'delete' a mode.

    :param deleted_mode_id: mode we want to 'delete' (surrogate for trace)

    :return psi_new: object of class GaussianState() with the desired mode deleted

    """
    if deleted_mode_id == 1:
        deleted_entries = [0, 1]
    elif deleted_mode_id == 2:
        deleted_entries = [2, 3]
    elif deleted_mode_id == 3:
        deleted_entries = [4, 5]
    else:
        raise Exception('deleted_mode_id must be 1, 2, or 3')

    FMV = psi_.R_vec  # get the first-moments vector
    CVM = psi_.sigma  # get the covariance matrix

    FMV_new = np.delete(FMV, deleted_entries, axis=0)  # delete the last two rows
    CVM_new = np.delete(np.delete(CVM, deleted_entries, axis=0), deleted_entries, axis=1)

    psi_new = gs.GaussianState(n_modes=2)
    psi_new.R_vec = FMV_new
    psi_new.sigma = CVM_new

    return psi_new