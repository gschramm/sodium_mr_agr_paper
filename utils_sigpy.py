from __future__ import annotations

import sigpy
import math
import numpy as np
import cupy as cp

from typing import Union

class NUFFTT2starDualEchoModel:

    def __init__(self,
                 ishape: tuple[int, int, int],
                 k_1_cm: np.ndarray,
                 field_of_view_cm: float = 22.,
                 acq_sampling_time_ms: float = 0.016,
                 time_bin_width_ms: float = 0.25,
                 scale: float = 1.,
                 echo_time_1_ms: float = 0.5,
                 echo_time_2_ms: float = 5,
                 nufft_kwargs=None) -> None:
        """sigpy (dual echo) forward nufft operator including monoexp. T2* decay modeling

        Parameters
        ----------
        ishape : tuple[int, int, int]
            shape of the input image
        k_1_cm : np.ndarray
            input kx, ky, kz coordinates - shape: (num_samples, num_readouts, 3)
            units 1/cm
        field_of_view_cm : float, optional
            field of view in cm, by default 220.
        acq_sampling_time_ms : float, optional
            samplignt time during acquisition in ms, by default 0.01
        time_bin_width_ms : float, optional
            time bin width for modeling T2* decay, by default 0.25
        scale : float, optional
            scale of the forward operator, by default 0.03
        echo_time_1_ms : float, optional
            first echo time in ms, by default 0.5
        echo_time_2_ms : float, optional
            second echo time in ms, by default 5.
        """
        self._ishape = ishape
        self._scale = scale
        if nufft_kwargs is None:
            self._nufft_kwargs = {}
        else:
            self._nufft_kwargs = nufft_kwargs

        self._acq_sampling_time_ms = acq_sampling_time_ms
        self._time_bin_width_ms = time_bin_width_ms
        self._echo_time_1_ms = echo_time_1_ms
        self._echo_time_2_ms = echo_time_2_ms

        self._time_bins_inds = np.array_split(
            np.arange(k_1_cm.shape[0]),
            math.ceil(k_1_cm.shape[0] /
                      (self._time_bin_width_ms / self._acq_sampling_time_ms)))

        self._coords = []

        for _, time_bin_inds in enumerate(self._time_bins_inds):
            chunk_coords_1_cm = k_1_cm[time_bin_inds, :, :].reshape(
                -1, k_1_cm.shape[-1])

            self._coords.append(chunk_coords_1_cm * field_of_view_cm)

        self._x = None
        self._dual_echo_data = None

        self._phase_factor_1 = None
        self._phase_factor_2 = None

        self._data_weights_1 = None
        self._data_weights_2 = None

    @property
    def x(self) -> Union[None, cp.ndarray]:
        return self._x

    @x.setter
    def x(self, value: Union[None, cp.ndarray]) -> None:
        self._x = value

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        self._scale = value

    @property
    def phase_factor_1(self) -> Union[None, cp.ndarray]:
        return self._phase_factor_1

    @phase_factor_1.setter
    def phase_factor_1(self, value: Union[None, cp.ndarray]) -> None:
        self._phase_factor_1 = value

    @property
    def phase_factor_2(self) -> Union[None, cp.ndarray]:
        return self._phase_factor_2

    @phase_factor_2.setter
    def phase_factor_2(self, value: Union[None, cp.ndarray]) -> None:
        self._phase_factor_2 = value

    @property
    def data_weights_1(self) -> Union[None, cp.ndarray]:
        return self._data_weights_1

    @data_weights_1.setter
    def data_weights_1(self, value: Union[None, cp.ndarray]) -> None:
        self._data_weights_1 = value

    @property
    def data_weights_2(self) -> Union[None, cp.ndarray]:
        return self._data_weights_2

    @data_weights_2.setter
    def data_weights_2(self, value: Union[None, cp.ndarray]) -> None:
        self._data_weights_2 = value

    @property
    def dual_echo_data(self) -> Union[None, cp.ndarray]:
        return self._dual_echo_data

    @dual_echo_data.setter
    def dual_echo_data(self, value: Union[None, cp.ndarray]) -> None:
        self._dual_echo_data = value

    def get_operators_wo_decay_model(
            self) -> tuple[sigpy.linop.Linop, sigpy.linop.Linop]:
        op1 = self._scale * sigpy.linop.NUFFT(
            self._ishape, cp.vstack(self._coords), **self._nufft_kwargs)

        op2 = self._scale * sigpy.linop.NUFFT(
            self._ishape, cp.vstack(self._coords), **self._nufft_kwargs)

        if self._phase_factor_1 is not None:
            op1 = op1 * sigpy.linop.Multiply(self._ishape,
                                             self._phase_factor_1)

        if self._phase_factor_2 is not None:
            op2 = op2 * sigpy.linop.Multiply(self._ishape,
                                             self._phase_factor_2)

        if self._data_weights_1 is not None:
            op1 = sigpy.linop.Multiply(op1.oshape, self._data_weights_1) * op1

        if self._data_weights_2 is not None:
            op2 = sigpy.linop.Multiply(op2.oshape, self._data_weights_2) * op2

        return op1, op2

    def get_operators_w_decay_model(
            self,
            r: cp.ndarray) -> tuple[sigpy.linop.Linop, sigpy.linop.Linop]:
        """NUFFT operators for dual echo including mono-exponential decay model

        Parameters
        ----------
        r : cp.ndarray
            ratio image (2nd / 1st echo image)

        Returns
        -------
        Union[sigpy.linop.Linop, sigpy.linop.Linop]
        """
        op1s = []
        op2s = []
        for i, time_bin_inds in enumerate(self._time_bins_inds):
            #setup the decay image
            readout_time_1_ms = self._echo_time_1_ms + time_bin_inds.mean(
            ) * self._acq_sampling_time_ms
            readout_time_2_ms = self._echo_time_2_ms + time_bin_inds.mean(
            ) * self._acq_sampling_time_ms

            n_1 = ((readout_time_1_ms) /
                   (self._echo_time_2_ms - self._echo_time_1_ms))
            n_2 = ((readout_time_2_ms) /
                   (self._echo_time_2_ms - self._echo_time_1_ms))

            op1s.append(
                sigpy.linop.NUFFT(self._ishape, self._coords[i], **
                                  self._nufft_kwargs) *
                sigpy.linop.Multiply(self._ishape, r**n_1))

            op2s.append(
                sigpy.linop.NUFFT(self._ishape, self._coords[i], **
                                  self._nufft_kwargs) *
                sigpy.linop.Multiply(self._ishape, r**n_2))

        operator1 = self._scale * sigpy.linop.Vstack(op1s)
        operator2 = self._scale * sigpy.linop.Vstack(op2s)

        if self._phase_factor_1 is not None:
            operator1 = operator1 * sigpy.linop.Multiply(
                self._ishape, self._phase_factor_1)

        if self._phase_factor_2 is not None:
            operator2 = operator2 * sigpy.linop.Multiply(
                self._ishape, self._phase_factor_2)

        if self._data_weights_1 is not None:
            operator1 = sigpy.linop.Multiply(operator1.oshape,
                                             self._data_weights_1) * operator1

        if self._data_weights_2 is not None:
            operator2 = sigpy.linop.Multiply(operator2.oshape,
                                             self._data_weights_2) * operator2

        return operator1, operator2

    def data_fidelity_gradient_r(self, r: cp.ndarray) -> cp.ndarray:
        """calculate the gradient of the dual echo data fidelity w.r.t to the ratio image

           The gradient is given by n * r*(n-1) Re (x.conj() A^H d)
           where d is the difference between current expectation and the model and
           A^H is the adjoint of the dual echo NUFFT operator without decay model
        
           Before calling the method, you need to set the x (current image estimate) 
           and dual_echo_data properties 

        Parameters
        ----------
        r : cp.ndarray
            the current ratio image

        Returns
        -------
        cp.ndarray
        """

        if self._x is None:
            raise ValueError("x is not set")

        if self._dual_echo_data is None:
            raise ValueError("dual echo data is not set")

        # calculate the difference between the current expectation and the model
        A_e1, A_e2 = self.get_operators_w_decay_model(r)
        A = sigpy.linop.Vstack([A_e1, A_e2])
        diff = A(self._x) - self._dual_echo_data

        # account for data weights
        if self._data_weights_1 is not None:
            diff[:A_e1.oshape[0]] *= self._data_weights_1
        if self._data_weights_2 is not None:
            diff[A_e1.oshape[0]:] *= self._data_weights_2

        del A
        del A_e1
        del A_e2

        # setup the operator we need to calculate the gradient
        f1s = []
        f2s = []
        for i, time_bin_inds in enumerate(self._time_bins_inds):
            #setup the decay image
            readout_time_1_ms = self._echo_time_1_ms + time_bin_inds.mean(
            ) * self._acq_sampling_time_ms
            readout_time_2_ms = self._echo_time_2_ms + time_bin_inds.mean(
            ) * self._acq_sampling_time_ms

            n_1 = ((readout_time_1_ms) /
                   (self._echo_time_2_ms - self._echo_time_1_ms))
            n_2 = ((readout_time_2_ms) /
                   (self._echo_time_2_ms - self._echo_time_1_ms))

            f1s.append(
                sigpy.linop.Multiply(self._ishape,
                                     n_1 * (r**(n_1 - 1)) * self._x.conj()) *
                sigpy.linop.NUFFT(self._ishape, self._coords[i], **
                                  self._nufft_kwargs).H)

            f2s.append(
                sigpy.linop.Multiply(self._ishape,
                                     n_2 * (r**(n_2 - 1)) * self._x.conj()) *
                sigpy.linop.NUFFT(self._ishape, self._coords[i], **
                                  self._nufft_kwargs).H)

        f1s = self._scale * sigpy.linop.Hstack(f1s)
        f2s = self._scale * sigpy.linop.Hstack(f2s)

        if self._phase_factor_1 is not None:
            f1s = sigpy.linop.Multiply(self._ishape,
                                       self._phase_factor_1.conj()) * f1s

        if self._phase_factor_2 is not None:
            f2s = sigpy.linop.Multiply(self._ishape,
                                       self._phase_factor_2.conj()) * f2s

        h_op = sigpy.linop.Hstack([f1s, f2s])

        return cp.real(h_op(diff))


def projected_gradient_operator(ishape: tuple[int, ...],
                                prior_image: cp.ndarray,
                                eta: float = 0.) -> sigpy.linop.Linop:
    """Projected gradient operator as defined in https://doi.org/10.1137/15M1047325.
       Gradient operator that return the component of a gradient that is orthogonal 
       to a joint gradient field (derived from a prior image)

    Parameters
    ----------
    ishape : tuple[int, ...]
        input image shape
    prior_image : Union[np.ndarray, cp.ndarray]
        the prior image used to calcuate the joint gradient field for the projection

    Returns
    -------
    sigpy.linop.Linop
    """

    # normalized "normal" gradient operator
    G = (1 / np.sqrt(4 * len(ishape))) * sigpy.linop.FiniteDifference(
        ishape, axes=None)

    xi = G(prior_image)

    # normalize the real and imaginary part of the joint gradient field
    real_norm = cp.sqrt(cp.linalg.norm(xi.real, axis=0)**2 + eta**2)
    imag_norm = cp.sqrt(cp.linalg.norm(xi.imag, axis=0)**2 + eta**2)

    ir = cp.where(real_norm > 0)
    ii = cp.where(imag_norm > 0)

    for i in range(xi.shape[0]):
        xi[i, ...].real[ir] /= real_norm[ir]
        xi[i, ...].imag[ii] /= imag_norm[ii]

    M = sigpy.linop.Multiply(G.oshape, xi)
    S = sigpy.linop.Sum(M.oshape, (0, ))
    I = sigpy.linop.Identity(M.oshape)

    # projection operator
    P = I - (M.H * S.H * S * M)

    # projected gradient operator
    PG = P * G

    return PG

