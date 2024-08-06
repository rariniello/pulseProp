import os

import numpy as np
from numpy.fft import fftfreq, fftshift, ifftshift

import fourierProp as fp
from pulseProp import load, getFpPath


def reconstructInitialPulse(savePath):
    """Calculates the temporal structure of the pulse from the Fourier coefficients.

    This function will only work if the pulse is seperable.
    """
    pulseAttrs, pulseData = load.loadPulse(savePath)

    Nt = pulseAttrs["Nt"]
    Nf = pulseAttrs["Nf"]
    U_t = pulseData["U_t"]
    k_prop = pulseData["k_prop"]

    # Reconstruct pulse from Fourier components
    E_tRecon = np.zeros(Nt, dtype="complex128")
    m = np.arange(Nt)
    for i in range(Nf):
        E_tRecon += (1 / Nt) * U_t[i] * np.exp(1j * 2 * np.pi * k_prop[i] / Nt * m)

    return E_tRecon


def reconstructXTFieldAtPlane(savePath, t_0, ind=None, name=None):
    """Calculate the electric field on the XT plane by summing the field at different frequencies.

    Currently only works for pulses created from a temporal array
    """
    pulseAttrs, pulseData = load.loadPulse(savePath)
    Nt = pulseAttrs["Nt"]
    Nf = pulseAttrs["Nf"]
    f_t = pulseData["f_t"]
    k_prop = pulseData["k_prop"]
    phi_t0 = 2 * np.pi * f_t * t_0

    fpPath = getFpPath(savePath)
    attrs, data = fp.load.loadGridAtPlane(fpPath, ind=ind, name=name)
    Nx = attrs["x"]["Nx"]
    Ny = attrs["y"]["Ny"]
    x = data["x"]["x"]

    # Sum the fields at each frequency
    m = np.arange(Nt)
    E_total = np.zeros((Nx, Nt), dtype="complex128")
    for i in range(Nf):
        path_i = getFpPath(savePath, i)
        E_i = fp.load.loadXFieldAtPlane(path_i, ind=ind, name=name)
        E_total += (
            (1 / Nt)
            * E_i[:, None]
            * np.exp(1j * 2 * np.pi * k_prop[i] / Nt * m[None, :])
            * np.exp(-1j * phi_t0[i])
        )
    return E_total


def reconstructOnAxisAtPlane(savePath, t_0, ind=None, name=None):
    pulseAttrs, pulseData = load.loadPulse(savePath)
    Nt = pulseAttrs["Nt"]
    Nf = pulseAttrs["Nf"]
    f_t = pulseData["f_t"]
    k_prop = pulseData["k_prop"]
    phi_t0 = 2 * np.pi * f_t * t_0

    fpPath = getFpPath(savePath)
    attrs, data = fp.load.loadGridAtPlane(fpPath, ind=ind, name=name)
    Nx = attrs["x"]["Nx"]
    Ny = attrs["y"]["Ny"]

    # Sum the fields at each frequency
    m = np.arange(Nt)
    E_total = np.zeros(Nt, dtype="complex128")
    for i in range(Nf):
        path_i = getFpPath(savePath, i)
        E_i = fp.load.loadOnAxisFieldAtPlane(path_i, ind=ind, name=name)
        E_total += (
            (1 / Nt)
            * E_i
            * np.exp(1j * 2 * np.pi * k_prop[i] / Nt * m)
            * np.exp(-1j * phi_t0[i])
        )
    return E_total


def reconstructFourierSpaceAtPlane(savePath, t_0, ind=None, name=None):
    pulseAttrs, pulseData = load.loadPulse(savePath)
    Nt = pulseAttrs["Nt"]
    Nf = pulseAttrs["Nf"]
    f_t = pulseData["f_t"]
    k_prop = pulseData["k_prop"]
    phi_t0 = 2 * np.pi * f_t * t_0

    fpPath = getFpPath(savePath)
    attrs, data = fp.load.loadGridAtPlane(fpPath, ind=ind, name=name)
    Nx = attrs["x"]["Nx"]
    Ny = attrs["y"]["Ny"]
    x = data["x"]["x"]

    # Sum the fields at each frequency
    U_total = np.zeros((Nx, Nf), dtype="complex128")
    for i in range(Nf):
        path_i = getFpPath(savePath, i)
        E_i = fp.load.loadXFieldAtPlane(path_i, ind=ind, name=name)
        # XXX This only works if cylSymmetry is turned on
        U_i = fftshift(np.fft.fft(E_i))
        U_total[:, i] = U_i * np.exp(-1j * phi_t0[i])
        # U_i = fftshift(np.fft.fft(E_i))
        # U_total[:, i] = U_i[:, int(Ny / 2)] * np.exp(-1j * phi_t0[i])
    return U_total
