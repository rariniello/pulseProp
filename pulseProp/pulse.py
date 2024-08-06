import numpy as np
from numpy.fft import fftfreq, fftshift, ifftshift
from scipy.constants import physical_constants
import pyfftw

c = physical_constants["speed of light in vacuum"][0]


class Pulse:
    """Describes a laser pulse as a set of plane waves of different frequencies.

    Available behavior depends on the type of source specified.

    Pulse object created from a seperable pulse E(x, y, t)=E_xy(x, y)E_t(t) on a (x, y, t) grid.
    Can be made using a source that is independent of wavelength. In this case source.E = E_xy(x, y).
    Two options are available for specifiying the pulse:
        Specify E_t(t) and the pulse class will handle the Fourier transform and amplitude.
        Specify U_t(f) directly.

    Attributes:
        Nf: Number of frequency components used to describe the pulse.
        lam_t: Array of wavelengths in vacuum corresponding to each frequency component.
    """

    def __init__(self, source, lam_0, name, Df_t, t=None, E_t=None, Nf=None, threads=1):
        if not source.isSource():
            raise TypeError(
                "source must be an instance of Source or one of its subclasses."
            )

        self.source = source
        self.lam_0 = lam_0
        self.name = name
        self.f_0 = c / lam_0
        self.Df_t = Df_t
        self.threads = threads

        if source.isIndependentOfWavelength():
            if t is not None and E_t is not None:
                self.U_t = self.frequenciesFromTemporalPulse(t, E_t)
                self.E_xy = source.E
                self.t = t
                self.E_t = E_t
        elif not source.isIndependentOfWavelength():
            if Nf is not None:
                self.createFrequencyGrid(Nf)
        self.lam_t = c / self.f_t

    def setSourceField(self, i):
        """Modifies the source field as a function of wavelength if required.

        If the source field is independent of wavelength, then this function multiplies
        the field by the Fourier coefficient for the current wavelength. Does nothing if
        the source depends on the wavelength.
        """
        if self.source.isIndependentOfWavelength():
            self.source.E = self.U_t[i] * self.E_xy
        elif not self.source.isIndependentOfWavelength():
            return

    def frequenciesFromTemporalPulse(self, t, E_t):
        """Calculates the frequencies to propagate from the Fourier transform of the passed field.

        Args:
            t: Temporal grid the pulse is defined along.
            E_t: Temporal dependence of the electric field.

        Returns:
            U_t: Fourier amplitude of the frequency components that are propagated.
        """
        f_0 = self.f_0
        Df_t = self.Df_t

        self.Nt = len(t)
        self.dt = t[1] - t[0]
        # Frequencies of the DFT unshifted
        ft_DFT = fftfreq(self.Nt, self.dt)
        sel = (ft_DFT < -f_0 + Df_t) * (ft_DFT > -f_0 - Df_t)
        # Frequencies to propagate
        self.f_t = -ft_DFT[sel]
        self.Nf = len(self.f_t)

        # Required for reconstructing the pulse from the Fourier amplitudes
        self.k_prop = np.arange(self.Nt)[sel]

        U_t = self.calculateFieldCoefficients(E_t)
        return U_t[sel]

    def createFrequencyGrid(self, Nf):
        f_0 = self.f_0
        self.Nf = Nf
        Df_t = self.Df_t
        self.f_t, self.dft = np.linspace(
            f_0 - Df_t, f_0 + Df_t, Nf, False, dtype="double", retstep=True
        )
        # XXX The simulation might work, this is not implemented in reconstruct
        raise NotImplementedError("This is not tested, but might work")

    def calculateFieldCoefficients(self, E_t):
        """Fourier transforms E_t(t) to get the Fourier coefficients.

        Args:
            E_t: Temporal dependence of the electric field.

        Returns:
            U_t: Fourier transform of the pulse.
        """
        # Create the FFt plan
        threads = self.threads
        efft = pyfftw.empty_aligned((self.Nt), dtype="complex128")
        fft_t = pyfftw.builders.fftn(
            efft, overwrite_input=True, avoid_copy=True, threads=threads, axes=(0,)
        )
        U_t = fft_t(E_t)
        return U_t

    def getSaveData(self) -> tuple[dict, dict]:
        """Creates two dictonaries fully describing the plane.

        Neither dictionary should be nested, i.e., they should only be one layer deep.

        Returns:
            Two dict, the first is attributes describing the plane. The second is any data
            that is larger than a simple attribute that must be saved to fully define the plane.
        """
        attr = {
            "name": self.name,
            "source": self.source.name,
            "lam_0": self.lam_0,
            "f_0": self.f_0,
            "Df_t": self.Df_t,
            "Nt": self.Nt,
            "Nf": self.Nf,
            "dt": self.dt,
        }
        data = {"f_t": self.f_t, "lam_t": self.lam_t}
        if self.source.isIndependentOfWavelength():
            data["t"] = self.t
            data["E_t"] = self.E_t
            data["U_t"] = self.U_t
            data["k_prop"] = self.k_prop
        return attr, data
