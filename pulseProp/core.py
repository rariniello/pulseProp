import os
from datetime import datetime
import numpy as np
import h5py
import pyfftw
import logging
import json

import fourierProp as fp

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def pulsePropagate(
    pulse,
    planes: list,
    savePath: str | os.PathLike,
    threads: int = 1,
    cylSymmetry: bool = False,
):
    """Performs the Fourier optics simulation of the pulse.

    Propagates each frequency from the source to the to each subsequent plane using Fourier optics.

    Args:
        pulse: Pulse object defining the initial pulse.
        planes: List of planes and volumes to the simulate the propagation through.
        savePath: Path to save the output files to.
        threads: Number of threads to run the FFTs on.
        cylSymmetry: Set to True to only save a 1D lineout of the field at each plane, defaults to False.
    """
    startTime = datetime.now()
    N = pulse.Nf
    createSimulationDirectory(savePath)
    saveSimulation(savePath, cylSymmetry)
    savePulse(pulse, savePath)
    logger.info("Starting pulse propagation simulation.")

    for i in range(N):
        path_i = getFpPath(savePath, i)
        logger.info("Propagating frequency {} of {}.".format(i + 1, N))
        # Set the source field to the field for the current frequency
        # in fourierProp, source planes recieve lam as an argument, the source can be implemented there
        pulse.setSourceField(i)
        # Go through the planes and modify any that are frequency dependent
        fp.fourierPropagate(
            pulse.lam_t[i], planes, path_i, threads=threads, cylSymmetry=cylSymmetry
        )
        logger.info("Finished propagating frequency {} of {}.\n".format(i + 1, N))

    logger.info("Finished pulse propagation simulation")
    endTime = datetime.now()
    logger.info("Total simulation time was {}".format(endTime - startTime))


def createSimulationDirectory(savePath):
    if not os.path.exists(savePath):
        os.makedirs(savePath)


def saveSimulation(savePath, cylSymmetry):
    """Saves simulation input parameters to a file.

    Args:
        lam: Wavelength of the light in vacuum [m].
        cylSymmetry: Whether the simulation saves only the y=0 slice.
    """
    filename = os.path.join(savePath, "simulation.json")
    logger.info(f"Saving simulation input parameters.")
    parameters = {"cylSymmetry": cylSymmetry}
    with open(filename, "w") as fp:
        json.dump(parameters, fp, indent=4)


def savePulse(pulse, savePath: str | os.PathLike):
    """Saves information about the pulse to a file.

    Args:
        pulse: Pulse object to save to file.
        savePath: Path to save the file at.
    """
    # TODO move directory creation into its own function and call it in pulsePropagate
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    filename = os.path.join(savePath, "pulse.h5")
    logger.info(f"Saving pulse information to {filename}")
    with h5py.File(filename, "w") as f:
        fp.core.saveObject(f, pulse, "pulse", 0)

    logger.info("Finished saving pulse information")


def getFpPath(savePath, i=0):
    return os.path.join(savePath, "f_{:02d}".format(i))
