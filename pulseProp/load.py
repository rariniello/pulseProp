import os

import numpy as np
import h5py
import json

import fourierProp as fp
from pulseProp import getFpPath


def loadSimulation(savePath: str | os.PathLike):
    """Loads the simulation input parameters from disk."""
    filename = os.path.join(savePath, "simulation.json")
    with open(filename, "r") as fp:
        parameters = json.load(fp)
    return parameters


def loadPulse(savePath: str | os.PathLike):
    filename = os.path.join(savePath, "pulse.h5")
    f = h5py.File(filename, "r")
    attrs, data = fp.load.loadGroup(f, "pulse")
    f.close()
    return attrs, data


def loadZAtPlane(savePath, ind=None, name=None):
    fpPath = getFpPath(savePath)
    if ind is not None:
        pass
    elif name is not None:
        ind = fp.load.getPlaneIndexFromName(fpPath, name)
    else:
        raise RuntimeError("Must specify either ind or planes arguments.")
    attrs, data = fp.load.loadPlane(fpPath, ind)
    return attrs["z"]
