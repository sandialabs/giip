import torch

import GIIP.distance
import GIIP.inner_product
import GIIP.util

def elementwiseSquared_precalculated(giipAB, giipAA, giipBB):
    """
    Args:
        giipAB: {nWeights?}*{...}
        giipAA: {nWeights?}*{...}
        giipBB: {nWeights?}*{...}

    Returns:
        : {nWeights?}*{...}
    """

    out = giipAB * -2
    out += giipAA
    out += giipBB
    return out

def elementwiseSquared(positionsA, positionsB, weightsA, weightsB, sigma, giipAA=None, giipBB=None):
    """
    Args:
        positionsA: {...}*nAtomsA*nDims
        positionsB: {...}*nAtomsB*nDims
        weightsA: {nWeights?}*{...}*nAtomsA
        weightsB: {nWeights?}*{...}*nAtomsB
        sigma: float | nWeights
        giipAA: None | {nWeights?}*{...}
        giipBB: None | {nWeights?}*{...}

    Returns:
        : {nWeights?}*{...A}*{...B}
    """

    if giipAA is None: giipAA = GIIP.inner_product.self(positionsA, weightsA, sigma)
    if giipBB is None: giipBB = GIIP.inner_product.self(positionsB, weightsB, sigma)
    giipAB = GIIP.inner_product.elementwise(positionsA, positionsB, weightsA, weightsB, sigma)
    return elementwiseSquared_precalculated(giipAB, giipAA, giipBB)
