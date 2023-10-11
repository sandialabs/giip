import torch

import GIIP.distance
import GIIP.inner_product
import GIIP.util

def pairwiseSquared_precalculated(giipAB, giipAA, giipBB):
    """
    Args:
        giipAB: {nWeights?}*{...A}*{...B}
        giipAA: {nWeights?}*{...A}
        giipBB: {nWeights?}*{...B}

    Returns:
        : {nWeights?}*{...}
    """

    multiWeights = (giipAB.dim() != giipAA.dim() + giipBB.dim())
    if not multiWeights:
        giipAB = giipAB.unsqueeze(0)
        giipAA = giipAA.unsqueeze(0)
        giipBB = giipBB.unsqueeze(0)

    giipAA, giipBB = GIIP.util.pairwiseToElementwise(1, 0, giipAA, giipBB)
    dSqr = GIIP.distance.elementwiseSquared_precalculated(giipAB, giipAA, giipBB)
    
    if not multiWeights: dSqr = dSqr.squeeze(dim=0)
    return dSqr

def pairwiseSquared(positionsA, positionsB, weightsA, weightsB, sigma, giipAA=None, giipBB=None):
    """Compute GIIP distances pairwise between configurations A and configurations B.
    Args:
        positionsA: {...A}*nAtomsA*nDims
        positionsB: {...B}*nAtomsB*nDims
        weightsA: {nWeights?}*{...A}*nAtomsA
        weightsB: {nWeights?}*{...B}*nAtomsB
        sigma: float | nWeights
        giipAA: None | {nWeights?}*{...A}
        giipBB: None | {nWeights?}*{...B}

    Returns:
        : {nWeights?}*{...A}*{...B}
    """
    
    if giipAA is None: giipAA = GIIP.inner_product.self(positionsA, weightsA, sigma)
    if giipBB is None: giipBB = GIIP.inner_product.self(positionsB, weightsB, sigma)
    giipAB = GIIP.inner_product.pairwise(positionsA, positionsB, weightsA, weightsB, sigma)
    return pairwiseSquared_precalculated(giipAB, giipAA, giipBB)

if __name__=='__main__':
    for dtype in [torch.float32, torch.float64]:
        for nDims in [2,3]:
            for neighborhoodShapeA in [[],[4],[4,5]]:
                for neighborhoodShapeB in [[],[3],[2,6]]:
                    for device in ['cuda','cpu']:
                        for multiWeights in [True,False]:
                            for nWeights in [1,3]:
                                for multiSigmas in ['singletonTensor','float','tensor']:
                                    for precalculateSelfGiips in [True, False]:

                                        nAtomsA = 6
                                        nAtomsB = 7

                                        positionsAShape = neighborhoodShapeA + [nAtomsA, nDims]
                                        positionsBShape = neighborhoodShapeB + [nAtomsB, nDims]
                                        if multiWeights:
                                            weightsBShape = [nWeights] + neighborhoodShapeB + [nAtomsB]
                                            weightsAShape = [nWeights] + neighborhoodShapeA + [nAtomsA]
                                        else:
                                            weightsBShape = neighborhoodShapeB + [nAtomsB]
                                            weightsAShape = neighborhoodShapeA + [nAtomsA]
                                        
                                        positionsA = torch.rand(positionsAShape, device=device, dtype=dtype)
                                        positionsB = torch.rand(positionsBShape, device=device, dtype=dtype)
                                        weightsA = torch.rand(weightsAShape, device=device, dtype=dtype)
                                        weightsB = torch.rand(weightsBShape, device=device, dtype=dtype)
                                        
                                        if multiSigmas=='tensor' and multiWeights:
                                            sigma = torch.rand((nWeights,), device=device, dtype=dtype)
                                        else:
                                            if multiSigmas=='singletonTensor':
                                                sigma=torch.tensor([0.1], device=device, dtype=dtype)
                                            elif multiSigmas=='float':
                                                sigma=0.1

                                        if precalculateSelfGiips:
                                            giipAA = GIIP.inner_product.self(positionsA, weightsA, sigma)
                                            giipBB = GIIP.inner_product.self(positionsB, weightsB, sigma)
                                            ret = pairwiseSquared(positionsA, positionsB, weightsA, weightsB, sigma, giipAA, giipBB)
                                        else:
                                            ret = pairwiseSquared(positionsA, positionsB, weightsA, weightsB, sigma)
                                        
                                        print(positionsA.shape, positionsB.shape, sigma, ret.shape)
