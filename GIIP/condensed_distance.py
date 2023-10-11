import torch

import GIIP.distance
import GIIP.inner_product

def condenseDistanceSqr(condensationMethod, distanceSqr):
    if condensationMethod=='L2':
        giipDistanceSquaredCondensed = torch.sum(distanceSqr, dim=0)
    elif condensationMethod=='L1':
        giipDistanceSquaredCondensed = torch.sum(torch.sqrt(distanceSqr), dim=0)
        giipDistanceSquaredCondensed *= giipDistanceSquaredCondensed
    elif condensationMethod=='max':
        giipDistanceSquaredCondensed = torch.max(distanceSqr, dim=0).values
    elif condensationMethod=='meanSquare':
        giipDistanceSquaredCondensed = torch.mean(distanceSqr, dim=0)
    return giipDistanceSquaredCondensed

def pairwise(condensationMethod, positionsA, positionsB, weightsA, weightsB, sigma, giipAA=None, giipBB=None):
    """Compute CONDENSED GIIP distances pairwise between configurations A and configurations B.
    Args:
        condensationMethod: 'L2' | 'L1' | 'max' | 'meanSquare'
        positionsA: {...A}*nAtomsA*nDims
        positionsB: {...B}*nAtomsB*nDims
        weightsA: {nWeights?}*{...A}*nAtomsA
        weightsB: {nWeights?}*{...B}*nAtomsB
        sigma: float | nWeights
        giipAA: None | {nWeights?}*{...A}
        giipBB: None | {nWeights?}*{...B}

    Returns:
        : {...A}*{...B}
    """

    multiWeights = (weightsA.dim() == positionsA.dim())
    if multiWeights: nWeights=weightsA.shape[0]
    else: nWeights = 1

    if (nWeights==1): return GIIP.distance.pairwiseSquared(positionsA, positionsB, weightsA, weightsB, sigma, giipAA, giipBB)

    distSquared_pairwise = GIIP.distance.pairwiseSquared(positionsA, positionsB, weightsA, weightsB, sigma, giipAA, giipBB)
    distSquared_pairwise_condensed = condenseDistanceSqr(condensationMethod, distSquared_pairwise)
    return distSquared_pairwise_condensed

def elementwise(condensationMethod, positionsA, positionsB, weightsA, weightsB, sigma, giipAA=None, giipBB=None):
    """Compute CONDENSED GIIP distances pairwise between configurations A and configurations B.
    Args:
        condensationMethod: 'L2' | 'L1' | 'max' | 'meanSquare'
        positionsA: {...}*nAtomsA*nDims
        positionsB: {...}*nAtomsB*nDims
        weightsA: {nWeights?}*{...}*nAtomsA
        weightsB: {nWeights?}*{...}*nAtomsB
        sigma: float | nWeights
        giipAA: None | {nWeights?}*{...}
        giipBB: None | {nWeights?}*{...}

    Returns:
        : {...}
    """
    
    multiWeights = (weightsA.dim() == positionsA.dim())
    if multiWeights: nWeights=weightsA.shape[0]
    else: nWeights = 1

    if (nWeights==1): return GIIP.distance.elementwiseSquared(positionsA, positionsB, weightsA, weightsB, sigma, giipAA, giipBB)

    distSquared_elementwise = GIIP.distance.elementwiseSquared(positionsA, positionsB, weightsA, weightsB, sigma, giipAA, giipBB)
    distSquared_elementwise_condensed = condenseDistanceSqr(condensationMethod, distSquared_elementwise)
    return distSquared_elementwise_condensed
     

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
                                        for condensationMethod in ['L2', 'L1', 'max', 'meanSquare']:

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
                                                ret = pairwise(condensationMethod, positionsA, positionsB, weightsA, weightsB, sigma, giipAA, giipBB)
                                            else:
                                                ret = pairwise(condensationMethod, positionsA, positionsB, weightsA, weightsB, sigma)
                                            
                                            print(positionsA.shape, positionsB.shape, sigma, ret.shape)
