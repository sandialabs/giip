import torch

import GIIP.util.reshape
import GIIP.inner_product.elementwise

def pairwise(positionsA, positionsB, weightsA, weightsB, sigma):
    """Compute Gaussian Integral Inner Products pairwise between configurations A and configurations B.

    Args:
        positionsA: {...A}*nAtomsA*nDims
        positionsB: {...B}*nAtomsB*nDims
        weightsA: {nWeights?}*{...A}*nAtomsA
        weightsB: {nWeights?}*{...B}*nAtomsB
        sigma: float | nWeights

    Returns:
        : {nWeights?}*{...A}*{...B}
    """

    multiWeights = weightsA.dim() == positionsA.dim()
    if not multiWeights:
        weightsA = weightsA.unsqueeze(0)
        weightsB = weightsB.unsqueeze(0)

    positionsA,positionsB = GIIP.util.pairwiseToElementwise(0,2,positionsA,positionsB)
    weightsA,weightsB = GIIP.util.pairwiseToElementwise(1,1,weightsA,weightsB)

    ret = GIIP.inner_product.elementwise(positionsA, positionsB, weightsA, weightsB, sigma)
    if not multiWeights: ret = ret.squeeze(dim=0) #eliminate multiWeights, if called for
    return ret

if __name__=='__main__':
    for dtype in [torch.float32, torch.float64]:
        for nDims in [2,3]:
            for neighborhoodShapeA in [[],[4],[4,5]]:
                for neighborhoodShapeB in [[],[3],[2,6]]:
                    for device in ['cuda','cpu']:
                        for multiWeights in [True,False]:
                            for nWeights in [1,3]:
                                for multiSigmas in ['singletonTensor','float','tensor']:

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
                                    
                                    ret = pairwise(positionsA, positionsB, weightsA, weightsB, sigma)

                                    print(positionsA.shape, positionsB.shape, sigma, ret.shape)
