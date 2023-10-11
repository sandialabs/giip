import torch

def _elementwise_multiWeights_singleSigma(positionsA, positionsB, weightsA, weightsB, sigma):
    """Compute Gaussian Integral Inner Products between configurations A and configurations B, elementwise.

    Args:
        positionsA: {...}*nAtomsA*nDims
        positionsB: {...}*nAtomsB*nDims
        weightsA: nWeights*{...}*nAtomsA
        weightsB: nWeights*{...}*nAtomsB
        sigma: float

    Returns:
        : nWeights*{...}
    """

    temp0 = torch.cdist(positionsA, positionsB) # {...}*nAtomsA*nAtomsB; |xA-xB|
    temp0 *= temp0 # {...}*nAtomsA*nAtomsB; |xA-xB|^2
    temp0 /= (-4*sigma*sigma) # {...}*nAtomsA*nAtomsB; -|xA-xB|^2 / (4*sigma*sigma)
    temp0 = torch.exp_(temp0) # {...}*nAtomsA*nAtomsB; exp(-|xA-xB|^2 / (4*sigma*sigma))
    temp0 = torch.einsum('w...a,...ab,w...b->w...',weightsA, temp0, weightsB) # nWeights*{...}; sum_AB wA*wB*exp(-|xA-xB|^2 / (4*sigma*sigma))
    return temp0

def _elementwise_singleWeight_singleSigma(positionsA, positionsB, weightsA, weightsB, sigma):
    """Compute Gaussian Integral Inner Products between configurations A and configurations B, elementwise.

    Args:
        positionsA: {...}*nAtomsA*nDims
        positionsB: {...}*nAtomsB*nDims
        weightsA: {...}*nAtomsA
        weightsB: {...}*nAtomsB
        sigma: float

    Returns:
        : {...}
    """

    weightsA = weightsA.unsqueeze(0)
    weightsB = weightsB.unsqueeze(0)
    return _elementwise_multiWeights_singleSigma(positionsA, positionsB, weightsA, weightsB, sigma).squeeze(dim=0)

def elementwise(positionsA, positionsB, weightsA, weightsB, sigma):
    """Compute Gaussian Integral Inner Products between configurations A and configurations B, elementwise.

    Args:
        positionsA: {...}*nAtomsA*nDims
        positionsB: {...}*nAtomsB*nDims
        weightsA: {nWeights?}*{...}*nAtomsA
        weightsB: {nWeights?}*{...}*nAtomsB
        sigma: float | nWeights

    Returns:
        : {nWeights?}*{...}
    """
    
    multiWeights = weightsA.dim() == positionsA.dim()

    if isinstance(sigma, torch.Tensor) and multiWeights: # there are nWeights sigmas, and nWeights sets of weights
        nWeights = len(sigma)
        ellpsis = list(positionsA.shape[:-2])
        out = torch.zeros(size=([nWeights] + ellpsis), device=positionsA.device, dtype=positionsA.dtype)
        for iWeight, sigmaSingle in enumerate(list(sigma)):
            out[iWeight] = _elementwise_singleWeight_singleSigma(positionsA, positionsB, weightsA[iWeight], weightsB[iWeight], sigmaSingle)
        return out
    elif multiWeights: # there is only one sigma, but multiple sets of weights
        return _elementwise_multiWeights_singleSigma(positionsA, positionsB, weightsA, weightsB, sigma)
    else: # there is only one sigma, and only one set of weights
        return _elementwise_singleWeight_singleSigma(positionsA, positionsB, weightsA, weightsB, sigma)

if __name__=='__main__':
    for dtype in [torch.float32, torch.float64]:
        for nWeights in [1,3]:
            for nDims in [2,3]:
                for neighborhoodShape in [[],[4],[4,5]]:
                    for device in ['cuda','cpu']:
                        for multiWeights in [True,False]:
                            for multiSigmas in ['singletonTensor','float','tensor']:

                                nAtomsA = 6
                                nAtomsB = 7

                                positionsAShape = neighborhoodShape + [nAtomsA, nDims]
                                positionsBShape = neighborhoodShape + [nAtomsB, nDims]
                                if multiWeights:
                                    weightsBShape = [nWeights] + neighborhoodShape + [nAtomsB]
                                    weightsAShape = [nWeights] + neighborhoodShape + [nAtomsA]
                                else:
                                    weightsBShape = neighborhoodShape + [nAtomsB]
                                    weightsAShape = neighborhoodShape + [nAtomsA]
                                
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
                                
                                ew = elementwise(positionsA, positionsB, weightsA, weightsB, sigma)

                                print(positionsA.shape, positionsB.shape, ew.shape)
