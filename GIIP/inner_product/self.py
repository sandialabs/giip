import torch

import GIIP.inner_product.elementwise

def self(positions, weights, sigma):
    """Compute Gaussian Integral Inner Products with self

    Args:
        positions: {...}*nAtoms*nDims
        weights: {nWeights?}*{...}*nAtoms
        sigma: float | nWeights

    Returns:
        : {nWeights?}*{...}
    """
    
    return GIIP.inner_product.elementwise(positions, positions, weights, weights, sigma)


if __name__=='__main__':
    for dtype in [torch.float32, torch.float64]:
        for nDims in [2,3]:
            for neighborhoodShape in [[],[4],[4,5]]:
                    for device in ['cuda','cpu']:
                        for multiWeights in [True,False]:
                            for nWeights in [1,3]:
                                for multiSigmas in ['singletonTensor','float','tensor']:

                                    nAtoms = 6

                                    positionsShape = neighborhoodShape + [nAtoms, nDims]
                                    if multiWeights:
                                        weightsShape = [nWeights] + neighborhoodShape + [nAtoms]
                                    else:
                                        weightsShape = neighborhoodShape + [nAtoms]
                                    
                                    positions = torch.rand(positionsShape, device=device, dtype=dtype)
                                    weights = torch.rand(weightsShape, device=device, dtype=dtype)
                                    
                                    if multiSigmas=='tensor' and multiWeights:
                                        sigma = torch.rand((nWeights,), device=device, dtype=dtype)
                                    else:
                                        if multiSigmas=='singletonTensor':
                                            sigma=torch.tensor([0.1], device=device, dtype=dtype)
                                        elif multiSigmas=='float':
                                            sigma=0.1
                                    
                                    ret = self(positions, weights, sigma)

                                    print(positions.shape, weights.shape, sigma, ret.shape)
