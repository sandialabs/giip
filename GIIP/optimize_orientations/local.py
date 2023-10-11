import itertools
import math
import torch

import GIIP.condensed_distance
import GIIP.inner_product
import GIIP.optimize_orientations

def pairwise(
    positionsA, positionsB,
    weightsA, weightsB,
    sigma,
    rotationMatrices,
    condensationMethod,
    minimalRotationMatricesCoarse,
    giipAA=None, giipBB=None
    ):
    """
    Args:
        positionsA: nNeighborhoodsA*nAtomsA*nDims
        positionsB: nNeighborhoodsB*nAtomsB*nDims
        weightsA: {nWeights?}*nNeighborhoodsA*nAtomsA
        weightsB: {nWeights?}*nNeighborhoodsB*nAtomsB
        sigma: float | nWeights
        rotationMatrices: nRotations*nDims*nDims
        condensationMethod: string
        minimalRotationMatricesCoarse: nNeighborhoodsA*nNeighborhoodsB*nDims*nDims
        giipAA: {nWeights?}*nNeighborhoodsA
        giipBB: {nWeights?}*nNeighborhoodsB

    Returns:
        minimalDistances: nNeighborhoodsA*nNeighborhoodsB
        minimalRotations: nNeighborhoodsA*nNeighborhoodsB*nDims*nDims
    """
    
    if giipAA is None: giipAA = GIIP.inner_product.self(positionsA, weightsA, sigma)
    if giipBB is None: giipBB = GIIP.inner_product.self(positionsB, weightsB, sigma)

    nRotations = rotationMatrices.shape[0]
    nDims = positionsA.shape[-1]

    multiWeights = (positionsA.dim()==weightsA.dim())
    if not multiWeights:
        weightsA = weightsA.unsqueeze(0)
        weightsB = weightsB.unsqueeze(0)
        giipAA = giipAA.unsqueeze(0)
        giipBB = giipBB.unsqueeze(0)
    nWeights = weightsA.shape[0]
    nNeighborhoodsA = weightsA.shape[1]
    nAtomsA = weightsA.shape[-1]
    nNeighborhoodsB = weightsB.shape[1]
    nAtomsB = weightsB.shape[-1]

    positionsAExpanded = positionsA.unsqueeze(1).unsqueeze(1).expand([nNeighborhoodsA, nNeighborhoodsB, nRotations, nAtomsA, nDims]) # nNeighborhoodsA*nNeighborhoodsB*nRotations*nAtomsA*nDims
    weightsAExpanded = weightsA.unsqueeze(-2).unsqueeze(-2).expand([nWeights, nNeighborhoodsA, nNeighborhoodsB, nRotations, nAtomsA]) # nWeights*nNeighborhoodsA*nNeighborhoodsB*nRotations*nAtomsA
    giipAAExpanded = giipAA.unsqueeze(-1).unsqueeze(-1).expand([nWeights, nNeighborhoodsA, nNeighborhoodsB, nRotations]) # nWeights*nNeighborhoodsA*nNeighborhoodsB*nRotations

    positionsBExpanded = torch.einsum('rkj,abji,bni->abrnk', rotationMatrices, minimalRotationMatricesCoarse, positionsB) #nNeighborhoodsA*nNeighborhoodsB*nRotations*nAtomsB*nDims
    weightsBExpanded = weightsB.unsqueeze(1).unsqueeze(-2).expand([nWeights, nNeighborhoodsA, nNeighborhoodsB, nRotations, nAtomsB]) # nWeights*nNeighborhoodsA*nNeighborhoodsB*nRotations*nAtomsB
    giipBBExpanded = giipBB.unsqueeze(1).unsqueeze(-1).expand([nWeights, nNeighborhoodsA, nNeighborhoodsB, nRotations]) # nWeights*nNeighborhoodsA*nNeighborhoodsB*nRotations

    distanceSquaredCondensed = GIIP.condensed_distance.elementwise(
        condensationMethod,
        positionsAExpanded, positionsBExpanded,
        weightsAExpanded, weightsBExpanded,
        sigma,
        giipAAExpanded, giipBBExpanded
        ) # nNeighborhoodsA*nNeighborhoodsB*nRotations
    
    minSoln = torch.min(distanceSquaredCondensed, dim=-1)
    minimalDistances = minSoln.values
    minimalIndices = minSoln.indices
    minimalRotationMatrices = rotationMatrices[minimalIndices]
    minimalRotationMatrices = torch.einsum('abij,abjk->abik',minimalRotationMatrices,minimalRotationMatricesCoarse)
    return (minimalDistances, minimalRotationMatrices)
    
if __name__=='__main__' and True:
    device='cuda'
    multiWeights = True
    multiSigmas = 'float'
    nDims = 3
    dtype = torch.float32
    neighborhoodShapeA = [5]
    neighborhoodShapeB = [4]
    nAtomsA = 7
    nAtomsB = 8
    nWeights = 3
    condensationMethod = 'L2'

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
    
    rotationMatrices = GIIP.optimize_orientations.getRotationMatrices(5,device=device)
    minimalValuesCoarse, minimalRotationsCoarse = GIIP.optimize_orientations.exhaustive.pairwise(positionsA, positionsB, weightsA, weightsB, sigma, rotationMatrices, condensationMethod)

    rotationMatricesFine = GIIP.optimize_orientations.getRotationMatricesRefined(1, 5, True, device=device)
    minimalValuesFine, minimalRotationsFine = GIIP.optimize_orientations.local.pairwise(positionsA, positionsB, weightsA, weightsB, sigma, rotationMatricesFine, condensationMethod, minimalRotationsCoarse)

    print(minimalValuesCoarse)
    print(minimalValuesFine)
    print(minimalValuesCoarse - minimalValuesFine)