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
    giipAA=None, giipBB=None):
    """
    Args:
        positionsA: nNeighborhoodsA*nAtomsA*nDims
        positionsB: nNeighborhoodsB*nAtomsB*nDims
        weightsA: {nWeights?}*nNeighborhoodsA*nAtomsA
        weightsB: {nWeights?}*nNeighborhoodsB*nAtomsB
        sigma: float | nWeights
        rotationMatrices: nRotations*nDims*nDims
        condensationMethod: string
        giipAA: {nWeights?}*nNeighborhoodsA
        giipBB: {nWeights?}*nNeighborhoodsB

    Returns:
        minimalDistances: nNeighborhoodsA*nNeighborhoodsB
        minimalRotations: nNeighborhoodsA*nNeighborhoodsB*nDims*nDims
    """

    if giipAA is None: giipAA = GIIP.inner_product.self(positionsA, weightsA, sigma)
    if giipBB is None: giipBB = GIIP.inner_product.self(positionsB, weightsB, sigma)

    nRotations = rotationMatrices.shape[0]

    multiWeights = (positionsA.dim()==weightsA.dim())
    if not multiWeights:
        weightsA = weightsA.unsqueeze(0)
        weightsB = weightsB.unsqueeze(0)
        giipAA = giipAA.unsqueeze(0)
        giipBB = giipBB.unsqueeze(0)
    nWeights = weightsA.shape[0]
    nNeighborhoodsB = weightsB.shape[1]
    nAtomsB = weightsB.shape[-1]

    positionsBExpanded = torch.einsum('rij,naj->nrai', rotationMatrices, positionsB) # nNeighborhoodsB*nRotations*nAtomsB*nDims
    weightsBExpanded = weightsB.unsqueeze(-2).expand([nWeights, nNeighborhoodsB, nRotations, nAtomsB]) # nWeights*nNeighborhoodsB*nRotations*nAtomsB
    giipBBExpanded = giipBB.unsqueeze(-1).expand([nWeights, nNeighborhoodsB, nRotations]) # nWeights*nNeighborhoodsB*nRotations

    distanceSquaredCondensed = GIIP.condensed_distance.pairwise(
        condensationMethod,
        positionsA, positionsBExpanded,
        weightsA, weightsBExpanded,
        sigma,
        giipAA, giipBBExpanded
        )
    
    minSoln = torch.min(distanceSquaredCondensed, dim=-1)
    minimalDistances = minSoln.values
    minimalIndices = minSoln.indices
    minimalRotations = rotationMatrices[minimalIndices]

    return (minimalDistances, minimalRotations)

def pairwise_batchR(
    positionsA, positionsB,
    weightsA, weightsB,
    sigma,
    rotationMatrices,
    condensationMethod,
    nBatchesRotation = 1,
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
        nBatchesRotation: None | int
        giipAA: {nWeights?}*nNeighborhoodsA
        giipBB: {nWeights?}*nNeighborhoodsB

    Returns:
        minimalDistances: nNeighborhoodsA*nNeighborhoodsB
        minimalRotations: nNeighborhoodsA*nNeighborhoodsB*nDims*nDims
    """

    if giipAA is None: giipAA = GIIP.inner_product.self(positionsA, weightsA, sigma)
    if giipBB is None: giipBB = GIIP.inner_product.self(positionsB, weightsB, sigma)

    nNeighborhoodsA = positionsA.shape[0]
    nNeighborhoodsB = positionsB.shape[0]
    nDims = positionsA.shape[-1]
    nRotations = rotationMatrices.shape[0]

    nRotationsPerBatch = int(math.ceil(nRotations/nBatchesRotation))

    minimalDistances = torch.zeros((nBatchesRotation,nNeighborhoodsA,nNeighborhoodsB), device=positionsA.device, dtype=positionsA.dtype)
    minimalRotations = torch.zeros((nBatchesRotation,nNeighborhoodsA,nNeighborhoodsB,nDims,nDims), device=positionsA.device, dtype=positionsA.dtype)
    for iBatchRotation in range(nBatchesRotation):
        rotationMatricesBatch = rotationMatrices[(iBatchRotation*nRotationsPerBatch):((iBatchRotation+1)*nRotationsPerBatch)]
        minimalDistances[iBatchRotation], minimalRotations[iBatchRotation] = pairwise(positionsA, positionsB, weightsA, weightsB, sigma, rotationMatricesBatch, condensationMethod, giipAA, giipBB)
    
    minSoln = torch.min(minimalDistances, dim=0)
    minimalDistances = minSoln.values
    minimalIndices = minSoln.indices
    minimalRotationsFinal = torch.zeros((nNeighborhoodsA, nNeighborhoodsB, nDims, nDims), device=minimalDistances.device, dtype=minimalDistances.dtype)
    for i in range(nDims):
        for j in range(nDims):
            minimalRotationsFinal[:,:,i,j] = torch.gather(minimalRotations[:,:,:,i,j], 0, minimalIndices.unsqueeze(0))
    return (minimalDistances, minimalRotationsFinal)

def pairwise_batchABR(
    positionsA, positionsB,
    weightsA, weightsB,
    sigma,
    rotationMatrices,
    condensationMethod = 'L2',
    nBatchesA = 1, nBatchesB = 1,
    batchSizeA = None, batchSizeB = None,
    nBatchesRotation = 1,
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
        nBatchesA: None | int
        nBatchesB: None | int
        batchSizeA: None | int
        batchSizeB: None | int
        nBatchesRotation: None | int
        giipAA: {nWeights?}*nNeighborhoodsA
        giipBB: {nWeights?}*nNeighborhoodsB

    Returns:
        minimalDistances: nNeighborhoodsA*nNeighborhoodsB
        minimalRotations: nNeighborhoodsA*nNeighborhoodsB*nDims*nDims
    """

    if giipAA is None: giipAA = GIIP.inner_product.self(positionsA, weightsA, sigma)
    if giipBB is None: giipBB = GIIP.inner_product.self(positionsB, weightsB, sigma)
    
    def preproc(positions, weights, nBatches, batchSize, giipSelf):
        nNeighborhoods = positions.shape[0]
        nDims = positions.shape[-1]
        multiWeights = (positions.dim() == weights.dim())
        if not multiWeights:
            weights = weights.unsqueeze(0)
            giipSelf = giipSelf.unsqueeze(0)
        if nBatches == None: nBatches = 1
        if batchSize == None: batchSize = int(math.ceil(nNeighborhoods / nBatches))
        else: nBatches = int(math.ceil(nNeighborhoods / batchSize))
        return positions, weights, nBatches, batchSize, giipSelf, nNeighborhoods, nDims
    
    positionsA, weightsA, nBatchesA, batchSizeA, giipAA, nNeighborhoodsA, nDimsA = preproc(
        positionsA, weightsA, nBatchesA, batchSizeA, giipAA)
    positionsB, weightsB, nBatchesB, batchSizeB, giipBB, nNeighborhoodsB, nDimsB = preproc(
        positionsB, weightsB, nBatchesB, batchSizeB, giipBB)

    minimalDistances = torch.zeros((nNeighborhoodsA, nNeighborhoodsB), device=positionsA.device, dtype=positionsA.dtype)
    minimalRotations = torch.zeros((nNeighborhoodsA, nNeighborhoodsB, nDimsA,nDimsA), device=positionsA.device, dtype=positionsA.dtype)
    def getBatchAB(iBatchA, iBatchB):
        """
        Args:
            iBatchA: int
            iBatchB: int

        Returns:
            positionsABatch: nNeighborhoodsBatchA*nAtomsA*nDims
            positionsBBatch: nNeighborhoodsBatchB*nAtomsB*nDims
            weightsABatch: nWeights*nNeighborhoodsBatchA*nAtoms
            weightsBBatch: nWeights*nNeighborhoodsBatchB*nAtoms
            giipAABatch: nWeights*nNeighborhoodsBatchA
            giipBBBatch: nWeights*nNeighborhoodsBatchB
            minimalDistancesBatch: nNeighborhoodsBatchA*nNeighborhoodsBatchB
            minimalRotationsBatch: nNeighborhoodsBatchA*nNeighborhoodsBatchB*3*3
        """

        positionsABatch = positionsA[(iBatchA*batchSizeA):((iBatchA+1)*batchSizeA)]
        weightsABatch = weightsA[:,(iBatchA*batchSizeA):((iBatchA+1)*batchSizeA)]
        giipAABatch = giipAA[:,(iBatchA*batchSizeA):((iBatchA+1)*batchSizeA)]
        
        positionsBBatch = positionsB[(iBatchB*batchSizeB):((iBatchB+1)*batchSizeB)]
        weightsBBatch = weightsB[:,(iBatchB*batchSizeB):((iBatchB+1)*batchSizeB)]
        giipBBBatch = giipBB[:,(iBatchB*batchSizeB):((iBatchB+1)*batchSizeB)]

        return (positionsABatch, positionsBBatch, 
            weightsABatch, weightsBBatch, 
            giipAABatch, giipBBBatch)

    for iBatchA in range(nBatchesA):
        for iBatchB in range(nBatchesB):

            (positionsABatch, positionsBBatch, 
            weightsABatch, weightsBBatch, 
            giipAABatch, giipBBBatch) = getBatchAB(iBatchA, iBatchB)

            (minimalDistances[(iBatchA*batchSizeA):((iBatchA+1)*batchSizeA), (iBatchB*batchSizeB):((iBatchB+1)*batchSizeB)], 
            minimalRotations[(iBatchA*batchSizeA):((iBatchA+1)*batchSizeA), (iBatchB*batchSizeB):((iBatchB+1)*batchSizeB), :,:]) = pairwise_batchR(
                positionsABatch, positionsBBatch, 
                weightsABatch, weightsBBatch, 
                sigma, 
                rotationMatrices, 
                condensationMethod, 
                nBatchesRotation, 
                giipAABatch, giipBBBatch)
    
    return minimalDistances, minimalRotations
