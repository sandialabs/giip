import torch

def slow(positionsA, positionsB, weightsA, weightsB, sigma):
    """Slooooowly compute GIIPs pairwise between configurations A and configurations B

    Args:
        positionsA: nAtomsA*3 [a tensor of neighborhoods, each of which contains nAtomsA]
        positionsB: nAtomsB*3 [a tensor of neighborhoods, each of which contains nAtomsA]
        weightsA: nAtomsA [a tensor of neighborhoods, each of which contains nAtomsA]
        weightsB: nAtomsB [a tensor of neighborhoods, each of which contains nAtomsA]
        sigma: float

    Returns:
        : singleton tensor
    """

    summand = 0
    for positionA, weightA in zip(positionsA,weightsA):
        for positionB, weightB in zip(positionsB,weightsB):
            distance = torch.linalg.norm(positionB-positionA)
            distanceSqr = distance*distance
            term = weightA*weightB*torch.exp(-distanceSqr/(4*sigma*sigma))
            summand += term
    return summand