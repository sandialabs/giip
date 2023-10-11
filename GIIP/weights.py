import torch

def distanceFade(rSkin,rCutoff,distances):
    ret = torch.zeros_like(distances)
    ret[distances<=rSkin]=1.0
    x = distances[distances>rSkin]
    ret[distances>rSkin] = (rCutoff-x)*(rCutoff-x)*(rCutoff-3*rSkin+2*x)/(rCutoff-rSkin)**3
    ret[distances>rCutoff] = 0.0
    return ret

def computeSpeciesAgnosticWeights(species, distances, rSkin, rCutoff):
    weights = distanceFade(rSkin, rCutoff, distances)
    weights[species < 0] = 0
    return weights


def computeSplitSpeciesWeights(species, distances, rSkin, rCutoff):
    uniqueSpecies = torch.sort(torch.unique(species[species>=0])).values
    nSpecies = uniqueSpecies.shape[0]
    nNeighborhoods = species.shape[0]
    nAtoms = species.shape[-1]
    fade = distanceFade(rSkin, rCutoff, distances)
    weights = torch.zeros((nSpecies, nNeighborhoods, nAtoms), device=species.device, dtype=distances.dtype)
    for iSpecies in range(nSpecies): weights[iSpecies,:,:] = fade * (species==uniqueSpecies[iSpecies])
    return weights

def computePositiveNegativeSpeciesWeights(species, distances, rSkin, rCutoff):
    uniqueSpecies = torch.sort(torch.unique(species[species>=0])).values
    nSpecies = uniqueSpecies.shape[0]
    nNeighborhoods = species.shape[0]
    nAtoms = species.shape[-1]
    fade = distanceFade(rSkin, rCutoff, distances)
    weights = fade * ( (species==uniqueSpecies[0]).short() - (species==uniqueSpecies[1]).short() )
    return weights
