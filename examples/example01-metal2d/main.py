import h5py
import math
import networkx as nx # for clustering
import numpy as np
import torch

import GIIP.condensed_distance
import GIIP.distance
import GIIP.inner_product
import GIIP.optimize_orientations
import GIIP.weights

if True: #A quick but uninteresting example
    tag = 'metal2d.1nn.100.h5'
    rCutoff = 1.77
    rSkin = 1.35
else: #A slower (20-ish minutes) example, but more interesting
    tag = 'metal2d.2nn.1000.h5'
    rCutoff = 2.80
    rSkin = 2.46

#We're going to save the output to this Hdf5 file
fout = h5py.File('examples/example01-metal2d/example01-out-' + tag,'w')

radii = np.array([rSkin,rCutoff])
fout.create_dataset('radii', shape=radii.shape, dtype=np.float32, data=radii)
fout.flush()

# Set parameters for this run
device = 'cpu' # use 'cuda' to perform calculations on GPU, and 'cpu' to do them on the cpu.
dtype = torch.float32 # can use float64 for double precision

neighborhoodsPath = 'examples/example01-metal2d/' + tag
condensationMethod = None

# Load the neighborhoods. We're only going to use distances and positions for this demo, since this is a unary material. Positions and distances are relative to central atoms.
# The dataset for this demo is 1000 neighborhoods from a defect-riddled 2d LJ metal, cut off at 2nd nearest neighbors.
def loadNeighborhoods(filePath, device='cpu', dtype=torch.float32):
    f = h5py.File(filePath,'r')
    centerSpecies = torch.tensor(f['/centerSpecies'], dtype=torch.int16, device=device)
    nNeighbors = torch.tensor(f['/nNeighbors'], dtype=torch.int32, device=device)
    species = torch.tensor(f['/species'], dtype=torch.int16, device=device)
    distances = torch.tensor(f['/distances'], dtype=dtype, device=device)
    positions = torch.tensor(f['/positions'], dtype=dtype, device=device)
    f.close()
    return (centerSpecies, nNeighbors, species, distances, positions)
centerSpecies, nNeighbors, species, distances, positions = loadNeighborhoods(neighborhoodsPath, dtype=dtype, device=device)
#Strip out the 3rd spatial dimension to positions, since we're operating in 2d
positions = positions[:,:,:-1]
nEnvironments = len(positions)
#The species tensor contains chemical information,
#   and is indexed iNeighborhood,iAtom.
#   Species -1 means a null atom.
print('species', species.shape, species.dtype)
#The distances tensor contains distances of atoms from the origin,
#   and is indexed iNeighborhood,iAtom.
print('distances', distances.shape, distances.dtype)
#The positions tensor contains positions of atoms,
#   and is indexed iNeighborhood,iAtom,iDimension
print('positions', positions.shape, positions.dtype)

weights = GIIP.weights.computeSpeciesAgnosticWeights(species, distances, rSkin, rCutoff)
#The weights tensor contains weight values,
#   and is indexed iNeighborhood,iAtom
print('weights', weights.shape, weights.dtype)
fout.create_dataset('weights', shape=weights.shape, dtype=np.float32, data=weights.to('cpu'))
fout.flush()

#Randomly rotate the positions, just to make life a little harder for us
randomRotations_radians = torch.rand((nEnvironments,), device=device, dtype=dtype) * 2*math.pi
c = torch.cos(randomRotations_radians)
s = torch.sin(randomRotations_radians)
randomRotationMatrices = torch.zeros((nEnvironments,2,2),device=device, dtype=dtype)
randomRotationMatrices[:,0,0]=c
randomRotationMatrices[:,0,1]=s
randomRotationMatrices[:,1,0]=-s
randomRotationMatrices[:,1,1]=c
positions = torch.einsum('nij,nxj->nxi',randomRotationMatrices,positions)
fout.create_dataset('positions', shape=positions.shape, dtype=np.float32, data=positions.to('cpu'))
fout.flush()

#Pick a sigma value. I don't have a perfect way to pick a sigma, just trial and error.
sigma = 0.35

# Now that we have positions, weights, and sigma, we have everything we need to do some orientation optimization.
# Get an O(2) covering. Find the optimal orientations that minimize the GIIP distance between each pair of environments. Return that minimal GIIP distance.
maxMisorientationDegrees = 1 #Resolution of global search through orientation space
reflections = True #Include roto-inversions in addition to rotations
normalize = True #Include identity in rotation set
rotationMatrices = GIIP.optimize_orientations.getRotationMatrices2d(maxMisorientationDegrees, reflections, normalize, device=device, dtype=dtype)
#The rotationMatrices tensor contains a set of rotation matrices,
#   and is indexed iRotation,iDimension1,iDimension2
print('rotationMatrices', rotationMatrices.shape, rotationMatrices.dtype)
nBatchesRotation = 1
batchSizeA = 20
batchSizeB = 20
giipDistanceSquaredL2Minimized, minimalOrientationMatrices = GIIP.optimize_orientations.exhaustive.pairwise_batchABR(
    positions, positions, weights, weights, sigma, rotationMatrices, condensationMethod=None, 
    batchSizeA=batchSizeA, batchSizeB=batchSizeB, nBatchesRotation = nBatchesRotation)
#The giipDistanceSquaredL2Minimized tensor contains the orientation-minimized L2 condensed GIIP distances,
#   and is indexed iNeighborhood1,iNeighborhood2
#The minimalOrientationMatrices tensor contains the rotation matrices that minimize the pairwise distance,
#   and is indexed iNeighborhood1,iNeighborhood2,iDimension1,iDimension2
print('giipDistanceSquaredL2Minimized', giipDistanceSquaredL2Minimized.shape, giipDistanceSquaredL2Minimized.dtype)
print('minimalOrientationMatrices', minimalOrientationMatrices.shape, minimalOrientationMatrices.dtype)
fout.create_dataset('giipDistanceSquaredL2Minimized', shape=giipDistanceSquaredL2Minimized.shape, dtype=np.float32, data=giipDistanceSquaredL2Minimized.to('cpu'))
fout.create_dataset('minimalOrientationMatrices', shape=minimalOrientationMatrices.shape, dtype=np.float32, data=minimalOrientationMatrices.to('cpu'))
fout.flush()

# Let's do some clustering
q = np.sqrt(np.abs((giipDistanceSquaredL2Minimized.cpu().numpy())))
g = nx.from_numpy_array(q < .1)
classes = [list(x) for x in nx.connected_components(g)]
# Combine clusters that are sufficiently close together.
cutoff = .5
interClassDistances = torch.tensor([[torch.max(giipDistanceSquaredL2Minimized[clsA,:][:,clsB]).item() for clsA in classes] for clsB in classes])
interClassDistancesThresh = interClassDistances < cutoff
classAssignments = [torch.min(torch.where(row)[0]).item() for row in interClassDistancesThresh]
classReassignments = {a:i for i,a in enumerate(sorted(list(set(classAssignments))))}
classAssignments = [classReassignments[i] for i in classAssignments]
classesCombined = [[] for i in classReassignments]
for iCls,cls in enumerate(classes):
    assignment = classAssignments[iCls]
    classesCombined[assignment] += cls
classes = classesCombined
nClasses = len(classes)
classAssignments = torch.zeros((nEnvironments,),device=device,dtype=torch.int16)
for iCls,cls in enumerate(classes): classAssignments[cls]=iCls
# Pull an exemplar from each class.
def getExemplar(cls):
    giipDistanceSquaredL2Minimized_cls = giipDistanceSquaredL2Minimized[cls,:][:,cls]
    iMinWithinCls = torch.argmin(torch.median(giipDistanceSquaredL2Minimized_cls,axis=0).values).item()
    iExemplar = cls[iMinWithinCls]
    return iExemplar
exemplars = torch.tensor([getExemplar(cls) for cls in classes])
#Save the classes
fout.create_dataset('classAssignments', shape=classAssignments.shape, dtype=np.int16, data=classAssignments.to('cpu'))
fout.create_dataset('exemplars', shape=exemplars.shape, dtype=np.int64, data=exemplars.to('cpu'))
fout.flush()

#Close the output file.
fout.close()