from math import pi,ceil
import torch

def getRotationMatrices3d(maxMisorientationDegrees = 5, reflections = False, normalize=False, device='cpu', dtype=torch.float32):
    """Loads and performs basic manipulations of coverings of orientation space.

    Augments orientation space covering with reflections, on request.
    Normalizes orientations so that first in the list is the identity matrix.

    Args:
        maxMisorientationDegrees: maximum misorientation of the covering, with options being 1,2,3,4, or 5.
        reflections: include orientation matrices with det=-1
        normalize: ensure that the first matrix in the list is identity
        device: Torch device for returned tensor
    
    Returns:
        A Torch tensor of dimension nOrientations x 3 x 3
    """

    rotationMatricesFilePath = 'assets/rotationMatrices' + str(maxMisorientationDegrees) + '.txt'
    with open(rotationMatricesFilePath,'r') as f: rotationMatricesFile = f.read().splitlines(keepends=False)
    nMatrices = int(rotationMatricesFile[1])
    matricesFlat = torch.tensor([[float(token) for token in line.strip().split(' ')] for line in rotationMatricesFile[2:]], dtype=dtype, device=device)
    matrices = torch.zeros((nMatrices,3,3), dtype=dtype, device=device)
    matrices[:,0,:] = matricesFlat[:,0:3]
    matrices[:,1,:] = matricesFlat[:,3:6]
    matrices[:,2,:] = matricesFlat[:,6:9]

    if reflections: matrices = torch.cat((matrices,-matrices), dim=0)
    
    if normalize:
        matrix0 = matrices[0,:,:]
        matrix0Inv = torch.linalg.inv(matrix0)
        matrices = torch.einsum('oab,bc->oac', matrices, matrix0Inv)

    return matrices

def getRotationMatricesRefined3d(maxMisorientationDegrees=1, refinementAngleDegrees=5, normalize=False, device='cpu', dtype=torch.float32):
    """Returns a solid angle section of an orientation space covering, centered on Identity.

    Args:
        maxMisorientationDegrees: maximum misorientation of the covering, with options being 1,2,3,4, or 5.
        refinementAngleDegrees: Angle of boundary of solid angle section, unrestricted
        device: Torch device for returned tensor

    Returns:
        A Torch tensor of dimension nOrientations x 3 x 3
    """

    fullCovering = getRotationMatrices3d(maxMisorientationDegrees, False, normalize, device, dtype)
    
    #Compute angle from axis-angle formulation of all rotation matrices
    fullCoveringTrace = fullCovering[:,0,0] + fullCovering[:,1,1] + fullCovering[:,2,2]
    fullCoveringRotationAngles = torch.arccos(0.5*(fullCoveringTrace - 1))
    fullCoveringRotationAnglesDegs = fullCoveringRotationAngles * (180/pi)

    sectionCovering = fullCovering[fullCoveringRotationAnglesDegs < refinementAngleDegrees,:,:]

    return sectionCovering

def getRotationMatrices2d(maxMisorientationDegrees = 5, reflections = False, normalize=False, device='cpu', dtype=torch.float32):
    """Loads and performs basic manipulations of coverings of orientation space.

    Augments orientation space covering with reflections, on request.
    Normalizes orientations so that first in the list is the identity matrix.

    Args:
        maxMisorientationDegrees: maximum misorientation of the covering, with options being 1,2, or 5.
        reflections: include orientation matrices with det=-1
        normalize: ensure that the first matrix in the list is identity
        device: Torch device for returned tensor
    
    Returns:
        A Torch tensor of dimension nOrientations x 2 x 2
    """

    angles = torch.linspace(0,2*pi,steps=int(ceil(360/maxMisorientationDegrees))+1)[:-1]
    nAngles = len(angles)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    matrices = torch.zeros((nAngles,2,2), dtype=dtype, device=device)
    matrices[:,0,0] = cos
    matrices[:,0,1] = -sin
    matrices[:,1,0] = sin
    matrices[:,1,1] = cos

    if reflections:
        matricesInv = torch.zeros_like(matrices)
        matricesInv[:,0,:] = matrices[:,1,:]
        matricesInv[:,1,:] = matrices[:,0,:]
        matrices = torch.cat((matrices,matricesInv))
    
    if normalize:
        matrix0 = matrices[0,:,:]
        matrix0Inv = torch.linalg.inv(matrix0)
        matrices = torch.einsum('oab,bc->oac', matrices, matrix0Inv)

    return matrices
