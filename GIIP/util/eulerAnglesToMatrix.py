import torch

def eulerAnglesToMatrix(eulerAngles):
    """
    Args:
        eulerAngles: {...}*3

    Returns:
        rotationMatrices: {...}*3*3
    """

    c = torch.cos(eulerAngles)
    s = torch.sin(eulerAngles)
    rotationMatricesShape = list(eulerAngles.shape[:-1])+[3,3]
    rotationMatrices = torch.zeros(rotationMatricesShape, device=eulerAngles.device, dtype=eulerAngles.dtype)

    rotationMatrices[...,0,0] = c[...,0]*c[...,1]*c[...,2] - s[...,0]*s[...,2]
    rotationMatrices[...,0,1] = -c[...,2]*s[...,0] - c[...,0]*c[...,1]*s[...,2]
    rotationMatrices[...,0,2] = c[...,0]*s[...,1]
    rotationMatrices[...,1,0] = c[...,1]*c[...,2]*s[...,0] + c[...,0]*s[...,2]
    rotationMatrices[...,1,1] = c[...,0]*c[...,2] - c[...,1]*s[...,0]*s[...,2]
    rotationMatrices[...,1,2] = s[...,0]*s[...,1]
    rotationMatrices[...,2,0] = -c[...,2]*s[...,1]
    rotationMatrices[...,2,1] = s[...,1]*s[...,2]
    rotationMatrices[...,2,2] = c[...,1]

    return rotationMatrices

if __name__=='__main__':
    device = 'cuda'
    dtype = torch.float32

    for nAngles in [[],[5],[5,6]]:
        for device in ['cpu','cuda']:
            for dtype in [torch.float32, torch.float64]:
                eulerAngles = torch.rand(nAngles+[3], device=device, dtype=dtype)
                rotationMatrices = eulerAnglesToMatrix(eulerAngles)
                print(rotationMatrices.shape)
