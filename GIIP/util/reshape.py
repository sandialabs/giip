import torch

def pairwiseToElementwise(prefixDim, postfixDim, A, B):
    """
    Args:
        prefixDim: int >=0
        postfixDim: int >=0
        A: {...prefix}*{...A}*{...postfix}
        B: {...prefix}*{...B}*{...postfix}
    Returns:
        : {...prefix}*{...A}*{...B}*{...postfix}
        : {...prefix}*{...A}*{...B}*{...postfix}
    """
    
    prefixList = [-1 for _ in range(prefixDim)]
    postfixList = [-1 for _ in range(postfixDim)]
    ellipsisAList = list(A.shape[prefixDim:(A.dim()-postfixDim)])
    ellipsisBList = list(B.shape[prefixDim:(B.dim()-postfixDim)])
    ellipsisList = ellipsisAList+ellipsisBList
    outList = prefixList + ellipsisList + postfixList

    for _ in ellipsisBList: A = A.unsqueeze(-postfixDim-1)
    A = A.expand(outList)
    
    for _ in ellipsisAList: B = B.unsqueeze(prefixDim)
    B = B.expand(outList)
    
    return (A,B)
