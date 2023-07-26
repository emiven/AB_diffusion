from skimage.color import lab2rgb
import torch
from IPython.utils import io as iol


def plotMinMax(Xsub):
    labels=["C{}".format(i) for i in range(Xsub.shape[1])]
    print("______________________________")
    for i, lab in enumerate(labels):
        mi = torch.min(Xsub[:, i, :, :])
        ma = torch.max(Xsub[:, i, :, :])
        print("{} : MIN={:8.4f}, MAX={:8.4f}".format(lab, mi.item(), ma.item()))


def de_normalize_lab(LAB):
    if LAB.shape[1] == 3:
        L, A, B = LAB[:,0,:,:],LAB[:,1,:,:],LAB[:,2,:,:] 
        L = (L+1)*50.0
        A = (A*128.0)
        B = (B*128.0)
        return torch.stack([L,A,B], axis = 1)
    elif LAB.shape[1] == 2:
        LAB = LAB*128.0
        return LAB
    elif LAB.shape[1] == 1:
        return (LAB+1)*50.0
    else:
        print("shape: " + str(LAB.shape))
        raise ValueError("LAB shape not recognized" )

def normalize_lab(LAB):
     if LAB.shape[1] == 3:
        L, A, B = LAB[:,0,:,:],LAB[:,1,:,:],LAB[:,2,:,:]
    
        L = L/ 50.0 - 1.0
        A = A/ 128.0
        B = B/ 128.0
        return torch.stack([L,A,B], axis = 1)
     elif LAB.shape[1] == 2:
        LAB = LAB/ 128.0
        return LAB
     elif LAB.shape[1] == 1:
        return LAB / 50.0 - 1.0
     else:
        print("shape: " + str(LAB.shape))
        raise ValueError("LAB shape not recognized" )

def LAB2RGB(im_lab):
    lab = de_normalize_lab(im_lab)#.cpu().detach().numpy().transpose(0,3,2,1)

    lab = lab.permute(0,2,3,1)
    with iol.capture_output() as captured:
        rgb = lab2rgb(lab)

    return rgb

