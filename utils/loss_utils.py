import torch
import torch.nn.functional as F
import ssl
import urllib.request

from torch.nn.functional import l1_loss
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

from scene.gaussian_model import GaussianModel

lpips_model = None

def psnr(img1, img2):
    img1 = img1.permute(2,0,1)[None]
    img2 = img2.permute(2,0,1)[None]
    loss = peak_signal_noise_ratio(img1, img2)
    return loss

def ssim_loss(img1, img2, bbox=None):
    if bbox is not None:
        img1 = img1[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        img2 = img2[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    img1 = img1.permute(2,0,1)[None]
    img2 = img2.permute(2,0,1)[None]
    loss = 1.0 - structural_similarity_index_measure(img1, img2)
    return loss

def lpips_loss(img1, img2):
    global lpips_model
    img1 = img1.permute(2,0,1)[None]
    img2 = img2.permute(2,0,1)[None]
    if lpips_model is None:
        try:
            # Try normal SSL verification first
            lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).cuda()
        except (ssl.SSLError, urllib.error.URLError) as e:
            print(f"SSL error when downloading VGG model: {e}")
            print("Attempting download with SSL verification disabled...")
            # Temporarily disable SSL verification
            old_context = ssl._create_default_https_context
            ssl._create_default_https_context = ssl._create_unverified_context
            try:
                lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).cuda()
                print("Successfully downloaded VGG model with SSL verification disabled")
            finally:
                # Restore SSL context
                ssl._create_default_https_context = old_context
        
        for p in lpips_model.parameters(): p.requires_grad = False
    loss = lpips_model(img1, img2)
    return loss

def dxyz_smooth_loss(gaussians: GaussianModel):
    dxyz_vt = gaussians.get_dxyz_vt
    N, N_nbr = dxyz_vt.shape[0], gaussians.nbr_vt.shape[1]
    dxyz_nbrs = torch.index_select(dxyz_vt, 0, gaussians.nbr_vt.reshape(-1)).reshape(N, N_nbr, -1)
    loss = torch.linalg.vector_norm(dxyz_nbrs - dxyz_vt.unsqueeze(1), dim=-1, ord=2).mean()
    return loss

def gaussian_scaling_loss(scaling, threshold=0.01):
    scale_sub = scaling - threshold
    loss = torch.where(scale_sub > 0, scaling, torch.tensor(0, device=scaling.device)).mean()
    return loss
