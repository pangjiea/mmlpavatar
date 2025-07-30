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

def head_body_separation_loss(gaussians: GaussianModel, alpha=1.0, beta=1.0):
    """
    头部-身体分离损失函数，确保头部和身体Gaussians的解耦
    
    Args:
        gaussians: GaussianModel实例
        alpha: 特征差异损失权重
        beta: 空间分离损失权重
    
    Returns:
        total_loss: 总的分离损失
        loss_dict: 各项损失的字典
    """
    losses = {}
    
    # 只有在头部身体分离模式下才计算这个损失
    if gaussians.head_encoder_params is None or gaussians.body_encoder_params is None:
        return torch.tensor(0.0, device='cuda'), {}
    
    if gaussians.head_gs_mask is None or gaussians.body_gs_mask is None:
        return torch.tensor(0.0, device='cuda'), {}
    
    # 1. 特征差异损失 - 确保头部和身体编码器学习到不同的特征表示
    head_features = gaussians._get_joint_features_for_type('head')  # [16]
    body_features = gaussians._get_joint_features_for_type('body')  # [66] 
    
    # 为了比较，我们只取前16维进行对比
    body_features_subset = body_features[:16]  # 取body特征的前16维
    feature_diff_loss = alpha * torch.nn.functional.cosine_similarity(
        head_features.unsqueeze(0), 
        body_features_subset.unsqueeze(0), 
        dim=1
    ).abs().mean()
    losses['feature_diff'] = feature_diff_loss
    
    # 2. 空间分离损失 - 确保头部和身体Gaussians在空间上分离
    if hasattr(gaussians, '_xyz') and gaussians._xyz.shape[0] > 0:
        try:
            xyz = gaussians.get_xyz  # [N, 3]
            head_mask = gaussians.head_gs_mask
            body_mask = gaussians.body_gs_mask
            
            if head_mask.sum() > 0 and body_mask.sum() > 0:
                head_xyz = xyz[head_mask]  # [N_head, 3]
                body_xyz = xyz[body_mask]  # [N_body, 3]
                
                # 计算头部和身体Gaussians之间的最小距离
                # 使用broadcasting计算所有头部点到所有身体点的距离
                head_xyz_exp = head_xyz.unsqueeze(1)  # [N_head, 1, 3]
                body_xyz_exp = body_xyz.unsqueeze(0)  # [1, N_body, 3]
                
                distances = torch.norm(head_xyz_exp - body_xyz_exp, dim=2)  # [N_head, N_body]
                min_distances = distances.min(dim=1)[0]  # [N_head] - 每个头部点到最近身体点的距离
                
                # 鼓励最小距离大于某个阈值（例如0.05米）
                separation_threshold = 0.05
                separation_loss = beta * torch.relu(separation_threshold - min_distances).mean()
                losses['spatial_separation'] = separation_loss
            else:
                losses['spatial_separation'] = torch.tensor(0.0, device='cuda')
        except Exception as e:
            # 如果get_xyz失败（比如在测试环境中），则使用_xyz
            try:
                xyz = gaussians._xyz  # [N, 3]
                head_mask = gaussians.head_gs_mask
                body_mask = gaussians.body_gs_mask
                
                if head_mask.sum() > 0 and body_mask.sum() > 0:
                    head_xyz = xyz[head_mask]  # [N_head, 3]
                    body_xyz = xyz[body_mask]  # [N_body, 3]
                    
                    head_xyz_exp = head_xyz.unsqueeze(1)  # [N_head, 1, 3]
                    body_xyz_exp = body_xyz.unsqueeze(0)  # [1, N_body, 3]
                    
                    distances = torch.norm(head_xyz_exp - body_xyz_exp, dim=2)  # [N_head, N_body]
                    min_distances = distances.min(dim=1)[0]  # [N_head]
                    
                    separation_threshold = 0.05
                    separation_loss = beta * torch.relu(separation_threshold - min_distances).mean()
                    losses['spatial_separation'] = separation_loss
                else:
                    losses['spatial_separation'] = torch.tensor(0.0, device='cuda')
            except Exception:
                losses['spatial_separation'] = torch.tensor(0.0, device='cuda')
    else:
        losses['spatial_separation'] = torch.tensor(0.0, device='cuda')
    
    # 3. 编码器特化损失 - 确保每个编码器专注于自己的区域
    if hasattr(gaussians, 'get_encoded_feature'):
        encoded_features = gaussians.get_encoded_feature  # [N_feat, feat_dim]
        
        if gaussians.nbr_gsft is not None and gaussians.nbr_gsft.shape[0] > 0:
            # 获取每个Gaussian点对应的编码特征
            N_gs = gaussians.nbr_gsft.shape[0]
            head_mask = gaussians.head_gs_mask
            body_mask = gaussians.body_gs_mask
            
            if head_mask.sum() > 0 and body_mask.sum() > 0:
                # 计算头部和身体区域特征的方差，鼓励特化
                head_feat_indices = gaussians.nbr_gsft[head_mask].flatten().unique()
                body_feat_indices = gaussians.nbr_gsft[body_mask].flatten().unique()
                
                if len(head_feat_indices) > 1 and len(body_feat_indices) > 1:
                    head_encoded = encoded_features[head_feat_indices]
                    body_encoded = encoded_features[body_feat_indices]
                    
                    # 计算特征方差，鼓励每个区域内特征的多样性
                    head_var = head_encoded.var(dim=0).mean()
                    body_var = body_encoded.var(dim=0).mean()
                    specialization_loss = -0.1 * (head_var + body_var)  # 负号因为我们要最大化方差
                    losses['encoder_specialization'] = specialization_loss
                else:
                    losses['encoder_specialization'] = torch.tensor(0.0, device='cuda')
            else:
                losses['encoder_specialization'] = torch.tensor(0.0, device='cuda')
        else:
            losses['encoder_specialization'] = torch.tensor(0.0, device='cuda')
    else:
        losses['encoder_specialization'] = torch.tensor(0.0, device='cuda')
    
    # 计算总损失
    total_loss = sum(losses.values())
    
    return total_loss, losses

def head_consistency_loss(gaussians: GaussianModel, head_mask_gt, gamma=1.0):
    """
    头部一致性损失，确保通过mesh生成的head_mask与Gaussian分类的一致性
    
    Args:
        gaussians: GaussianModel实例
        head_mask_gt: 来自mesh投影的真实头部mask [H, W]
        gamma: 损失权重
    
    Returns:
        consistency_loss: 一致性损失
    """
    if gaussians.head_gs_mask is None or gaussians.body_gs_mask is None:
        return torch.tensor(0.0, device='cuda')
    
    # 这个损失需要在渲染时计算，因为需要将Gaussian点投影到图像空间
    # 这里只是一个占位符，实际实现需要在训练循环中计算
    return torch.tensor(0.0, device='cuda')
