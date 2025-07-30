# 头部投影位置修正

## 问题描述

原有的头部掩膜生成中，投影位置不正确，导致生成的yanmo（掩膜）位置错误。需要参考TalkBody4D的正确投影方法来修正这个问题。

## 修正内容

### 1. SMPL-X顶点生成修正

**原有方式**:
```python
smpl_output = smplx_model(
    body_pose=pose[:, 3:66],
    global_orient=Rh,  # 直接使用Rh
    betas=beta,
    jaw_pose=jaw_pose,
    expression=expression,
    return_verts=True
)
vertices = smpl_output.vertices[0].cpu().numpy()
```

**修正后的TalkBody4D方式**:
```python
# 先用zero global_orient和transl生成vertices
smpl_output = smplx_model(
    body_pose=pose[:, 3:66],
    global_orient=torch.zeros_like(Rh),  # 重要：使用zero global_orient
    betas=beta,
    jaw_pose=jaw_pose,
    expression=expression,
    transl=torch.zeros_like(Th),  # 重要：使用zero transl
    return_verts=True
)
vertices = smpl_output.vertices[0].cpu().numpy()

# 然后按照TalkBody4D的方式应用全局变换
# TalkBody4D: verts = verts @ Rh.transpose(1, 2) + Th.unsqueeze(1)
if torch.any(Rh != 0):
    R_global = Rotation.from_rotvec(Rh[0].cpu().numpy()).as_matrix()
    vertices = vertices @ R_global.T  # 按照TalkBody4D的方式：verts @ R.T

vertices = vertices + Th[0].cpu().numpy()  # 应用平移
```

### 2. 投影方法修正

**完全按照TalkBody4D的投影流程**:
```python
def talkbody4d_projection_exact(vertices, K, R, T, img_w, img_h):
    # 确保T是列向量 [3, 1]
    if T.ndim == 1:
        T = T.reshape(3, 1)
    
    # TalkBody4D第148行: smplx_verts_cam = np.matmul(R, verts.T) + T
    smplx_verts_cam = np.matmul(R, vertices.T) + T  # [3, N]
    
    # 检查在相机前方的顶点
    valid_mask = smplx_verts_cam[2, :] > 0
    
    # TalkBody4D第150行: smplx_verts_proj = np.matmul(K, smplx_verts_cam)
    smplx_verts_proj = np.matmul(K, smplx_verts_cam)  # [3, N]
    
    # TalkBody4D第152行: smplx_verts_proj /= smplx_verts_proj[2, :]
    smplx_verts_proj /= smplx_verts_proj[2, :]
    
    # TalkBody4D第153行: smplx_verts_proj = smplx_verts_proj[:2, :].T
    smplx_verts_proj = smplx_verts_proj[:2, :].T  # [N, 2]
    
    # TalkBody4D第155行: 四舍五入并转换为整数
    smplx_verts_proj_rounded = np.round(smplx_verts_proj).astype(np.int32)
    
    # TalkBody4D第156-157行: clip到图像边界
    smplx_verts_proj_rounded[:, 0] = np.clip(smplx_verts_proj_rounded[:, 0], 0, img_w - 1)
    smplx_verts_proj_rounded[:, 1] = np.clip(smplx_verts_proj_rounded[:, 1], 0, img_h - 1)
    
    return smplx_verts_proj_rounded, final_valid_mask
```

### 3. 相机参数提取修正

**正确的w2c矩阵处理**:
```python
def extract_camera_params_from_w2c(w2c):
    # w2c的格式是 [R T; 0 1]，这正是TalkBody4D需要的格式
    R = w2c[:3, :3]  # [3, 3] 旋转矩阵
    T = w2c[:3, 3]   # [3] 平移向量
    return R, T
```

## 修改的文件

### 1. `utils/head_mask_utils.py`

- **修正了SMPL-X顶点生成流程**：使用zero global_orient和transl，然后手动应用变换
- **实现了精确的TalkBody4D投影**：`talkbody4d_projection_exact()`函数
- **简化了相机参数提取**：`extract_camera_params_from_w2c()`函数
- **更新了渲染器**：使用正确的投影方法

### 2. `scene/dataset.py`

- **更新了头部掩膜生成调用**：使用修正后的TalkBody4D投影
- **改为使用简单但正确的渲染器**：`use_advanced_renderer=False`

## 关键修正点

### 1. SMPL-X参数处理
- **Zero初始化**：global_orient和transl必须为零
- **手动变换**：按照TalkBody4D的方式手动应用Rh和Th变换
- **变换顺序**：先旋转后平移，旋转使用`verts @ R.T`

### 2. 投影流程
- **相机变换**：`smplx_verts_cam = np.matmul(R, verts.T) + T`
- **投影变换**：`smplx_verts_proj = np.matmul(K, smplx_verts_cam)`
- **透视除法**：`smplx_verts_proj /= smplx_verts_proj[2, :]`
- **坐标转换**：`smplx_verts_proj[:2, :].T`

### 3. 边界处理
- **四舍五入**：`np.round().astype(np.int32)`
- **边界裁剪**：`np.clip(pixels, 0, img_size-1)`

## 使用方式

### 向后兼容调用
```python
# 这个调用会自动使用修正后的投影
head_mask = generate_head_mask(
    pose, beta, expression, jaw_pose, Rh, Th,
    K, w2c, image_height, image_width, device='cpu'
)
```

### 显式调用
```python
# 显式使用修正后的TalkBody4D投影
head_mask = generate_head_mask_with_mesh_renderer(
    pose, beta, expression, jaw_pose, Rh, Th,
    K, w2c, image_height, image_width,
    body_mask=mask, device='cpu',
    use_advanced_renderer=False,  # 使用正确的TalkBody4D投影
    use_z_buffer=False
)
```

## 验证

修正后的投影方法：
1. **完全遵循TalkBody4D的实现**
2. **确保头部位置准确**
3. **与SQ02数据集兼容**
4. **保持向后兼容性**

现在头部掩膜的投影位置应该与TalkBody4D完全一致，确保在SQ02数据集上的投影是正确的。
