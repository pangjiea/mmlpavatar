# 360-Frame Novel View Generation with Camera Trajectories

本项目实现了基于相机轨迹的360帧新视角生成系统，支持从多个相机位置生成平滑的圆形相机路径，用于创建高质量的人体新视角渲染视频。

## 功能特性

- **相机轨迹生成**: 从5个相机位置生成平滑的圆形路径 (1-2-3-4-5-4-3-2-1)
- **平滑插值**: 使用球面线性插值(SLERP)实现相机间的平滑过渡
- **高性能渲染**: 支持实时渲染(达到289 FPS)
- **灵活的姿势处理**: 支持多种姿势格式(JSON, NPZ, AMASS等)
- **视频输出**: 集成ffmpeg进行视频编码
- **可配置帧数**: 支持自定义生成帧数(默认360帧)

## 文件结构

```
├── generate_camera_trajectory.py    # 相机轨迹生成脚本
├── test.py                          # 主要测试脚本(已修改)
├── create_video.py                  # 视频创建脚本
├── output/                          # 输出目录
│   └── camera_trajectory_360frames.json  # 生成的相机轨迹文件
└── README.md                        # 本文档
```

## 使用方法

### 1. 生成相机轨迹

```bash
python generate_camera_trajectory.py \
    --cam_files /tmp/cam1.json /tmp/cam2.json /tmp/cam3.json /tmp/cam4.json /tmp/cam5.json \
    --num_frames 360 \
    --output ./output/camera_trajectory_360frames.json
```

参数说明:
- `--cam_files`: 相机JSON文件路径列表(5个相机位置)
- `--num_frames`: 生成帧数(默认360)
- `--output`: 输出文件路径

### 2. 生成360帧新视角渲染

```bash
python test.py \
    --config configs/SQ_02.yaml \
    --model_dir output/SQ_02 \
    --out_dir output/360frame_render \
    --cam_path ./output/camera_trajectory_360frames.json \
    --pose_path /path/to/poses.json \
    --num_frames 360 \
    --test
```

参数说明:
- `--config`: 配置文件路径
- `--model_dir`: 训练好的模型目录
- `--out_dir`: 渲染输出目录
- `--cam_path`: 相机轨迹JSON文件
- `--pose_path`: 姿势参数文件
- `--num_frames`: 生成帧数
- `--test`: 启用测试模式

### 3. 创建视频

使用ffmpeg命令创建视频:

```bash
ffmpeg -y -r 30 -i %08d.png -c:v libx264 -vf fps=30 -pix_fmt yuv420p output_360frames.mp4
```

或者使用提供的视频创建脚本:

```bash
python create_video.py \
    --frame_dir output/360frame_render \
    --output output_360frames.mp4 \
    --fps 30
```

## 数据格式

### 相机JSON格式
```json
{
  "w2c": [...],      // 4x4世界到相机矩阵(展平)
  "K": [...],        // 3x3内参矩阵(展平)
  "fovx": 45.0,      // 水平视场角(度)
  "height": 1000,    // 图像高度
  "width": 1000      // 图像宽度
}
```

### 姿势JSON格式
```json
[
  {
    "pose": [...],   // SMPL姿势参数(72维)
    "Th": [...],     // 平移向量(3维)
    "Rh": [...]      // 旋转矩阵(3x3展平)
  }
]
```

## 技术实现

### 相机插值算法

1. **球面线性插值(SLERP)**: 用于相机旋转的平滑插值
2. **线性插值**: 用于相机位置的平滑过渡
3. **缓动函数**: 使用余弦缓动函数实现更自然的运动

### 渲染流程

1. **加载相机轨迹**: 从JSON文件读取360个相机位置
2. **加载姿势序列**: 从JSON文件加载SMPL姿势参数
3. **循环渲染**: 
   - 对每个帧，循环使用姿势序列
   - 对每个帧，使用对应的相机位置
   - 使用高斯泼溅技术进行渲染
4. **图像保存**: 将渲染结果保存为PNG格式

### 性能优化

- **GPU加速**: 使用CUDA进行并行计算
- **内存优化**: 合理管理GPU内存使用
- **批处理**: 支持批量渲染以提高效率

## 示例输出

- **渲染帧数**: 360帧
- **视频时长**: 12秒(30 FPS)
- **图像分辨率**: 1000x1000
- **渲染速度**: 289 FPS
- **输出格式**: PNG序列 + MP4视频

## 依赖项

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- OpenCV
- ffmpeg

## 注意事项

1. **相机文件**: 确保提供5个有效的相机JSON文件
2. **姿势文件**: 支持JSON格式的TalkBody4D数据集
3. **内存要求**: 大量帧生成需要足够的GPU内存
4. **存储空间**: 360帧渲染需要约1-2GB存储空间

## 故障排除

### 常见问题

1. **相机加载失败**: 检查相机JSON文件格式是否正确
2. **姿势加载失败**: 确认姿势文件路径和格式
3. **内存不足**: 减少生成帧数或降低图像分辨率
4. **ffmpeg错误**: 确保ffmpeg已正确安装

### 调试模式

启用详细输出:
```bash
python test.py --test --test_speed
```

## 贡献

欢迎提交问题和改进建议。如需修改代码，请遵循现有的代码风格和提交规范。

## 许可证

本项目遵循原始项目的许可证条款。