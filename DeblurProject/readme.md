# DeblurGAN 复现与对比实验

本项目为 **数字图像处理** 课程作业，主要内容为基于生成对抗网络（GAN）的图像去模糊模型复现与对比实验。选取三种代表性模型：

- **DeblurGAN**（Kupyn 等，CVPR 2018）
- **DeblurGAN-v2**（Kupyn 等，ArXiv 2019）
- **Ghost-DeblurGAN**（IROS 2022 轻量化版）

实验在精简的 GoPro 子集（约 8.9 GB）上进行，包含推理和PSNR/SSIM 定量评估。

## 数据集：GoPro 轻量版（≈ 8.9 GB）