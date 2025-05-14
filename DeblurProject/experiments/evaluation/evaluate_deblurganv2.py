# 保存为 D:\objection\python\DeblurProject\experiments\evaluation\evaluate_deblurganv2.py

import os
import cv2
import numpy as np
import time
import sys
import glob
from tabulate import tabulate

# 添加项目根目录到路径
sys.path.append("D:\\objection\\python\\DeblurProject")


def psnr(img1, img2):
    """计算PSNR (Peak Signal-to-Noise Ratio)"""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def ssim(img1, img2):
    """计算SSIM (Structural Similarity Index)"""
    # 确保图像是uint8类型
    if img1.dtype != np.uint8:
        img1 = np.clip(img1, 0, 255).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = np.clip(img2, 0, 255).astype(np.uint8)

    # 转换为灰度图
    if len(img1.shape) > 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) > 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # SSIM参数
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # 计算均值和方差
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    # 计算SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)


def create_dummy_reference(blur_img_path, blur_dir, ref_dir):
    """
    由于DeblurGANv2测试目录中没有参考图像，我们创建一个副本作为假参考
    这只用于测试，使用GOPRO数据集时将有真实的参考图像
    """
    blur_img_name = os.path.basename(blur_img_path)
    ref_img_path = os.path.join(ref_dir, blur_img_name)

    if not os.path.exists(ref_dir):
        os.makedirs(ref_dir, exist_ok=True)

    if not os.path.exists(ref_img_path):
        img = cv2.imread(blur_img_path)
        if img is not None:
            cv2.imwrite(ref_img_path, img)
            print(f"创建了假参考图像: {ref_img_path}")
            return ref_img_path

    return None


def find_matching_pairs(input_dir, result_dir, reference_dir=None, create_dummy_refs=False):
    """查找匹配的图像对（模糊/清晰/结果）用于评估"""
    pairs = []

    if reference_dir is None:
        reference_dir = input_dir

    print(f"查找图像对...\n输入目录: {input_dir}\n结果目录: {result_dir}\n参考目录: {reference_dir}")

    # 检查目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在 - {input_dir}")
        return pairs

    if not os.path.exists(result_dir):
        print(f"错误: 结果目录不存在 - {result_dir}")
        return pairs

    # 如果参考目录不存在且需要创建假参考，则创建目录
    if not os.path.exists(reference_dir) and create_dummy_refs:
        os.makedirs(reference_dir, exist_ok=True)
        print(f"创建了参考目录: {reference_dir}")
    elif not os.path.exists(reference_dir):
        print(f"错误: 参考目录不存在 - {reference_dir}")
        if not create_dummy_refs:
            return pairs

    # 查找所有的模糊输入图像
    blur_images = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"找到 {len(blur_images)} 个输入图像")

    # 查找所有的结果图像
    result_images = [f for f in os.listdir(result_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"找到 {len(result_images)} 个结果图像")

    # 逐一匹配图像
    for blur_img in blur_images:
        # 先检查结果图像是否存在
        result_img = None
        if blur_img in result_images:
            result_img = blur_img
            print(f"找到相同名称的结果: {blur_img}")
        else:
            # 尝试查找包含输入文件名的结果文件
            blur_name = os.path.splitext(blur_img)[0]
            for res_img in result_images:
                if blur_name in os.path.splitext(res_img)[0]:
                    result_img = res_img
                    print(f"找到包含输入名称的结果: {blur_img} -> {result_img}")
                    break

        if not result_img:
            print(f"未找到 {blur_img} 的结果图像，跳过")
            continue

        # 查找对应的清晰参考图像
        sharp_img = None

        # 检查参考目录中是否有同名文件
        if os.path.exists(os.path.join(reference_dir, blur_img)):
            sharp_img = blur_img
            print(f"在参考目录中找到同名图像: {blur_img}")

        # 如果没有找到参考图像，且允许创建假参考
        if sharp_img is None and create_dummy_refs:
            blur_path = os.path.join(input_dir, blur_img)
            dummy_path = create_dummy_reference(blur_path, input_dir, reference_dir)
            if dummy_path:
                sharp_img = os.path.basename(dummy_path)
                print(f"使用创建的假参考图像: {sharp_img}")

        # 如果找到了清晰参考图像和结果，添加到评估对中
        if sharp_img and result_img:
            pairs.append((blur_img, sharp_img, result_img))
            print(f"添加评估对 #{len(pairs)}: {blur_img}, {sharp_img}, {result_img}")
        else:
            if not sharp_img:
                print(f"警告: 未找到 {blur_img} 的参考图像")

    print(f"总共找到 {len(pairs)} 个有效的图像评估对")
    return pairs


def main():
    # ============== 路径设置 ==============
    # 当前测试路径
    # input_dir = "D:\\objection\\python\\DeblurProject\\models\\DeblurGANv2-master\\test_img"
    # result_dir = "D:\\objection\\python\\DeblurProject\\models\\DeblurGANv2-master\\results"

    # 为测试创建一个临时参考目录（仅用于计算测试指标）
    reference_dir = "D:\\objection\\python\\DeblurProject\\models\\DeblurGANv2-master\\temp_reference"
    create_dummy_refs = True  # 允许创建假参考图像用于测试

    # GOPRO数据集路径（使用时取消注释）
    # input_dir = "D:\\objection\\python\\DeblurProject\\datasets\\GOPRO\\test\\blur"
    # input_dir = "D:\objection\python\DeblurProject\data\test\blur"
    # reference_dir = "D:\\objection\\python\\DeblurProject\\datasets\\GOPRO\\test\\sharp"
    # reference_dir = "D:\objection\python\DeblurProject\data\test\sharp"
    # result_dir = "D:\\objection\\python\\DeblurProject\\results\\DeblurGANv2_GOPRO"
    # result_dir = "D:\objection\python\DeblurProject\results\DeblurGANv2"

    input_dir = r"D:\objection\python\DeblurProject\data\test\blur"
    reference_dir = r"D:\objection\python\DeblurProject\data\test\sharp"
    result_dir = r"D:\objection\python\DeblurProject\results\DeblurGANv2"
    create_dummy_refs = False  # 使用GOPRO时不需要创建假参考
    # =======================================

    output_dir = "D:\\objection\\python\\DeblurProject\\results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "deblurganv2_metrics.csv")

    print(f"评估DeblurGANv2的性能...\n输入目录: {input_dir}\n结果目录: {result_dir}\n参考目录: {reference_dir}")

    # 寻找匹配的图像对，允许创建假参考用于测试
    image_pairs = find_matching_pairs(input_dir, result_dir, reference_dir, create_dummy_refs)

    if not image_pairs:
        print("未找到可用于评估的图像对！")
        return

    # 存储结果
    results = []
    all_psnr = []
    all_ssim = []

    # 评估每对图像
    print("\n开始评估DeblurGANv2性能...")
    for blur_img, sharp_img, result_img in image_pairs:
        blur_path = os.path.join(input_dir, blur_img)
        sharp_path = os.path.join(reference_dir, sharp_img)
        result_path = os.path.join(result_dir, result_img)

        # 读取图像
        blur = cv2.imread(blur_path)
        sharp = cv2.imread(sharp_path)
        result = cv2.imread(result_path)

        # 检查图像是否成功加载
        if blur is None or sharp is None or result is None:
            print(f"无法读取图像: {blur_img} 或 {sharp_img} 或 {result_img}")
            continue

        # 确保尺寸匹配
        if sharp.shape != result.shape:
            print(f"调整尺寸: {result.shape} -> {sharp.shape}")
            result = cv2.resize(result, (sharp.shape[1], sharp.shape[0]))

        # 计算指标
        psnr_value = psnr(sharp, result)
        ssim_value = ssim(sharp, result)

        # 对于假参考图像，这些指标不具有实际意义
        if create_dummy_refs:
            print(f"图像 {blur_img}: PSNR = {psnr_value:.2f} dB*, SSIM = {ssim_value:.4f}* (*使用假参考)")
        else:
            print(f"图像 {blur_img}: PSNR = {psnr_value:.2f} dB, SSIM = {ssim_value:.4f}")

        # 存储结果
        results.append([blur_img, psnr_value, ssim_value])
        all_psnr.append(psnr_value)
        all_ssim.append(ssim_value)

    # 计算平均指标
    avg_psnr = np.mean(all_psnr) if all_psnr else 0
    avg_ssim = np.mean(all_ssim) if all_ssim else 0

    # 计算模型大小
    model_size_mb = 0
    model_files = [
        os.path.join("D:\\objection\\python\\DeblurProject\\models\\DeblurGANv2-master\\weights", "fpn_inception.h5")
    ]

    for model_file in model_files:
        if os.path.exists(model_file):
            model_size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"模型文件: {model_file}, 大小: {model_size_mb:.2f} MB")

    # 打印平均指标
    print("\n========= DeblurGANv2 性能指标 =========")
    if create_dummy_refs:
        print("注意: 以下指标使用假参考图像计算，仅用于测试目的")
    print(f"平均 PSNR: {avg_psnr:.2f} dB")
    print(f"平均 SSIM: {avg_ssim:.4f}")
    print(f"模型大小: {model_size_mb:.2f} MB")
    print("=======================================")

    # 打印每个图像的结果表格
    if results:
        print("\n每个图像的详细指标:")
        headers = ["图像", "PSNR (dB)", "SSIM"]
        print(tabulate(results, headers=headers, tablefmt="grid"))

        # 保存结果到CSV
        with open(output_file, "w") as f:
            f.write("Image,PSNR,SSIM\n")
            for row in results:
                f.write(f"{row[0]},{row[1]:.2f},{row[2]:.4f}\n")
            f.write(f"Average,{avg_psnr:.2f},{avg_ssim:.4f}\n")

        print(f"\n结果已保存到: {output_file}")

        if create_dummy_refs:
            print(f"\n注意: 由于使用假参考图像，上述指标不具有实际意义。")
            print(f"      在使用GOPRO数据集时，将有真实的参考图像用于评估。")
    else:
        print("没有有效的评估结果！")


if __name__ == "__main__":
    main()