# 保存为 D:\objection\python\DeblurProject\experiments\evaluation\evaluate_deblurgan.py

import os
import cv2
import numpy as np
import time
import sys
import glob
from tabulate import tabulate

# 添加项目根目录到路径
sys.path.append(r"D:\objection\python\DeblurProject")


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


def find_matching_pairs(input_dir, result_dir, reference_dir):
    """查找匹配的图像对（模糊/清晰/结果）用于评估"""
    pairs = []

    print(f"查找图像对...\n输入目录: {input_dir}\n结果目录: {result_dir}\n参考目录: {reference_dir}")

    # 检查目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在 - {input_dir}")
        return pairs

    if not os.path.exists(result_dir):
        print(f"错误: 结果目录不存在 - {result_dir}")
        return pairs

    if not os.path.exists(reference_dir):
        print(f"错误: 参考目录不存在 - {reference_dir}")
        return pairs

    # 查找所有的模糊输入图像
    blur_images = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"找到 {len(blur_images)} 个输入图像")

    # 查找所有的清晰参考图像
    sharp_images = [f for f in os.listdir(reference_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"找到 {len(sharp_images)} 个参考图像")

    # 查找所有的结果图像
    result_images = [f for f in os.listdir(result_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    fake_b_images = [f for f in result_images if "_fake_B" in f]
    real_a_images = [f for f in result_images if "_real_A" in f]

    print(f"找到 {len(result_images)} 个结果图像，其中:")
    print(f"- {len(fake_b_images)} 个是fake_B格式 (去模糊结果)")
    print(f"- {len(real_a_images)} 个是real_A格式 (原始模糊图像)")

    # 按文件名匹配
    for blur_img in blur_images:
        # 查找对应的清晰参考图像
        blur_base = os.path.splitext(blur_img)[0]  # 获取不含扩展名的文件名
        sharp_img = None

        # 直接查找同名文件
        if blur_img in sharp_images:
            sharp_img = blur_img
            print(f"找到同名参考图像: {blur_img}")

        # 如果没找到同名文件，尝试去掉前导下划线查找
        if not sharp_img and blur_base.startswith('_'):
            clean_name = blur_base[1:] + os.path.splitext(blur_img)[1]
            if clean_name in sharp_images:
                sharp_img = clean_name
                print(f"找到匹配参考图像: {blur_img} -> {sharp_img}")

        # 尝试数字匹配
        if not sharp_img:
            try:
                # 如果文件名是纯数字，尝试直接匹配
                num = None
                # 移除所有下划线和前缀
                clean_base = blur_base.replace('_', '')
                # 提取数字部分
                num_part = ''.join(c for c in clean_base if c.isdigit())
                if num_part:
                    num = int(num_part)
                    # 查找同样数字的参考图像
                    for sharp_candidate in sharp_images:
                        sharp_base = os.path.splitext(sharp_candidate)[0]
                        sharp_clean = sharp_base.replace('_', '')
                        sharp_num = ''.join(c for c in sharp_clean if c.isdigit())
                        if sharp_num and int(sharp_num) == num:
                            sharp_img = sharp_candidate
                            print(f"通过数字匹配找到参考图像: {blur_img}({num}) -> {sharp_img}")
                            break
            except ValueError:
                pass

        # 查找对应的去模糊结果
        result_img = None

        # 1. 直接尝试按文件名中的数字查找fake_B文件
        try:
            num_part = ''.join(c for c in blur_base.replace('_', '') if c.isdigit())
            if num_part:
                num = int(num_part)
                result_pattern = f"{num}_fake_B.png"
                padded_result_pattern = f"{str(num).zfill(3)}_fake_B.png"

                if result_pattern in fake_b_images:
                    result_img = result_pattern
                    print(f"找到直接匹配结果: {blur_img} -> {result_img}")
                elif padded_result_pattern in fake_b_images:
                    result_img = padded_result_pattern
                    print(f"找到补零匹配结果: {blur_img} -> {result_img}")
        except ValueError:
            pass

        # 如果没找到结果，尝试按索引匹配
        if not result_img and fake_b_images:
            # 按文件名排序，确保顺序一致
            blur_images_sorted = sorted(blur_images)
            fake_b_images_sorted = sorted(fake_b_images)

            if blur_images_sorted.index(blur_img) < len(fake_b_images_sorted):
                idx = blur_images_sorted.index(blur_img)
                result_img = fake_b_images_sorted[idx]
                print(f"按索引匹配结果: {blur_img} -> {result_img}")

        # 如果找到了清晰参考图像和结果，添加到评估对中
        if sharp_img and result_img:
            pairs.append((blur_img, sharp_img, result_img))
            print(f"添加评估对 #{len(pairs)}: {blur_img}, {sharp_img}, {result_img}")
        else:
            if not sharp_img:
                print(f"警告: 未找到 {blur_img} 的参考图像")
            if not result_img:
                print(f"警告: 未找到 {blur_img} 的结果图像")

    print(f"总共找到 {len(pairs)} 个有效的图像评估对")
    return pairs


def main():
    # ============== 路径设置 ==============
    # 根据您的目录结构更新这些路径
    input_dir = r"D:\objection\python\DeblurProject\data\blurred_sharp\blurred_sharp\blurblur"
    reference_dir = r"D:\objection\python\DeblurProject\data\blurred_sharp\blurred_sharp\sharpsharp"
    result_dir = r"D:\objection\python\DeblurProject\results\DeblurGAN_last\experiment_name\test_latest"

    # 设置输出目录
    output_dir = r"D:\objection\python\DeblurProject\results"
    os.makedirs(output_dir, exist_ok=True)

    # 设置输出文件
    output_file = os.path.join(output_dir, "deblurgan_metrics.csv")

    print(f"评估DeblurGAN的性能...\n输入目录: {input_dir}\n结果目录: {result_dir}\n参考目录: {reference_dir}")

    # 寻找匹配的图像对
    image_pairs = find_matching_pairs(input_dir, result_dir, reference_dir)

    if not image_pairs:
        print("未找到可用于评估的图像对！")

        # 提供目录内容以便调试
        print("\n调试信息:")
        if os.path.exists(input_dir):
            input_files = os.listdir(input_dir)[:5]  # 仅列出前5个文件
            print(f"输入目录前5个文件: {input_files}")

        if os.path.exists(result_dir):
            result_files = os.listdir(result_dir)[:5]
            print(f"结果目录前5个文件: {result_files}")

        return

    # 存储结果
    results = []
    all_psnr = []
    all_ssim = []
    processing_times = []

    # 评估每对图像
    print("\n开始评估DeblurGAN性能...")
    for blur_img, sharp_img, result_img in image_pairs:
        blur_path = os.path.join(input_dir, blur_img)
        sharp_path = os.path.join(reference_dir, sharp_img)
        result_path = os.path.join(result_dir, result_img)

        # 读取图像
        blur = cv2.imread(blur_path)
        sharp = cv2.imread(sharp_path)
        result = cv2.imread(result_path)

        # 检查图像是否成功加载
        if blur is None:
            print(f"错误: 无法读取模糊图像 {blur_path}")
            continue
        if sharp is None:
            print(f"错误: 无法读取清晰图像 {sharp_path}")
            continue
        if result is None:
            print(f"错误: 无法读取结果图像 {result_path}")
            continue

        # 确保尺寸匹配
        if sharp.shape != result.shape:
            print(f"调整尺寸: {result.shape} -> {sharp.shape}")
            result = cv2.resize(result, (sharp.shape[1], sharp.shape[0]))

        # 计算指标
        psnr_value = psnr(sharp, result)
        ssim_value = ssim(sharp, result)

        # 存储结果
        results.append([blur_img, psnr_value, ssim_value])
        all_psnr.append(psnr_value)
        all_ssim.append(ssim_value)

        print(f"图像 {blur_img}: PSNR = {psnr_value:.2f} dB, SSIM = {ssim_value:.4f}")

    # 计算平均指标
    avg_psnr = np.mean(all_psnr) if all_psnr else 0
    avg_ssim = np.mean(all_ssim) if all_ssim else 0
    avg_time = np.mean(processing_times) if processing_times else 0

    # 计算模型大小
    model_size_mb = 0
    checkpoint_dir = r"D:\objection\python\DeblurProject\models\DeblurGAN-master\checkpoints\experiment_name"
    if os.path.exists(checkpoint_dir):
        for weights_file in glob.glob(os.path.join(checkpoint_dir, "*.pth")):
            file_size = os.path.getsize(weights_file) / (1024 * 1024)
            model_size_mb += file_size
            print(f"模型文件: {weights_file}, 大小: {file_size:.2f} MB")

    # 打印平均指标
    print("\n========= DeblurGAN 性能指标 =========")
    print(f"评估图像数量: {len(image_pairs)}")
    print(f"平均 PSNR: {avg_psnr:.2f} dB")
    print(f"平均 SSIM: {avg_ssim:.4f}")
    print(f"平均处理时间: {avg_time * 1000:.2f} ms/image")
    print(f"模型大小: {model_size_mb:.2f} MB")
    print("======================================")

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
    else:
        print("没有有效的评估结果！")

    # 创建可视化比较
    if image_pairs:
        visual_dir = os.path.join(output_dir, "visual_comparison")
        os.makedirs(visual_dir, exist_ok=True)

        html_path = os.path.join(output_dir, "deblurgan_visual_comparison.html")
        with open(html_path, "w") as f:
            f.write("<html><head><title>DeblurGAN结果可视化比较</title></head><body>\n")
            f.write("<h1>DeblurGAN结果可视化比较</h1>\n")
            f.write("<table border='1'>\n")
            f.write("<tr><th>模糊图像</th><th>清晰参考</th><th>去模糊结果</th><th>PSNR/SSIM</th></tr>\n")

            for i, (blur_img, sharp_img, result_img) in enumerate(image_pairs):
                if i >= 10:  # 限制为前10个以避免文件过大
                    break

                blur_path = os.path.join(input_dir, blur_img)
                sharp_path = os.path.join(reference_dir, sharp_img)
                result_path = os.path.join(result_dir, result_img)

                # 计算当前对的指标
                blur = cv2.imread(blur_path)
                sharp = cv2.imread(sharp_path)
                result = cv2.imread(result_path)

                if blur is not None and sharp is not None and result is not None:
                    if sharp.shape != result.shape:
                        result = cv2.resize(result, (sharp.shape[1], sharp.shape[0]))

                    psnr_val = psnr(sharp, result)
                    ssim_val = ssim(sharp, result)

                    # 复制图像到visual_dir以便HTML引用
                    cv2.imwrite(os.path.join(visual_dir, f"blur_{i}.png"), blur)
                    cv2.imwrite(os.path.join(visual_dir, f"sharp_{i}.png"), sharp)
                    cv2.imwrite(os.path.join(visual_dir, f"result_{i}.png"), result)

                    # 在HTML中添加图像和指标
                    f.write("<tr>\n")
                    f.write(f"<td><img src='visual_comparison/blur_{i}.png' width='256'></td>\n")
                    f.write(f"<td><img src='visual_comparison/sharp_{i}.png' width='256'></td>\n")
                    f.write(f"<td><img src='visual_comparison/result_{i}.png' width='256'></td>\n")
                    f.write(f"<td>PSNR: {psnr_val:.2f} dB<br>SSIM: {ssim_val:.4f}</td>\n")
                    f.write("</tr>\n")

            f.write("</table>\n")
            f.write("</body></html>\n")

        print(f"\n可视化比较已保存到: {html_path}")


if __name__ == "__main__":
    main()