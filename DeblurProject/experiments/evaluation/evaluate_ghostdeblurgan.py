# 保存为 D:\objection\python\DeblurProject\experiments\evaluation\evaluate_ghostdeblurgan.py

import os
import cv2
import numpy as np
import time
import sys
import glob
from tabulate import tabulate

# 添加项目根目录到路径
sys.path.append("D:\\objection\\python\\DeblurProject")
# 添加Ghost-DeblurGAN到路径
sys.path.append("D:\\objection\\python\\DeblurProject\\models\\Ghost-DeblurGAN-main")

# 从Ghost-DeblurGAN导入预测器，如果导入有问题，确保路径正确
try:
    from predict import Predictor
except ImportError:
    print("错误：无法导入Ghost-DeblurGAN的Predictor类")
    print("请确保Ghost-DeblurGAN路径正确，并且包含predict.py文件")
    sys.exit(1)


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


def find_matching_pairs(input_dir, result_dir, reference_dir=None):
    """查找匹配的图像对（模糊/清晰/结果）用于评估"""
    pairs = []

    # 如果没有提供参考目录，使用输入目录
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

    if not os.path.exists(reference_dir):
        print(f"错误: 参考目录不存在 - {reference_dir}")
        return pairs

    # 常见的模糊-清晰图像对应规则
    pair_rules = {
        "test1_blur.jpg": "test1_sharp.jpg",
        "yolo_b.jpg": "yolo_s.jpg"
    }

    # 查找所有的模糊输入图像
    blur_images = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"找到 {len(blur_images)} 个输入图像")

    # 查找所有的结果图像
    result_images = [f for f in os.listdir(result_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"找到 {len(result_images)} 个结果图像")

    # 按文件名排序
    blur_images.sort()
    result_images.sort()

    # 逐一匹配图像
    for blur_img in blur_images:
        # 查找对应的清晰参考图像
        sharp_img = None

        # 使用规则查找
        if blur_img in pair_rules:
            sharp_candidate = pair_rules[blur_img]
            if os.path.exists(os.path.join(reference_dir, sharp_candidate)):
                sharp_img = sharp_candidate
                print(f"根据规则找到配对: {blur_img} -> {sharp_img}")

        # 如果没找到，尝试命名模式查找
        if sharp_img is None:
            blur_base = os.path.splitext(blur_img)[0].split('_')[0]
            for f in os.listdir(reference_dir):
                if "sharp" in f.lower() and blur_base in f:
                    sharp_img = f
                    print(f"根据命名模式找到配对: {blur_img} -> {sharp_img}")
                    break

        # GOPRO数据集匹配规则：同名不同目录
        if sharp_img is None and reference_dir != input_dir:
            if os.path.exists(os.path.join(reference_dir, blur_img)):
                sharp_img = blur_img
                print(f"根据同名原则找到配对: {blur_img} (在参考目录中)")

        # 查找对应的去模糊结果
        result_img = None

        # 尝试在结果目录中查找同名文件
        if os.path.exists(os.path.join(result_dir, blur_img)):
            result_img = blur_img
            print(f"在结果目录中找到同名图像: {blur_img}")
        else:
            # 尝试查找包含输入文件名的结果文件
            blur_name = os.path.splitext(blur_img)[0]
            for res_img in result_images:
                if blur_name in os.path.splitext(res_img)[0]:
                    result_img = res_img
                    print(f"找到包含输入名称的结果: {blur_img} -> {result_img}")
                    break

        # 如果找到了清晰参考图像和结果，添加到评估对中
        if sharp_img and result_img:
            pairs.append((blur_img, sharp_img, result_img))
            print(f"添加评估对 #{len(pairs)}: {blur_img}, {sharp_img}, {result_img}")

    print(f"总共找到 {len(pairs)} 个有效的图像评估对")
    return pairs


def process_dataset(input_dir, output_dir, weights_path):
    """使用Ghost-DeblurGAN处理数据集中的所有图像"""
    print(f"使用Ghost-DeblurGAN处理数据集: {input_dir}")
    print(f"输出目录: {output_dir}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 初始化预测器
    try:
        predictor = Predictor(weights_path=weights_path)
    except Exception as e:
        print(f"初始化Ghost-DeblurGAN预测器失败: {str(e)}")
        return False

    # 查找所有模糊图像
    blur_images = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not blur_images:
        print(f"错误: 在 {input_dir} 中未找到图像")
        return False

    print(f"找到 {len(blur_images)} 个图像需要处理")

    # 处理每张图像
    success_count = 0
    for blur_img in blur_images:
        try:
            img_path = os.path.join(input_dir, blur_img)
            output_path = os.path.join(output_dir, blur_img)

            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                print(f"错误: 无法读取图像 {img_path}")
                continue

            # 转换颜色空间
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 使用Ghost-DeblurGAN处理图像
            print(f"处理图像: {blur_img}")
            result = predictor(img_rgb, None)

            # 转换回BGR并保存
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_bgr)

            print(f"已保存处理结果到: {output_path}")
            success_count += 1

        except Exception as e:
            print(f"处理图像 {blur_img} 时出错: {str(e)}")

    print(f"成功处理 {success_count}/{len(blur_images)} 张图像")
    return success_count > 0


def main():
    # ============== 路径设置 ==============
    # Ghost-DeblurGAN模型路径
    # ghost_deblurgan_dir = "D:\\objection\\python\\DeblurProject\\models\\Ghost-DeblurGAN-main"
    weights_path = None

    # 自动查找权重文件
    # weight_files = glob.glob(os.path.join(ghost_deblurgan_dir, "**", "*.pth"), recursive=True)
    # if weight_files:
        # 选择最新的权重文件
        # weights_path = sorted(weight_files, key=os.path.getmtime, reverse=True)[0]
        # print(f"找到权重文件: {weights_path}")
    # else:
        # print("错误: 未找到权重文件，请指定正确的路径")
        # sys.exit(1)

    # Ghost-DeblurGAN模型路径
    ghost_deblurgan_dir = r"D:\objection\python\DeblurProject\models\Ghost-DeblurGAN-main"
    # 直接指定权重文件
    weights_path = r"D:\objection\python\DeblurProject\models\Ghost-DeblurGAN-main\trained_weights\fpn_ghostnet_gm_hin.h5"

    # 检查权重文件是否存在
    if not os.path.exists(weights_path):
        print(f"错误: 权重文件不存在 - {weights_path}")
        # 尝试寻找备选权重文件
        alt_weight_path = r"D:\objection\python\DeblurProject\models\Ghost-DeblurGAN-main\trained_weights\fpn_mobilenet_v2.h5"
        if os.path.exists(alt_weight_path):
            print(f"找到备选权重文件: {alt_weight_path}")
            weights_path = alt_weight_path
        else:
            print("错误: 未找到任何可用的权重文件，请指定正确的路径")
            sys.exit(1)

    # 测试数据集路径
    input_dir = "D:\\objection\\python\\DeblurProject\\data\\test\\blur"
    # 参考清晰图像路径
    reference_dir = "D:\\objection\\python\\DeblurProject\\data\\test\\sharp"
    # 结果输出路径
    result_dir = "D:\\objection\\python\\DeblurProject\\results\\Ghost-DeblurGAN"
    # 评估结果保存路径
    output_dir = "D:\\objection\\python\\DeblurProject\\results"

    # 1. 首先使用Ghost-DeblurGAN处理数据集
    if not os.path.exists(result_dir) or len(os.listdir(result_dir)) == 0:
        print("需要先生成Ghost-DeblurGAN的去模糊结果")
        success = process_dataset(input_dir, result_dir, weights_path)
        if not success:
            print("处理数据集失败，无法继续评估")
            return

    print(f"\n评估Ghost-DeblurGAN的性能...\n输入目录: {input_dir}\n结果目录: {result_dir}\n参考目录: {reference_dir}")

    # 2. 寻找匹配的图像对进行评估
    image_pairs = find_matching_pairs(input_dir, result_dir, reference_dir)

    if not image_pairs:
        print("未找到可用于评估的图像对！")
        return

    # 3. 存储结果
    results = []
    all_psnr = []
    all_ssim = []
    output_file = os.path.join(output_dir, "ghostdeblurgan_metrics.csv")

    # 4. 评估每对图像
    print("\n开始评估Ghost-DeblurGAN性能...")
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

        # 存储结果
        results.append([blur_img, psnr_value, ssim_value])
        all_psnr.append(psnr_value)
        all_ssim.append(ssim_value)

        print(f"图像 {blur_img}: PSNR = {psnr_value:.2f} dB, SSIM = {ssim_value:.4f}")

    # 5. 计算平均指标
    avg_psnr = np.mean(all_psnr) if all_psnr else 0
    avg_ssim = np.mean(all_ssim) if all_ssim else 0

    # 6. 计算模型大小
    model_size_mb = 0
    if weights_path and os.path.exists(weights_path):
        model_size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        print(f"模型文件: {weights_path}, 大小: {model_size_mb:.2f} MB")

    # 7. 打印平均指标
    print("\n========= Ghost-DeblurGAN 性能指标 =========")
    print(f"平均 PSNR: {avg_psnr:.2f} dB")
    print(f"平均 SSIM: {avg_ssim:.4f}")
    print(f"模型大小: {model_size_mb:.2f} MB")
    print("============================================")

    # 8. 打印每个图像的结果表格
    if results:
        print("\n每个图像的详细指标:")
        headers = ["图像", "PSNR (dB)", "SSIM"]
        print(tabulate(results, headers=headers, tablefmt="grid"))

        # 9. 保存结果到CSV
        with open(output_file, "w") as f:
            f.write("Image,PSNR,SSIM\n")
            for row in results:
                f.write(f"{row[0]},{row[1]:.2f},{row[2]:.4f}\n")
            f.write(f"Average,{avg_psnr:.2f},{avg_ssim:.4f}\n")

        print(f"\n结果已保存到: {output_file}")
    else:
        print("没有有效的评估结果！")


if __name__ == "__main__":
    main()