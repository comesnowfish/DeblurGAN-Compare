import os
import shutil


def reorganize_dataset(source_root, output_root):
    """
    将原始数据集从复杂结构重构为简化结构。

    :param source_root: 原始数据集的根目录
    :param output_root: 输出重构后的数据集根目录
    """

    # 设置目标路径
    train_blur_dir = os.path.join(output_root, "train", "blur")
    train_sharp_dir = os.path.join(output_root, "train", "sharp")
    test_blur_dir = os.path.join(output_root, "test", "blur")
    test_sharp_dir = os.path.join(output_root, "test", "sharp")

    # 创建目标文件夹
    os.makedirs(train_blur_dir, exist_ok=True)
    os.makedirs(train_sharp_dir, exist_ok=True)
    os.makedirs(test_blur_dir, exist_ok=True)
    os.makedirs(test_sharp_dir, exist_ok=True)

    # 遍历原始数据集中的目录
    for dirpath, dirnames, filenames in os.walk(source_root):
        if "blur" in dirpath:
            # 对应的清晰图像文件夹
            corresponding_sharp_dir = dirpath.replace("blur", "sharp")

            # 判断目录是否包含清晰图像
            if os.path.exists(corresponding_sharp_dir):
                # 获取图像文件并整理
                blur_files = sorted([f for f in filenames if f.endswith(".png")])
                sharp_files = sorted([f for f in os.listdir(corresponding_sharp_dir) if f.endswith(".png")])

                for blur_file, sharp_file in zip(blur_files, sharp_files):
                    blur_path = os.path.join(dirpath, blur_file)
                    sharp_path = os.path.join(corresponding_sharp_dir, sharp_file)

                    # 判断文件属于测试集还是训练集
                    if "test" in dirpath:
                        shutil.copy(blur_path, os.path.join(test_blur_dir, blur_file))
                        shutil.copy(sharp_path, os.path.join(test_sharp_dir, sharp_file))
                    else:
                        shutil.copy(blur_path, os.path.join(train_blur_dir, blur_file))
                        shutil.copy(sharp_path, os.path.join(train_sharp_dir, sharp_file))

    print(f"Dataset reorganized and saved to {output_root}")


# 输入原始数据集路径和目标输出路径
source_root = "D:\\objection\\python\\DeblurProject\\data\\GOPRO_Large"  # 你的原始数据集路径
output_root = "D:\\objection\\python\\DeblurProject\\data"  # 输出路径

# 调用函数进行数据集重构
reorganize_dataset(source_root, output_root)
