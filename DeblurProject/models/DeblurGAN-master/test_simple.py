# 保存为 D:\objection\python\DeblurProject\models\DeblurGAN-master\simple_test.py

import os
import torch
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from PIL import Image
import numpy as np


def tensor2im(input_image, imtype=np.uint8):
    """将张量转换为PIL图像"""
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def main():
    # 1. 初始化配置
    opt = TestOptions().parse()
    opt.num_threads = 0  # 强制单进程
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True

    # 2. 创建数据加载器
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    # 3. 初始化模型
    model = create_model(opt)

    # 4. 创建结果目录
    results_dir = opt.results_dir if opt.results_dir else os.path.join('results', opt.name)
    output_dir = os.path.join(results_dir, f'{opt.phase}_{opt.which_epoch}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"结果将保存到: {output_dir}")

    # 5. 测试循环
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break

        try:
            # 5.1 模型推理
            model.set_input(data)
            model.test()

            # 5.2 获取结果和文件名
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()
            short_path = os.path.basename(img_path[0])
            name = os.path.splitext(short_path)[0]

            print(f'处理图像: {img_path}')

            # 5.3 保存结果
            for label, image in visuals.items():
                image_numpy = tensor2im(image)
                image_name = f'{name}_{label}.png'
                save_path = os.path.join(output_dir, image_name)
                Image.fromarray(image_numpy).save(save_path)
                print(f'  保存: {save_path}')

        except Exception as e:
            print(f'处理图像时出错: {str(e)}')
            continue

    print(f'所有图像处理完成，结果保存在: {output_dir}')


if __name__ == '__main__':
    main()