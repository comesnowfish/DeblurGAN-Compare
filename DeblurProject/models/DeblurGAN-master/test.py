import os
import torch
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util import html
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
    web_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.which_epoch}')
    os.makedirs(web_dir, exist_ok=True)
    webpage = html.HTML(web_dir, f'Test Results - {opt.name}')

    # 5. 测试循环
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break

        try:
            # 5.1 模型推理
            model.set_input(data)
            model.test()  # 这个方法应该包含前向传播

            # 5.2 获取并保存结果
            visuals = model.get_current_visuals()
            for label, image in visuals.items():
                img_path = model.get_image_paths()
                print(f'Processing: {img_path}')

                # 转换张量为图像
                if isinstance(image, torch.Tensor):
                    image = tensor2im(image)

                # 保存图像
                save_path = os.path.join(web_dir, f'{i}_{label}.png')
                Image.fromarray(image).save(save_path)

        except Exception as e:
            print(f'Error processing image: {str(e)}')
            continue

    # 6. 生成结果网页
    webpage.save()
    print(f'Results saved to: {web_dir}')


if __name__ == '__main__':
    main()