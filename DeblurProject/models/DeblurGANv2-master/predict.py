import os
from glob import glob
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from fire import Fire
from tqdm import tqdm

from aug import get_normalize
from models.networks import get_generator


class Predictor:
    def __init__(self, weights_path: str, model_name: str = ''):
        print("初始化模型...")
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/config.yaml')
        with open(config_path, encoding='utf-8') as cfg:
            config = yaml.load(cfg, Loader=yaml.FullLoader)

        model = get_generator(model_name or config['model'])

        print(f"加载权重文件: {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))['model'])

        # 自动检测设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        self.model = model.to(self.device)
        self.model.train(True)
        # GAN inference should be in train mode to use actual stats in norm layers,
        # it's not a bug
        self.normalize_fn = get_normalize()
        print("模型初始化完成")

    @staticmethod
    def _array_to_batch(x):
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    def _preprocess(self, x: np.ndarray, mask: Optional[np.ndarray]):
        print(f"预处理图像，原始尺寸: {x.shape}")
        x, _ = self.normalize_fn(x, x)
        if mask is None:
            mask = np.ones_like(x, dtype=np.float32)
        else:
            mask = np.round(mask.astype('float32') / 255)

        h, w, _ = x.shape
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size

        pad_params = {'mode': 'constant',
                      'constant_values': 0,
                      'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
                      }
        x = np.pad(x, **pad_params)
        mask = np.pad(mask, **pad_params)

        print(f"图像已填充到: {x.shape}")
        return map(self._array_to_batch, (x, mask)), h, w

    @staticmethod
    def _postprocess(x: torch.Tensor) -> np.ndarray:
        x, = x
        x = x.detach().cpu().float().numpy()
        x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
        return x.astype('uint8')

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray], ignore_mask=True) -> np.ndarray:
        print("开始处理图像...")
        (img, mask), h, w = self._preprocess(img, mask)
        with torch.no_grad():
            # 将所有输入移至同一设备
            inputs = [img.to(self.device)]
            if not ignore_mask:
                inputs += [mask.to(self.device)]

            print("开始模型推理...")
            pred = self.model(*inputs)
            print("模型推理完成")

        print("后处理结果...")
        result = self._postprocess(pred)[:h, :w, :]
        print(f"处理完成，输出尺寸: {result.shape}")
        return result


def process_video(pairs, predictor, output_dir):
    for video_filepath, mask in tqdm(pairs):
        video_filename = os.path.basename(video_filepath)
        output_filepath = os.path.join(output_dir, os.path.splitext(video_filename)[0] + '_deblur.mp4')
        video_in = cv2.VideoCapture(video_filepath)
        fps = video_in.get(cv2.CAP_PROP_FPS)
        width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frame_num = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))
        video_out = cv2.VideoWriter(output_filepath, cv2.VideoWriter_fourcc(*'MP4V'), fps, (width, height))
        tqdm.write(f'process {video_filepath} to {output_filepath}, {fps}fps, resolution: {width}x{height}')
        for frame_num in tqdm(range(total_frame_num), desc=video_filename):
            res, img = video_in.read()
            if not res:
                break
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred = predictor(img, mask)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            video_out.write(pred)


def main(img_pattern: str,
         mask_pattern: Optional[str] = None,
         weights_path='weights/fpn_inception.h5',
         out_dir='results/',
         side_by_side: bool = False,
         video: bool = False):
    print(f"处理图像: {img_pattern}")
    print(f"权重文件: {weights_path}")
    print(f"输出目录: {out_dir}")

    def sorted_glob(pattern):
        files = sorted(glob(pattern))
        print(f"找到 {len(files)} 个文件匹配模式 '{pattern}'")
        return files

    imgs = sorted_glob(img_pattern)
    if len(imgs) == 0:
        print(f"错误: 没有找到匹配 '{img_pattern}' 的文件")
        return

    masks = sorted_glob(mask_pattern) if mask_pattern is not None else [None for _ in imgs]
    pairs = zip(imgs, masks)
    names = sorted([os.path.basename(x) for x in glob(img_pattern)])

    print("初始化预测器...")
    predictor = Predictor(weights_path=weights_path)

    os.makedirs(out_dir, exist_ok=True)
    print(f"创建输出目录: {out_dir}")

    if not video:
        print(f"开始处理 {len(names)} 张图像...")
        for name, pair in tqdm(zip(names, pairs), total=len(names)):
            f_img, f_mask = pair
            print(f"处理图像: {f_img}")

            try:
                img = cv2.imread(f_img)
                if img is None:
                    print(f"错误: 无法读取图像 {f_img}")
                    continue

                mask = None if f_mask is None else cv2.imread(f_mask)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                pred = predictor(img, mask)
                if side_by_side:
                    pred = np.hstack((img, pred))
                pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

                output_path = os.path.join(out_dir, name)
                cv2.imwrite(output_path, pred)
                print(f"图像已保存到: {output_path}")
            except Exception as e:
                print(f"处理图像 {f_img} 时出错: {str(e)}")
    else:
        print("处理视频...")
        process_video(pairs, predictor, out_dir)


def process_batch():
    """批量处理图片"""
    image_list = get_files()
    print(f"找到 {len(image_list)} 张图片需要批量处理")
    for img_path in image_list:
        try:
            print(f"处理图像: {img_path}")
            main(img_path)
        except Exception as e:
            print(f"处理 {img_path} 时出错: {str(e)}")


def get_files():
    list = []
    for filepath, dirnames, filenames in os.walk(r'.\dataset1\blur'):
        for filename in filenames:
            list.append(os.path.join(filepath, filename))
    return list


if __name__ == '__main__':
    Fire(main)
    # 取消下面的注释以启用批量处理：
    # process_batch()