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
    def __init__(self, weights_path: str, model_name: str = 'fpn_ghostnet_gm_hin', cuda: bool = True):
        print(f"初始化Predictor，使用模型: {model_name}")

        # 创建模型配置字典，避免传递字符串引起的问题
        model_config = {
            'g_name': model_name,
            'norm_layer': 'hin',
            'learn_residual': True,
            'dropout': True,
            'blocks': 9
        }

        try:
            model = get_generator(model_config, cuda=cuda)
            print(f"模型初始化成功")
        except Exception as e:
            print(f"初始化生成器模型时出错: {str(e)}")
            raise

        # 加载权重
        try:
            weights = torch.load(weights_path, map_location='cuda' if cuda else 'cpu')
            if 'model' in weights:
                model.load_state_dict(weights['model'])
            else:
                model.load_state_dict(weights)
            print(f"成功加载权重: {weights_path}")
        except Exception as e:
            print(f"加载权重时出错: {str(e)}")
            raise

        self.model = model.module.cpu() if not cuda else model.cuda()
        self.cuda = cuda
        self.model.train(True)
        # GAN inference should be in train mode to use actual stats in norm layers,
        # it's not a bug
        self.normalize_fn = get_normalize()

    @staticmethod
    def _array_to_batch(x):
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    def _preprocess(self, x: np.ndarray, mask: Optional[np.ndarray]):
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

        return map(self._array_to_batch, (x, mask)), h, w

    @staticmethod
    def _postprocess(x: torch.Tensor) -> np.ndarray:
        x, = x
        x = x.detach().cpu().float().numpy()
        x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
        return x.astype('uint8')

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray], ignore_mask=True) -> np.ndarray:
        (img, mask), h, w = self._preprocess(img, mask)
        with torch.no_grad():
            inputs = [img.cuda() if self.cuda else img.cpu()]
            if not ignore_mask:
                inputs += [mask]
            pred = self.model(*inputs)
        return self._postprocess(pred)[:h, :w, :]


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
         weights_path='trained_weights/fpn_ghostnet_gm_hin.h5',
         out_dir='D:/objection/python/DeblurProject/results/Ghost-DeblurGAN_results',
         side_by_side: bool = False,
         video: bool = False,
         cuda: bool = True,
         model_name: str = 'fpn_ghostnet_gm_hin'):
    # 增加网络超时时间
    try:
        import timm.models._hub
        timm.models._hub.GITHUB_TIMEOUT = 30
        timm.models._hub.HF_TIMEOUT = 30
        print("已增加网络超时时间")
    except:
        print("无法修改timm库超时设置")

    # 打印当前目录检查
    print(f"当前工作目录: {os.getcwd()}")
    print(f"检查权重文件: {weights_path}")
    if not os.path.exists(weights_path):
        alt_weights_path = os.path.join(os.path.dirname(__file__), weights_path)
        if os.path.exists(alt_weights_path):
            weights_path = alt_weights_path
            print(f"找到权重文件: {weights_path}")
        else:
            print(f"警告: 找不到权重文件 {weights_path} 或 {alt_weights_path}")
            print(f"可用权重文件:")
            for root, dirs, files in os.walk(os.path.dirname(__file__)):
                for file in files:
                    if file.endswith('.h5'):
                        print(f" - {os.path.join(root, file)}")

    def sorted_glob(pattern):
        return sorted(glob(pattern))

    imgs = sorted_glob(img_pattern)
    if not imgs:
        print(f"警告: 没有找到匹配 '{img_pattern}' 的图像文件")
        return

    masks = sorted_glob(mask_pattern) if mask_pattern is not None else [None for _ in imgs]
    pairs = zip(imgs, masks)
    names = sorted([os.path.basename(x) for x in glob(img_pattern)])

    try:
        os.environ['TORCH_HOME'] = os.path.join(os.path.dirname(__file__), 'pretrained')
        print(f"设置TORCH_HOME为: {os.environ['TORCH_HOME']}")

        predictor = Predictor(weights_path=weights_path, model_name=model_name, cuda=cuda)
    except Exception as e:
        print(f"初始化预测器时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    os.makedirs(out_dir, exist_ok=True)
    print(f"输出目录: {out_dir}")

    if not video:
        for name, pair in tqdm(zip(names, pairs), total=len(names)):
            f_img, f_mask = pair
            img = cv2.imread(f_img)
            if img is None:
                tqdm.write(f"警告: 无法读取图像 {f_img}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = None if f_mask is None else cv2.imread(f_mask)

            pred = predictor(img, mask)
            if side_by_side:
                pred = np.hstack((img, pred))
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

            output_path = os.path.join(out_dir, name)
            cv2.imwrite(output_path, pred)
            tqdm.write(f"处理完成 {name} -> {output_path}")
    else:
        process_video(pairs, predictor, out_dir)


if __name__ == '__main__':
    Fire(main)