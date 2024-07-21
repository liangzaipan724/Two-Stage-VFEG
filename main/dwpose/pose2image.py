import os

import cv2
from PIL import Image
import numpy as np

from DM.dwpose import DWposeDetector
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def HWC3(x):
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

if __name__ == '__main__':
    device = torch.device("cuda:4" if torch.cuda.is_available() else 'cpu')
    data_dir = '/data/mead/crop_image/'
    person_name1 = os.listdir(data_dir)
    person_name = person_name1[0:9]
    person_name=['M011']
    print(person_name)
    for i in person_name:
        print(i)
        video_name_list = os.listdir(os.path.join(data_dir, i))  #
        video_path_list = [os.path.join(data_dir, i, video_name) for video_name in video_name_list]
        for j in video_path_list:
            x = os.listdir(j)
            x=sorted(x)
            for png_name in x:
                pil_im = Image.open(os.path.join(data_dir, i, j, png_name))

                 # print(pil_im.mode)          #rgb
                np_array = np.asarray(pil_im)
                # np_array=np_array.astype(np.uint8)
                if np_array.dtype!=np.uint8:
                    continue
                #assert np_array.dtype == np.uint8
                image = HWC3(np_array)
                        #
                dwprocessor = DWposeDetector()                    # apply_openpose = OpenposeDetector()
                detected_map = dwprocessor(image)

                j_name = j.split('/')
                # print(j)
                j_1 = j_name[-1]
                # print(j_1)
                image = Image.fromarray(detected_map)

                file_path = os.path.join('/data/pose1/', i, j_1, f'{i}+{j_1}+{png_name}+dwpose.png')
                if not os.path.exists(os.path.join('/data/pose1/', i, j_1)):
                    os.makedirs(os.path.join('/data/pose1/', i, j_1))
                image.save(file_path)  # 保存为PNG格式
