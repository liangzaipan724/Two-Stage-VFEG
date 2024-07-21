import os
from pathlib import Path
import numpy as np
import cv2
import imageio

def find_overall_bbox():#找bounding box
    root_dir_path = "/data/mead/train"
    depth_dir_path = os.path.join(root_dir_path, "Depth")

    data_sum = np.zeros((240, 320), dtype=np.int64)
    for idx, action in enumerate(range(1, 28)):
        print(idx)
        for subject in range(1, 9):
            for trial in range(1, 5):
                data = import_depth_data(depth_dir_path, action, subject, trial)
                if data is not None:
                    data_sum += np.sum(data, axis=2, dtype=np.int64)

def import_depth_data(depth_dir_path, emotion, level, vid):
    filename = os.path.join(depth_dir_path, f'{emotion}/level_{level}/{vid:03}.mp4')
    if Path(filename).is_file():
        video_reader = imageio.get_reader(filename)
        return video_reader
    else:
        return None


def import_rgb_data(rgb_dir_path, emotion, level, vid):
    filename = os.path.join(rgb_dir_path, f'{emotion}/level_{level}/{vid:03}.mp4')
    frame_list = []
    if Path(filename).is_file():
        cap = cv2.VideoCapture(filename)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_list.append(frame[:, :, ::-1])
            else:
                break
        return frame_list
    else:
        return None



def save_crop_image():
    save_path='/data/mead/'
    root_dir_path = "/data/mead/MEAD/"
    save_dir_path = os.path.join(save_path, "crop_image")
    os.makedirs(save_dir_path, exist_ok=True)

    Y_min = 0
    Y_max = 239 + 1
    X_min = 78
    X_max = 245 + 1
    person_idx=os.listdir('/data/mead/MEAD/')
    person_id= ['M011', 'W033', 'M033', 'M032', 'W026']
    print(person_id)
    for i in person_id:
            print(i)
            x = os.path.join('/data/mead/MEAD/', i)
            folder_path2 = os.path.join(x, 'video')
            folder_path = os.path.join(folder_path2, 'front')
            file_names = os.listdir(folder_path)
            for idx, emotion in enumerate(file_names):
                # print(emotion)
                if emotion=='neutral':
                    level=1
                    xxx=os.listdir(os.path.join(folder_path, emotion, 'level_1'))
                    for vid in range(0, len(xxx)):
                        data_path = os.path.join(root_dir_path, i, 'video', 'front')
                        depth_data = import_depth_data(data_path, emotion, level, vid)
                        if depth_data is None:
                            continue
                        frame_list = import_rgb_data(data_path, emotion, level, vid)

                        image_dir_name = f'{emotion}level_{level}_{vid:03}'

                        image_dir_path = os.path.join(save_dir_path,i, image_dir_name)
                        os.makedirs(image_dir_path, exist_ok=True)
                        for image_idx, crop_frame in enumerate(frame_list):
                            if image_idx%5==0:
                                crop_frame_name = image_dir_name + "_%03d.png" % image_idx
                                crop_frame_path = os.path.join(image_dir_path, crop_frame_name)
                                imageio.imsave(crop_frame_path, crop_frame)
                else:
                    for level in range(1, 4):
                        # print(level)
                        for vid in range(0, len(os.listdir(os.path.join(folder_path,emotion,f'level_{level}')))):
                                data_path = os.path.join(root_dir_path, i,'video','front')
                                # print(data_path)
                                depth_data = import_depth_data(data_path,emotion, level, vid)
                                if depth_data is None:
                                    continue
                                frame_list = import_rgb_data(data_path, emotion, level, vid)

                                image_dir_name = f'{emotion}level_{level}_{vid:03}'
                                image_dir_path = os.path.join(save_dir_path, i,image_dir_name)
                                os.makedirs(image_dir_path, exist_ok=True)
                                for image_idx, crop_frame in enumerate(frame_list):
                                    if image_idx%5==0:
                                        crop_frame_name = image_dir_name + "_%03d.png" % image_idx
                                        crop_frame_path = os.path.join(image_dir_path, crop_frame_name)
                                        imageio.imsave(crop_frame_path, crop_frame)


if __name__ == "__main__":

    pass
