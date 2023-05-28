from glob import glob
from skimage.io import imread
import os
import numpy as np
from tqdm import tqdm

full_dose_path = '/home/zhengxin/mnt/projects/ct-reconstruction/dataset/TestSet/Full_Dose'
quarter_dose_path = '/home/zhengxin/mnt/projects/ct-reconstruction/dataset/TestSet/Quarter_Dose'
save_dir_prefix = '/home/zhengxin/mnt/projects/ct-reconstruction/processed_data/test'

full_dose_person_list = sorted(os.listdir(full_dose_path))
quarter_dose_person_list = sorted(os.listdir(quarter_dose_path))

processed_fd_list = []
processed_qd_list = []

assert full_dose_person_list == quarter_dose_person_list

for person in full_dose_person_list:
    full_dose_img_list = sorted(glob(os.path.join(full_dose_path, person, '*.png')))
    quarter_dose_img_list = sorted(glob(os.path.join(quarter_dose_path, person, '*.png')))
    # 序号的index为26, 27, 28, 29
    for i, full_dose_img_file in enumerate(tqdm(full_dose_img_list, desc=person)):
        assert full_dose_img_file.split('/')[-1][26:30] == quarter_dose_img_list[i].split('/')[-1][26:30]

        processed_fd_list.append(imread(full_dose_img_file)[np.newaxis,np.newaxis, ...] / 255.0)
        processed_qd_list.append(imread(quarter_dose_img_list[i])[np.newaxis,np.newaxis, ...] / 255.0)

        assert processed_fd_list[-1].shape == processed_qd_list[-1].shape
        assert processed_fd_list[-1].shape == (1, 1, 256, 256)

processed_fd_list = np.concatenate(processed_fd_list, axis=0)
processed_qd_list = np.concatenate(processed_qd_list, axis=0)

# 保存结果
np.save(os.path.join(save_dir_prefix, 'full_dose.npy'), processed_fd_list)
np.save(os.path.join(save_dir_prefix, 'quarter_dose.npy'), processed_qd_list)