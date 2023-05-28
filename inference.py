from glob import glob
from skimage.io import imread, imsave
import os
import numpy as np
from tqdm import tqdm
import torch

desc = 'final'
device = 'cuda:0'


# full_dose_path = '/home/zhengxin/mnt/projects/ct-reconstruction/dataset/TestSet/Full_Dose'
quarter_dose_path = '/home/zhengxin/mnt/projects/ct-reconstruction/dataset/TestSet/Quarter_Dose'
save_dir_prefix = '/home/zhengxin/mnt/projects/ct-reconstruction/processed_data/result'
save_dir = os.path.join(save_dir_prefix, desc)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# full_dose_person_list = sorted(os.listdir(full_dose_path))
quarter_dose_person_list = sorted(os.listdir(quarter_dose_path))


from net import RED_CNN
gen = RED_CNN(out_channels=16, num_layers=10, kernel_size=5, padding=0).to(device)
ckpt_path = './final_ckpt.pth'




gen.load_state_dict(torch.load(ckpt_path))
gen.eval()

with torch.no_grad():
    for person in quarter_dose_person_list:
        # full_dose_img_list = sorted(glob(os.path.join(full_dose_path, person, '*.png')))
        quarter_dose_img_list = sorted(glob(os.path.join(quarter_dose_path, person, '*.png')))
        # 序号的index为26, 27, 28, 29
        for i, quarter_dose_img_file in enumerate(tqdm(quarter_dose_img_list, desc=person)):
            # print(os.path.basename(quarter_dose_img_file))
            # input_img = imread(quarter_dose_img_file)
            # print(np.min(input_img), np.max(input_img), input_img.dtype)    
            input_img = imread(quarter_dose_img_file)[np.newaxis,np.newaxis, ...] / 255.0
            input_img = torch.from_numpy(input_img).to(device).float()
            output_img = gen(input_img).clamp(0., 1.) * 255.0
            output_img = output_img.squeeze(0, 1).cpu().numpy().astype(np.uint8)
            assert output_img.shape == (256, 256)
            # 写入文件
            save_name = os.path.basename(quarter_dose_img_file).replace('.png', '_Rec.png')
            save_path = os.path.join(save_dir, save_name)
            imsave(save_path, output_img)