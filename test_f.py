from dataset.ct import CT
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import piq
from fvcore.nn import FlopCountAnalysis, flop_count_table


device = 'cuda:0'

from net import RED_CNN
gen = RED_CNN(out_channels=16, num_layers=10, kernel_size=5, padding=0).to(device)

ckpt_path = './final_ckpt.pth'

gen.load_state_dict(torch.load(ckpt_path))
gen.eval()

print('loading dataset')

testset = CT('/home/zhengxin/mnt/projects/ct-reconstruction/processed_data/test', transform=False)
dataloder = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

def test(loader):
    psnr_sum = 0.
    ssim_num = 0.
    with torch.no_grad():
        for i, (input, target) in enumerate(tqdm(loader)):
            input = input.to(device).float()
            target = target.to(device).float()
            output = torch.clip(gen(input), min=0., max=1.)
            psnr_of_one_batch = piq.psnr(output, target, data_range=1.0, reduction='none')
            psnr_sum += torch.sum(psnr_of_one_batch).item()
            ssim_num += piq.ssim(output, target, data_range=1.0, reduction='sum')
        print(f'mean_psnr: {psnr_sum / len(loader.dataset)}, mean_ssim: {ssim_num / len(loader.dataset)}')
    return psnr_sum / len(loader.dataset)

def get_params_and_flops(model, input_shape=(1, 1, 256, 256)):
    flops= FlopCountAnalysis(model, torch.randn(input_shape).to(device))
    print("Total params: {:.5f}M".format(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6))
    print(f"Total GFLOPs: {flops.total() / (10 ** 9)}")

test(dataloder)
get_params_and_flops(gen)
