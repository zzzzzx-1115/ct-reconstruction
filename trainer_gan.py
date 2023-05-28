import torch
from torch.utils.data import DataLoader
from dataset.ct import CT
# from net.conv_unet import ConvUNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import datetime
import piq
import os
from utils.loss import Weighted_MSE_SSIM, ls_gan
import net
from fvcore.nn import FlopCountAnalysis, flop_count_table


class Trainer:
    def __init__(self, configs):
        # trainingset = CT('/home/zhengxin/mnt/projects/ct-reconstruction/processed_data/train')
        # validationset = CT('/home/zhengxin/mnt/projects/ct-reconstruction/processed_data/valid')
        print(configs)
        g = getattr(net, configs['generator_name'])
        d = getattr(net, configs['discriminator_name'])

        self.gen = g(**configs['generator']).to(configs['device']).float()
        self.dis = d(**configs['discriminator']).to(configs['device']).float()

        flops= FlopCountAnalysis(self.gen, torch.randn(1, 1, 256, 256).to(configs['device']))
        print("Total params: {:.5f}M".format(sum(p.numel() for p in self.gen.parameters() if p.requires_grad) / 1e6))
        print(f"Total GFLOPs: {flops.total() / (10 ** 9)}")



        self.configs = configs
        self.img_loss = Weighted_MSE_SSIM(configs['loss_alpha'], scale=configs['loss_scale']).to(configs['device'])
        self.discriminative_loss = ls_gan
        self.writer = SummaryWriter(log_dir=configs['log_dir'])
        # 优化器
        self.g_optimizer = torch.optim.Adam(self.gen.parameters(), lr=configs['g_lr'])
        self.d_optimizer = torch.optim.Adam(self.dis.parameters(), lr=configs['d_lr'])
        self.initialization()
        # save path
        if not os.path.exists(configs['save_dir']):
            os.makedirs(configs['save_dir'])
        if not os.path.exists(configs['log_dir']):
            os.makedirs(configs['log_dir'])

        self.init_params()

        trainingset = CT(**configs['training_data'])
        validationset = CT(**configs['validation_data'])
        testset = CT(**configs['test_data'])
        # testset = CT('/home/zhengxin/mnt/projects/ct-reconstruction/processed_data/test')

        self.train_loader = DataLoader(trainingset, batch_size=configs['batch_size'], shuffle=True, num_workers=4)
        self.validation_loader = DataLoader(validationset, batch_size=configs['batch_size'], shuffle=True,
                                            num_workers=4)
        self.test_loader = DataLoader(testset, batch_size=configs['batch_size'], shuffle=True, num_workers=4)



    def initialization(self):
        if self.configs['checkpoint'] is not None:
            ckpt = torch.load(self.configs['checkpoint'])

            self.gen.load_state_dict(ckpt['gen_state_dict'])
            self.dis.load_state_dict(ckpt['dis_state_dict'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer_state_dict'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer_state_dict'])
            print('load model from %s' % self.configs['checkpoint'])
            print(ckpt['configs'])
            print(ckpt['desc'])
        else:
            print('init model')

    def init_params(self):
        for param in self.gen.parameters():
            if isinstance(param, nn.Conv2d):
                nn.init.xavier_uniform_(param.weight.data)
                nn.init.constant_(param.bias.data, 0.1)
            elif isinstance(param, nn.BatchNorm2d):
                param.weight.data.fill_(1)
                param.bias.data.zero_()
            elif isinstance(param, nn.Linear):
                param.weight.data.normal_(0, 0.01)
                param.bias.data.zero_()

        for param in self.gen.parameters():
            if isinstance(param, nn.Conv2d):
                nn.init.xavier_uniform_(param.weight.data)
                nn.init.constant_(param.bias.data, 0.1)
            elif isinstance(param, nn.BatchNorm2d):
                param.weight.data.fill_(1)
                param.bias.data.zero_()
            elif isinstance(param, nn.Linear):
                param.weight.data.normal_(0, 0.01)
                param.bias.data.zero_()

    def train(self):
        total_itr = 0
        best_valid_psnr = -1
        for epoch in range(1, self.configs['total_epochs'] + 1):
            psnr_sum = 0.
            ssim_num = 0.
            min_psnr = 1000
            for i, (input, target) in enumerate(tqdm(self.train_loader, desc=f'training: {epoch}')):
                total_itr += 1
                input = input.to(self.configs['device']).float()
                target = target.to(self.configs['device']).float()
                output = self.gen(input)


                disc_loss = self.train_discriminator(input, output, target, total_itr)

                if total_itr % self.configs['g_update_interval'] == 0:
                    gen_loss = self.train_generator(input, output, target, total_itr)



                    psnr_of_one_batch = piq.psnr(output, target, data_range=1.0, reduction='none')
                    psnr_sum += torch.sum(psnr_of_one_batch).item()
                    min_psnr_of_one_batch = torch.min(psnr_of_one_batch).item()
                    if min_psnr_of_one_batch < min_psnr:
                        min_psnr = min_psnr_of_one_batch

                    temp = piq.ssim(output, target, data_range=1.0, reduction='sum')
                    ssim_num += temp.item()
                    # if temp.item() < 0:
                    #     print('ssim is less than 0')
                    #     print(output)
                else:
                    gen_loss = None
                if total_itr % self.configs['log_itr_interval'] == 0:
                    self.writer.add_scalar('training/disc_loss', disc_loss, total_itr)
                if gen_loss is not None and total_itr % self.configs['gen_log_interval'] == 0:
                    self.writer.add_scalar('training/gen_loss', gen_loss, total_itr)
            if epoch % self.configs['save_epoch_interval'] == 0:
                self.save(f'epoch_{epoch}', name=str(epoch))

            mean_psnr = psnr_sum / len(self.train_loader.dataset)
            mean_ssim = ssim_num / len(self.train_loader.dataset)

            # 验证
            if epoch % self.configs['validation_interval'] == 0:
                self.gen.eval()
                valid_psnr = self.validation_or_test(self.validation_loader, epoch, 'validation')
                if valid_psnr > best_valid_psnr:
                    best_valid_psnr = valid_psnr
                    self.save(f'best_valid_epoch_{epoch}', name='best_valid')
                self.validation_or_test(self.test_loader, epoch, 'test')
                self.gen.train()


            self.writer.add_scalar('training/epoch_mean_psnr', mean_psnr, epoch)
            self.writer.add_scalar('training/epoch_min_psnr', min_psnr, epoch)
            self.writer.add_scalar('training/epoch_mean_ssim', mean_ssim, epoch)
            self.writer.flush()
            print(f'training epoch: {epoch}, mean_psnr: {mean_psnr}, min_psnr: {min_psnr}, mean_ssim: {mean_ssim}')


    @torch.no_grad()
    def validation_or_test(self, loader, epoch, desc):
        psnr_sum = 0.
        min_psnr = 1000
        ssim_num = 0.
        with torch.no_grad():
            for i, (input, target) in enumerate(tqdm(loader, desc=desc+f':{epoch}')):
                input = input.to(self.configs['device']).float()
                target = target.to(self.configs['device']).float()
                output = self.gen(input)
                psnr_of_one_batch = piq.psnr(output, target, data_range=1.0, reduction='none')
                psnr_sum += torch.sum(psnr_of_one_batch).item()
                min_psnr_of_one_batch = torch.min(psnr_of_one_batch)
                ssim_num += piq.ssim(output, target, data_range=1.0, reduction='sum')
                if min_psnr_of_one_batch < min_psnr:
                    min_psnr = min_psnr_of_one_batch
            self.writer.add_scalar(desc + '/epoch_mean_psnr', psnr_sum / len(loader.dataset), epoch)
            self.writer.add_scalar(desc + '/epoch_min_psnr', min_psnr, epoch)
            self.writer.add_scalar(desc + '/epoch_mean_ssim', ssim_num / len(loader.dataset), epoch)
            print(f'{desc} epoch: {epoch}, mean_psnr: {psnr_sum / len(loader.dataset)}, min_psnr: {min_psnr}, mean_ssim: {ssim_num / len(loader.dataset)}')
        return psnr_sum / len(loader.dataset)


    def train_discriminator(self, input, output,target, n_itr):
        self.d_optimizer.zero_grad()
        real_enc, real_dec = self.dis(target)
        fake_enc, fake_dec = self.dis(output.detach())
        source_enc, source_dec = self.dis(input)

        disc_loss = self.discriminative_loss(real_enc, 1.) + self.discriminative_loss(real_dec, 1.) + \
                    self.discriminative_loss(fake_enc, 0.) + self.discriminative_loss(fake_dec, 0.) + \
                    self.discriminative_loss(source_enc, 0.) + self.discriminative_loss(source_dec, 0.)
        total_loss = disc_loss

        # TODO: 加cutmix之类的loss

        total_loss.backward()
        self.d_optimizer.step()
        return total_loss

    def train_generator(self, input, output, target, n_itr):
        self.g_optimizer.zero_grad()
        img_gen_enc, img_gen_dec = self.dis(output)
        img_gen_loss = self.discriminative_loss(img_gen_enc, 1.) + self.discriminative_loss(img_gen_dec, 1.)

        total_loss = 0.

        total_loss += self.img_loss(output, target) + img_gen_loss
        total_loss.backward()
        self.g_optimizer.step()
        return total_loss


    def save(self, desc, name):
        torch.save({
            'gen_state_dict': self.gen.state_dict(),
            'dis_state_dict': self.dis.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'configs': self.configs,
            'desc': desc,

        }, os.path.join(self.configs['save_dir'], name + '.pth'))
        print(f'save model to {os.path.join(self.configs["save_dir"], name + ".pth")}')

