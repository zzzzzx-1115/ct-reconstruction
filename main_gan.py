from trainer_gan import Trainer


def main():
    configs = {
        'generator_name': 'RED_CNN',
        'discriminator_name': 'UNet_disc',
        'generator': {
            'out_channels': 16
        },
        'discriminator': {
            'in_channels': 1,
            'out_chennels': 1,
            'hid_dims': [8, 16, 32, 64, 128],
        },
        'training_data':
            {
                'root': '/home/zhengxin/mnt/projects/ct-reconstruction/processed_data/train',
                # 'patch_size': 4,
            },
        'validation_data':
            {
                'root':'/home/zhengxin/mnt/projects/ct-reconstruction/processed_data/valid',
                # 'patch_size': 4,
            },
        'test_data':
            {
                'root': '/home/zhengxin/mnt/projects/ct-reconstruction/processed_data/test',
                # 'patch_size': 4,
            },
        # 'test_dir': '/home/zhengxin/mnt/projects/ct-reconstruction/processed_data/test',
        'device': 'cuda:4',
        'log_dir': '/home/zhengxin/mnt/projects/ct-temp/logs/gan_test',
        'save_dir': '/home/zhengxin/mnt/projects/ct-temp/ckpts/gan_test',
        'batch_size': 32,
        'checkpoint': None,
        'g_lr': 1e-6,
        'd_lr': 1e-5,
        'total_epochs': 500,
        'log_itr_interval': 2,
        'g_update_interval': 1,
        'gen_log_interval': 2,
        'save_epoch_interval': 2,
        'validation_interval': 2,
        'loss_alpha': 0.7,
        'loss_scale': 50.0,
    }

    trainer = Trainer(configs)
    trainer.train()

main()