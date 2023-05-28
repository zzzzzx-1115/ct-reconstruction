# CT Low Dose Reconstruction

## Environment
All packages needed to run this repo are described as in `environment.yml`

## Script description
`main_gan.py` is the script used to train this model. After specifying some key settings like path to datasets in this script, you can just run
```
python main_gan.py
```
to start training.


`test_f.py` is the script used to test this model. You will see PSNR, SSIM, #params, GFLOPS of this model.


`inference.py` is the script used to generate reconstructed CT full dose images.