REM output_dir, is model exist, epoch num, is aug, aug type, drop rate %, network type

python train.py unet_no-aug 0 200 0 0 0 0
python predict.py unet_no-aug 200 0

python train.py unet_aug 0 200 1 0 0 0
python predict.py unet_aug 200 0

python train.py unet_aug-no-zoom 0 200 1 4 0 0
python predict.py unet_aug-no-zoom 200 0



python train.py resnet_no-aug 0 200 0 0 0 1
python predict.py resnet_no-aug 200 1

python train.py resnet_aug 0 200 1 0 0 1
python predict.py resnet_aug 200 1

python train.py resnet_aug-no-zoom 0 200 1 4 0 1
python predict.py resnet_aug-no-zoom 200 1



python train.py dense-resnet_no-aug 0 200 0 0 0 2
python predict.py dense-resnet_no-aug 200 2

python train.py dense-resnet_aug 0 200 1 0 0 2
python predict.py dense-resnet_aug 200 2

python train.py dense-resnet_aug-no-zoom 0 200 1 4 0 2
python predict.py dense-resnet_aug-no-zoom 200 2
