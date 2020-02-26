REM output_dir, is model exist, epoch num, is aug, aug type, drop rate %, network type

python train.py resnet_drop-5_no-aug 0 500 0 0 5 1
python predict.py resnet_drop-5_no-aug 500 1

python train.py resnet_drop-5_aug 0 500 1 0 5 1
python predict.py resnet_drop-5_aug 500 1

python train.py resnet_drop-5_aug-no-zoom 0 500 1 4 5 1
python predict.py resnet_drop-5_aug-no-zoom 500 1

REM python train.py resnet_drop-5_aug-zoom 0 500 1 3 5 1
REM python predict.py resnet_drop-5_aug-zoom 500 1

REM python train.py resnet_drop-5_aug-no-rotate 0 500 1 5 5 1
REM python predict.py resnet_drop-5_aug-no-rotate 500 1
