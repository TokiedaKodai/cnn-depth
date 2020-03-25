REM output_dir, is model exist, epoch num, aug type, drop rate %, network type

REM python train.py resnet_drop-5_no-aug 0 500 0 5 1
REM python predict.py resnet_drop-5_no-aug 500 1

REM python train.py resnet_drop-5_aug 0 500 1 5 1
REM python predict.py resnet_drop-5_aug 500 1

REM python train.py resnet_drop-5_aug-no-scale 0 500 7 5 1
REM python predict.py resnet_drop-5_aug-no-scale 500 1



REM python train.py resnet_drop-5_no-aug 1 1000 0 5 1
REM python predict.py resnet_drop-5_no-aug 1000 1

REM python train.py resnet_drop-5_aug 1 1000 1 5 1
REM python predict.py resnet_drop-5_aug 1000 1

REM python train.py resnet_drop-5_aug-no-scale 1 1000 7 5 1
REM python predict.py resnet_drop-5_aug-no-scale 1000 1



python train.py resnet_drop-5_no-aug 1 1500 0 5 1
python predict.py resnet_drop-5_no-aug 1500 1