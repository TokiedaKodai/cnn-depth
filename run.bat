REM output_dir, is model exist, epoch num, is aug, aug type, drop rate %, network type

REM python train.py unet_no-aug 0 500 0 0 0 0
python predict.py unet_no-aug 500 0

REM python train.py unet_aug 0 500 1 0 0 0
python predict.py unet_aug 500 0

REM python train.py unet_aug-no-zoom 0 500 1 4 0 0
python predict.py unet_aug-no-zoom 500 0



REM python train.py resnet_no-aug 0 500 0 0 0 1
python predict.py resnet_no-aug 500 1

REM python train.py resnet_aug 0 500 1 0 0 1
python predict.py resnet_aug 500 1

REM python train.py resnet_aug-no-zoom 0 500 1 4 0 1
python predict.py resnet_aug-no-zoom 500 1



REM python train.py dense-resnet_no-aug 0 500 0 0 0 2
python predict.py dense-resnet_no-aug 500 2

REM python train.py dense-resnet_aug 0 500 1 0 0 2
python predict.py dense-resnet_aug 500 2

REM python train.py dense-resnet_aug-no-zoom 0 500 1 4 0 2
python predict.py dense-resnet_aug-no-zoom 500 2



REM python train.py unet_drop-5 0 500 0 0 5 0
python predict.py unet_drop-5 500 0

REM python train.py unet_drop-10 0 500 0 0 10 0
python predict.py unet_drop-10 500 0


REM python train.py resnet_drop-5 0 500 0 0 5 1
python predict.py resnet_drop-5 500 1

REM python train.py resnet_drop-10 0 500 0 0 10 1
python predict.py resnet_drop-10 500 1


REM python train.py dense-resnet_drop-5 0 500 0 0 5 2
python predict.py dense-resnet_drop-5 500 2

REM python train.py dense-resnet_drop-10 0 500 0 0 10 2
python predict.py dense-resnet_drop-10 500 2



REM python train.py unet_drop-20 0 500 0 0 20 0
python predict.py unet_drop-20 500 0

REM python train.py resnet_drop-20 0 500 0 0 20 1
python predict.py resnet_drop-20 500 1

REM python train.py dense-resnet_drop-20 0 500 0 0 20 2
python predict.py dense-resnet_drop-20 500 2