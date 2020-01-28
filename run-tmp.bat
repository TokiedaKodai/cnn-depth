REM python train.py no-aug 0 500 0 0 0
REM python predict.py no-aug 500 0

REM python train.py aug_shift 0 500 1 1 0
REM python predict.py aug_shift 500 0

REM python train.py aug_rotate 0 500 1 2 0
REM python predict.py aug_rotate 500 0

REM python train.py aug_zoom 0 500 1 3 0
REM python predict.py aug_zoom 500 0

REM python train.py aug_no-zoom 0 500 1 4 0
REM python predict.py aug_no-zoom 500 0

REM python train.py aug 0 500 1 0 0
REM python predict.py aug 500 0


REM python train.py unet_no-drop 0 500 0 0 0
REM python predict.py unet_no-drop 500 0

REM python train.py unet_drop-10 0 500 0 0 1
REM python predict.py unet_drop-10 500 0

REM python train.py unet_drop-20 0 500 0 0 2
REM python predict.py unet_drop-20 500 0

REM python train.py unet_drop-30 0 500 0 0 3
REM python predict.py unet_drop-30 500 0


REM python train.py unet_no-drop 1 1000 0 0 0
REM python predict.py unet_no-drop 1000 0

REM python train.py unet_drop-10 1 1000 0 0 1
REM python predict.py unet_drop-10 1000 0

REM python train.py unet_drop-20 1 1000 0 0 2
REM python predict.py unet_drop-20 1000 0

REM python train.py unet_drop-30 1 1000 0 0 3
REM python predict.py unet_drop-30 1000 0


python train.py dense-resnet_no-drop 0 500 0 0 0
python predict.py dense-resnet_no-drop 500 0

python train.py dense-resnet_drop-10 0 500 0 0 1
python predict.py dense-resnet_drop-10 500 0

python train.py dense-resnet_drop-20 0 500 0 0 2
python predict.py dense-resnet_drop-20 500 0

python train.py dense-resnet_drop-30 0 500 0 0 3
python predict.py dense-resnet_drop-30 500 0


REM timeout /t 60 > nul
REM shutdown -s -t 300