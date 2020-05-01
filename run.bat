REM output_dir, learning type, start, epoch num, aug type, drop rate %

REM python train.py vloss_aug_drop10 0 0 100 7 10
REM python predict.py vloss_aug_drop10 100 0
REM python predict.py vloss_aug_drop10 100 1
REM python predict.py vloss_aug_drop10 100 2

REM python train.py vloss_aug_drop10 0 1 200 7 10
REM python predict.py vloss_aug_drop10 200 0
REM python predict.py vloss_aug_drop10 200 1
REM python predict.py vloss_aug_drop10 200 2

REM python train.py vloss_aug_drop10 0 1 300 7 10
REM python predict.py vloss_aug_drop10 300 0
REM python predict.py vloss_aug_drop10 300 1
REM python predict.py vloss_aug_drop10 300 2

REM python train.py vloss_aug_drop10 0 1 400 7 10
REM python predict.py vloss_aug_drop10 400 0
REM python predict.py vloss_aug_drop10 400 1
REM python predict.py vloss_aug_drop10 400 2

REM python train.py vloss_aug_drop10 0 1 500 7 10
python predict.py vloss_aug_drop10 500 0
python predict.py vloss_aug_drop10 500 1
python predict.py vloss_aug_drop10 500 2



REM python train.py vloss_no-aug_drop10 0 0 100 0 10
REM python predict.py vloss_no-aug_drop10 100 0
REM python predict.py vloss_no-aug_drop10 100 1
REM python predict.py vloss_no-aug_drop10 100 2

REM python train.py vloss_no-aug_drop10 0 1 200 0 10
REM python predict.py vloss_no-aug_drop10 200 0
REM python predict.py vloss_no-aug_drop10 200 1
REM python predict.py vloss_no-aug_drop10 200 2

REM python train.py vloss_no-aug_drop10 0 1 300 0 10
REM python predict.py vloss_no-aug_drop10 300 0
REM python predict.py vloss_no-aug_drop10 300 1
REM python predict.py vloss_no-aug_drop10 300 2

REM python train.py vloss_no-aug_drop10 0 1 400 0 10
REM python predict.py vloss_no-aug_drop10 400 0
REM python predict.py vloss_no-aug_drop10 400 1
REM python predict.py vloss_no-aug_drop10 400 2

REM python train.py vloss_no-aug_drop10 0 1 500 0 10
python predict.py vloss_no-aug_drop10 500 0
python predict.py vloss_no-aug_drop10 500 1
python predict.py vloss_no-aug_drop10 500 2


REM timeout /t 100 > nul
REM shutdown -s -t 300