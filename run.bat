REM output_dir, learning type, start, epoch num, aug type, drop rate %

python train.py vloss_aug_drop10 0 0 100 7 10
python predict.py vloss_aug_drop10 100 0
python predict.py vloss_aug_drop10 100 1

REM python train.py vloss_aug_drop10 0 1 200 7 10
REM python predict.py vloss_aug_drop10 200 0
REM python predict.py vloss_aug_drop10 200 1

REM python train.py vloss_aug_drop10 0 1 300 7 10
REM python predict.py vloss_aug_drop10 300 0
REM python predict.py vloss_aug_drop10 300 1



python train.py vloss_aug_drop20 0 0 100 7 20
python predict.py vloss_aug_drop20 100 0
python predict.py vloss_aug_drop20 100 1


python train.py vloss_aug_drop30 0 0 100 7 30
python predict.py vloss_aug_drop30 100 0
python predict.py vloss_aug_drop30 100 1


REM python train.py vloss_aug_drop20 0 1 200 7 20
REM python predict.py vloss_aug_drop20 200 0
REM python predict.py vloss_aug_drop20 200 1

REM python train.py vloss_aug_drop20 0 1 300 7 20
REM python predict.py vloss_aug_drop20 300 0
REM python predict.py vloss_aug_drop20 300 1



REM timeout /t 100 > nul
REM shutdown -s -t 300