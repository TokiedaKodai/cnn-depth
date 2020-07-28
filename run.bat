REM output_dir, learning type, start, epoch num, aug type, drop rate %, lr
REM timeout /t 100 > nul
REM shutdown -s -t 300


REM python train.py wave1-norm_dif-norm_no-aug 0 0 200 0 10
REM python predict_2.py wave1-norm_dif-norm_no-aug 200 1
REM python predict_2.py wave1-norm_dif-norm_no-aug 200 0
REM python predict_2.py wave1-norm_dif-norm_no-aug 200 2

python train.py wave1-norm_dif-norm_aug 0 0 200 7 10
python predict_2.py wave1-norm_dif-norm_aug 200 1
python predict_2.py wave1-norm_dif-norm_aug 200 0
python predict_2.py wave1-norm_dif-norm_aug 200 2


REM python train.py wave1-norm_dif-norm_no-aug_TL 1 1 250 0 10
REM python predict_2.py wave1-norm_dif-norm_no-aug_TL 250 0
REM python predict_2.py wave1-norm_dif-norm_no-aug_TL 250 2

REM python train.py wave1-norm_dif-norm_no-aug_TL 1 1 400 0 10
REM python predict_2.py wave1-norm_dif-norm_no-aug_TL 400 0
REM python predict_2.py wave1-norm_dif-norm_no-aug_TL 400 2

REM python train.py wave1-norm_dif-norm_no-aug_FT 2 1 250 0 10
REM python predict_2.py wave1-norm_dif-norm_no-aug_FT 250 0
REM python predict_2.py wave1-norm_dif-norm_no-aug_FT 250 2



REM python train.py board_dif-norm_no-aug 2 0 400 0 10
REM python predict_2.py board_dif-norm_no-aug 400 0
REM python predict_2.py board_dif-norm_no-aug 400 2



REM python train.py wave1-norm-direct_dif-norm_no-aug_TL 1 1 250 0 10
REM python predict_2.py wave1-norm-direct_dif-norm_no-aug_TL 250 0
REM python predict_2.py wave1-norm-direct_dif-norm_no-aug_TL 250 2

REM python train.py wave1-norm-direct_dif-norm_no-aug_FT 2 1 250 0 10
REM python predict_2.py wave1-norm-direct_dif-norm_no-aug_FT 250 0
REM python predict_2.py wave1-norm-direct_dif-norm_no-aug_FT 250 2



REM REM python train.py wave1-norm-direct_dif-norm_no-aug_TL 1 1 400 0 10
REM python predict_2.py wave1-norm-direct_dif-norm_no-aug_TL 400 0
REM python predict_2.py wave1-norm-direct_dif-norm_no-aug_TL 400 2

REM REM python train.py wave1-norm-direct_dif-norm_no-aug_FT 2 1 400 0 10
REM python predict_2.py wave1-norm-direct_dif-norm_no-aug_FT 400 0
REM python predict_2.py wave1-norm-direct_dif-norm_no-aug_FT 400 2