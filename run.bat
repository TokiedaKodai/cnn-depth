REM output_dir, learning type, start, epoch num, aug type, drop rate %, lr
REM timeout /t 100 > nul
REM shutdown -s -t 300


REM python train.py wave1-norm_dif-norm_no-aug 0 0 200 0 10
REM python predict_2.py wave1-norm_dif-norm_no-aug 200 1
REM python predict_2.py wave1-norm_dif-norm_no-aug 200 0
REM python predict_2.py wave1-norm_dif-norm_no-aug 200 2



REM REM REM python train.py wave1-norm_dif-norm_aug 0 0 200 7 10
REM REM python predict_2.py wave1-norm_dif-norm_aug 200 1
REM REM python predict_2.py wave1-norm_dif-norm_aug 200 0
REM REM python predict_2.py wave1-norm_dif-norm_aug 200 2

REM REM python train.py wave1-norm_dif-norm_aug 0 1 400 7 10
REM python predict_2.py wave1-norm_dif-norm_aug 400 1
REM python predict_2.py wave1-norm_dif-norm_aug 400 0
REM python predict_2.py wave1-norm_dif-norm_aug 400 2



REM REM python train.py wave1-norm_dif-norm_no-aug_TL 1 1 400 0 10
REM python predict_2.py wave1-norm_dif-norm_no-aug_TL 400 0
REM python predict_2.py wave1-norm_dif-norm_no-aug_TL 400 2

REM REM python train.py wave1-norm_dif-norm_no-aug_FT 2 1 400 0 10
REM python predict_2.py wave1-norm_dif-norm_no-aug_FT 400 0
REM python predict_2.py wave1-norm_dif-norm_no-aug_FT 400 2

REM python predict_2.py wave1-norm_dif-norm_no-aug_TL 220 0
REM python predict_2.py wave1-norm_dif-norm_no-aug_TL 220 2
REM python predict_2.py wave1-norm_dif-norm_no-aug_FT 220 0
REM python predict_2.py wave1-norm_dif-norm_no-aug_FT 220 2



REM python train.py wave1-norm_dif-norm_aug_TL 1 1 800 7 10
REM python predict_2.py wave1-norm_dif-norm_aug_TL 800 1
REM python predict_2.py wave1-norm_dif-norm_aug_TL 800 0
REM python predict_2.py wave1-norm_dif-norm_aug_TL 800 2

REM python train.py wave1-norm_dif-norm_aug_FT 2 1 800 7 10
REM python predict_2.py wave1-norm_dif-norm_aug_FT 800 1
REM python predict_2.py wave1-norm_dif-norm_aug_FT 800 0
REM python predict_2.py wave1-norm_dif-norm_aug_FT 800 2







REM python train.py wave1-norm-new_dif-norm_no-aug 0 1 200 0 10
REM python predict_2.py wave1-norm-new_dif-norm_no-aug 200 1
REM python predict_2.py wave1-norm-new_dif-norm_no-aug 200 0
python predict_2.py wave1-norm-new_dif-norm_no-aug 200 2


REM python train.py wave1-norm-new_dif-norm_aug-shift 0 0 200 2 10
REM python predict_2.py wave1-norm-new_dif-norm_aug-shift 200 1
REM python predict_2.py wave1-norm-new_dif-norm_aug-shift 200 0
REM python predict_2.py wave1-norm-new_dif-norm_aug-shift 200 2

REM python train.py wave1-norm-new_dif-norm_aug-shift 0 1 400 2 10
REM python predict_2.py wave1-norm-new_dif-norm_aug-shift 400 1
REM python predict_2.py wave1-norm-new_dif-norm_aug-shift 400 0
python predict_2.py wave1-norm-new_dif-norm_aug-shift 400 2