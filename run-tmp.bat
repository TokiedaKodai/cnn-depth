REM output_dir, learning type, start, epoch num, aug type, drop rate %, lr
REM timeout /t 100 > nul
REM shutdown -s -t 300


REM python train.py wave1-norm_dif-norm_no-aug 0 0 200 0 10
REM python predict_2.py wave1-norm_dif-norm_no-aug 200 1
REM python predict_2.py wave1-norm_dif-norm_no-aug 200 0
REM python predict_2.py wave1-norm_dif-norm_no-aug 200 2


REM python train.py wave1-norm_400_dif-norm_no-aug 0 0 100 0 10
REM python predict_2.py wave1-norm_400_dif-norm_no-aug 100 1
REM python predict_2.py wave1-norm_400_dif-norm_no-aug 100 0
REM python predict_2.py wave1-norm_400_dif-norm_no-aug 100 2


REM python train.py wave1-norm_dif-norm_aug 0 0 200 7 10
REM python predict_2.py wave1-norm_dif-norm_aug 200 1
REM python predict_2.py wave1-norm_dif-norm_aug 200 0
REM python predict_2.py wave1-norm_dif-norm_aug 200 2


REM python train.py wave1-norm-2_dif-norm_no-aug 0 0 200 0 10
REM python predict_2.py wave1-norm-2_dif-norm_no-aug 200 1
REM python predict_2.py wave1-norm-2_dif-norm_no-aug 200 0
REM python predict_2.py wave1-norm-2_dif-norm_no-aug 200 2

REM python train.py wave1-norm-direct_dif-norm_shading-no-norm_no-aug 0 0 200 0 10
REM python predict_2.py wave1-norm-direct_dif-norm_shading-no-norm_no-aug 200 1
REM python predict_2.py wave1-norm-direct_dif-norm_shading-no-norm_no-aug 200 0
REM python predict_2.py wave1-norm-direct_dif-norm_shading-no-norm_no-aug 200 2

REM python train.py wave2-norm-direct_dif-norm_shading-no-norm_no-aug 0 0 200 0 10
REM python predict_2.py wave2-norm-direct_dif-norm_shading-no-norm_no-aug 200 1
REM python predict_2.py wave2-norm-direct_dif-norm_shading-no-norm_no-aug 200 0
REM python predict_2.py wave2-norm-direct_dif-norm_shading-no-norm_no-aug 200 2

REM python train.py wave2-norm_dif-norm_no-aug 0 0 200 0 10
python predict_2.py wave2-norm_dif-norm_no-aug 200 1
python predict_2.py wave2-norm_dif-norm_no-aug 200 0
python predict_2.py wave2-norm_dif-norm_no-aug 200 2