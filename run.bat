REM output_dir, learning type, start, epoch num, aug type, drop rate %, lr
REM timeout /t 100 > nul
REM shutdown -s -t 300

REM python train.py vaug_half_x4_one_reduce 0 0 200 7 10 2
REM python predict.py vaug_half_x4_one_reduce 200 1

REM python train.py vaug_half_x4_lr0001_one_3 0 0 200 7 10 3
REM python predict.py vaug_half_x4_lr0001_one_3 200 1

REM python train.py vaug_half_x4_lr002_one 0 0 200 7 10 2
REM python predict.py vaug_half_x4_lr002_one 200 1

REM python train.py vaug_half_x4_lr005_one 0 0 200 7 10 3
REM python predict.py vaug_half_x4_lr005_one 200 1




REM python train.py vaug_half_x4_dif-no-norm 0 0 50 7 10
REM python predict.py vaug_half_x4_dif-no-norm 50 1

REM python train.py vaug_half_x4_dif-no-norm_lr0001 0 0 50 7 10
REM python predict.py vaug_half_x4_dif-no-norm_lr0001 50 1


REM python train.py vaug_half_x4_dif-norm 0 0 50 7 10
REM python predict.py vaug_half_x4_dif-norm 50 1

REM python train.py vaug_half_x4_dif-norm 0 1 150 7 10
REM python predict.py vaug_half_x4_dif-norm 150 1

REM python train.py vaug_half_x4_dif-norm 0 1 200 7 10
REM python predict.py vaug_half_x4_dif-norm 200 1


REM python train.py vaug_half_x4_dif-norm_lr01 0 0 50 7 10
REM python predict.py vaug_half_x4_dif-norm_lr01 50 1



REM python train.py vaug_half_x4_scale-std 0 0 50 7 10
REM python predict_2.py vaug_half_x4_scale-std 50 1

python train.py vaug_half_x4_scale-std 0 1 150 7 10
python predict_2.py vaug_half_x4_scale-std 150 1

python train.py vaug_half_x4_scale-std 0 1 200 7 10
python predict_2.py vaug_half_x4_scale-std 200 1