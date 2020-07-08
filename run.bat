REM output_dir, learning type, start, epoch num, aug type, drop rate %, lr
REM timeout /t 100 > nul
REM shutdown -s -t 300


REM python train.py wave1_scale-std 0 0 200 7 10
REM python predict_2.py wave1_scale-std 200 1

REM python train.py wave1_scale-std 0 1 400 7 10
REM python predict_2.py wave1_scale-std 400 1



REM python train.py wave1_dif-norm 0 0 200 7 10
REM python predict_2.py wave1_dif-norm 200 1

REM python train.py wave1_dif-norm 0 1 400 7 10
REM python predict_2.py wave1_dif-norm 400 1


python train.py wave1_dif-norm_lr0001 0 0 200 7 10
python predict_2.py wave1_dif-norm_lr0001 200 1

python train.py wave1_dif-norm_lr0001 0 1 400 7 10
python predict_2.py wave1_dif-norm_lr0001 400 1