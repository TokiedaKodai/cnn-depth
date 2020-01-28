REM python train_difference_learn.py output_no-aug 0 500 0 0 0
REM python predict_difference_learn.py output_no-aug 500 0

REM python train_difference_learn.py output_aug_shift 0 500 1 1 0
REM python predict_difference_learn.py output_aug_shift 500 0

REM python train_difference_learn.py output_aug_rotate 0 500 1 2 0
REM python predict_difference_learn.py output_aug_rotate 500 0

python train_difference_learn.py output_aug_zoom 0 500 1 3 0
python predict_difference_learn.py output_aug_zoom 500 0

REM python train_difference_learn.py output_aug_no-zoom 0 500 1 4 0
REM python predict_difference_learn.py output_aug_no-zoom 500 0

python train_difference_learn.py output_aug 0 500 1 0 0
python predict_difference_learn.py output_aug 500 0



python train_difference_learn.py output_no-aug 1 1000 0 0 0
python predict_difference_learn.py output_no-aug 1000 0

REM python train_difference_learn.py output_aug_shift 1 1000 1 1 0
REM python predict_difference_learn.py output_aug_shift 1000 0

REM python train_difference_learn.py output_aug_rotate 1 1000 1 2 0
REM python predict_difference_learn.py output_aug_rotate 1000 0

python train_difference_learn.py output_aug_zoom 1 1000 1 3 0
python predict_difference_learn.py output_aug_zoom 1000 0

REM python train_difference_learn.py output_aug_no-zoom 1 1000 1 4 0
REM python predict_difference_learn.py output_aug_no-zoom 1000 0

python train_difference_learn.py output_aug 1 1000 1 0 0
python predict_difference_learn.py output_aug 1000 0



REM python train_difference_learn.py output_no-aug 0 1000 0 0 10
REM python predict_difference_learn.py output_no-aug 1000 0

REM python train_difference_learn.py output_aug_shift 0 1000 1 1 10
REM python predict_difference_learn.py output_aug_shift 1000 0

REM python train_difference_learn.py output_aug_rotate 0 1000 1 2 10
REM python predict_difference_learn.py output_aug_rotate 1000 0

REM python train_difference_learn.py output_aug_zoom 0 1000 1 3 10
REM python predict_difference_learn.py output_aug_zoom 1000 0

REM python train_difference_learn.py output_aug 0 1000 1 0 10
REM python predict_difference_learn.py output_aug 1000 0



REM python train_difference_learn.py output_aug_drop=0 0 1000 1 0 0
REM python predict_difference_learn.py output_aug_drop=0 1000 0

REM python train_difference_learn.py output_aug_new_drop=10 0 1000 1 0 10
REM python predict_difference_learn.py output_aug_new_drop=10 1000 0



REM timeout /t 60 > nul
REM shutdown -s -t 300