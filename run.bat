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


python train.py no-drop 1 500 0 0 0
python predict.py no-drop 500 0

python train.py drop-10 0 500 0 0 1
python predict.py drop-10 500 0

REM python train.py no-drop 0 1000 0 0 0
REM python predict.py no-drop 1000 0

REM python train.py drop-10 0 1000 0 0 1
REM python predict.py drop-10 1000 0



REM timeout /t 60 > nul
REM shutdown -s -t 300