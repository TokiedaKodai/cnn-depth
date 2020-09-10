REM output_dir, learning type, start, epoch num, aug type, drop rate %, lr

python predict.py wave2_2000 400 0
python predict.py wave2_2000 400 3

REM python train.py wave2_2000_TL 1 1 600 0 10
python predict.py wave2_2000_TL 600 0
python predict.py wave2_2000_TL 600 3

REM python train.py wave2_2000_FT 2 1 600 0 10
python predict.py wave2_2000_FT 600 0
python predict.py wave2_2000_FT 600 3

REM python train.py wave2_2000_TL-enc 3 1 600 0 10
python predict.py wave2_2000_TL-enc 600 0
python predict.py wave2_2000_TL-enc 600 3



python predict.py wave2_2000 400 0
python predict.py wave2_2000 400 3

REM python train.py wave2_2000_TL 1 1 600 0 10
python predict.py wave2_2000_TL 600 0
python predict.py wave2_2000_TL 600 3

REM python train.py wave2_2000_FT 2 1 600 0 10
python predict.py wave2_2000_FT 600 0
python predict.py wave2_2000_FT 600 3