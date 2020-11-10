REM output_dir, learning type, start, epoch num, aug type, drop rate %, lr


@REM python train.py wave1_FT 2 1 600 0 10
@REM python train.py wave2_FT 2 1 600 0 10

@REM python train.py wave1_FT-median 2 1 600 0 10
@REM python train.py wave2_FT-median 2 1 600 0 10


@REM python train.py wave1_FT_final 2 1 600 0 10
@REM python train.py wave2_FT_final 2 1 600 0 10

python train.py wave1_FT-median_final 2 1 600 0 10
python train.py wave2_FT-median_final 2 1 600 0 10