REM output_dir, learning type, start, epoch num, aug type, drop rate %, lr


@REM python train.py wave1_coord_FT 2 1 600 0 10
@REM python train.py wave2_coord_FT 2 1 600 0 10

@REM python train.py wave1_coord_FT-median 2 1 600 0 10
@REM python train.py wave2_coord_FT-median 2 1 600 0 10


@REM python train.py wave1_coord_FT_final 2 1 600 0 10
@REM python train.py wave2_coord_FT_final 2 1 600 0 10

python train.py wave1_coord_FT-median_final 2 1 600 0 10
python train.py wave2_coord_FT-median_final 2 1 600 0 10