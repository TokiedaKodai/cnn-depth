REM output_dir, learning type, start, epoch num, aug type, drop rate %


python train.py fine_tloss_aug_drop10 2 0 100 7 10
python predict.py fine_tloss_aug_drop10 100

python train.py fine_tloss_aug_drop10 2 1 500 7 10
python predict.py fine_tloss_aug_drop10 500


REM timeout /t 100 > nul
REM shutdown -s -t 300