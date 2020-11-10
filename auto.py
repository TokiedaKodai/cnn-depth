import os
import shutil
from subprocess import call

os.chdir(os.path.dirname(os.path.abspath(__file__))) #set current dir

predict_file_name = 'predict.py'

def printexec(cmdstring, paramstring):
    print( cmdstring + ' ' + paramstring)
    call( [cmdstring] + paramstring.strip().split(' ')  )

def run_py(filename, *params):
    paramstring = ''
    for param in params:
        paramstring += str(param) + ' '
    printexec('python', filename + ' ' + paramstring)

def run_predict(dir_name, epoch, param):
    run_py(predict_file_name, dir_name, epoch, param)


list_dirname = ['wave1', 'wave2']
# list_dirname = ['wave1_noise-shading', 'wave2_noise-shading']
# list_dirname = ['wave1_noise-gt', 'wave2_noise-gt']
list_dirname = ['board-gap', 'board-med']
list_dirname = ['wave2-board']
list_dirname = ['wave1', 'wave2', 'board-gap', 'board-med']
list_datatype = [0, 1, 2, 3]
list_datatype = [1, 2]
list_datatype = [0, 3]
# list_datatype = [0]
# list_datatype = [3]
for dirname in list_dirname:
    for datatype in list_datatype:
        run_predict(dirname, 300, datatype)
        # run_predict(dirname, 600, datatype)


list_dirname = ['wave1_FT', 'wave2_FT', 'wave1_FT-median', 'wave2_FT-median']
# list_dirname = ['wave2_FT_final', 'wave1_FT-median_final', 'wave2_FT-median_final']
list_dirname = ['wave1_FT', 'wave2_FT', 'wave1_FT-median', 'wave2_FT-median', 'wave1_FT_final', 'wave2_FT_final', 'wave1_FT-median_final', 'wave2_FT-median_final']
list_datatype = [0, 1, 2, 3]
list_datatype = [1, 2]
list_datatype = [0, 3]
# list_datatype = [0]
# list_datatype = [3]
for dirname in list_dirname:
    for datatype in list_datatype:
        # run_predict(dirname, 300, datatype)
        run_predict(dirname, 600, datatype)