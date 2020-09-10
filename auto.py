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


for dirname in ['wave1', 'wave2']:
    for datatype in [0, 3]:
        run_predict(dirname, 300, datatype)

for dirname in ['wave1_TL', 'wave1_FT', 'wave2_TL', 'wave2_FT']:
    for datatype in [0, 3]:
        run_predict(dirname, 600, datatype)