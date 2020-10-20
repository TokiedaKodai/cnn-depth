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


list_dirname = ['wave1', 'wave2', 'board']
list_dirname = ['wave1-board', 'wave2-board']
list_dirname = ['boardA']
list_dirname = ['wave1', 'wave2', 'boardA']
list_dirname = ['wave1', 'wave2', 'board', 'boardA']
list_dirname = ['wave1']
# for dirname in list_dirname:
#     for datatype in [0, 3]:
#         run_predict(dirname, 300, datatype)

# # list_dirname = ['wave1_TL', 'wave1_FT', 'wave2_TL', 'wave2_FT']
list_dirname = ['wave1_FT', 'wave2_FT']
# list_dirname = ['wave1_FT_med', 'wave2_FT_med']
# list_dirname = ['wave2-2000_FT_med']
# for dirname in list_dirname:
#     for datatype in [0, 3]:
#         run_predict(dirname, 600, datatype)


# for dirname in ['wave1', 'wave2']:
#     for datatype in [1, 2]:
#         run_predict(dirname, 300, datatype)



# list_dirname = ['wave1', 'wave2', 'board', 'boardA']
# for dirname in list_dirname:
#     for datatype in [2]:
#         run_predict(dirname, 300, datatype)

# list_dirname = ['wave1_FT', 'wave2_FT']
# for dirname in list_dirname:
#     for datatype in [2]:
#         run_predict(dirname, 600, datatype)


list_dirname = ['wave1_noise-shading', 'wave2_noise-shading']
list_dirname = ['wave1_noise-gt', 'wave2_noise-gt']
for dirname in list_dirname:
    # for datatype in [0, 1, 2, 3]:
    # for datatype in [0, 3]:
    for datatype in [3]:
        run_predict(dirname, 300, datatype)