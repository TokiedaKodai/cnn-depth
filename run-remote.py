import os
import shutil
from subprocess import call

local_dir = 'C:/Users/b19.tokieda/Desktop/cnn-depth_remote/local-dir/'
remote_server = 'limu7@10.200.12.115'
remote_dir = '~/tokieda/cnn-depth_remote/remote-dir/'
remote_key = '~/.ssh/limu7_key'

train_file_name = 'train_difference_learn.py'
common_file_name = 'shade_cnn_common.py'
predict_file_name = 'predict_difference_learn.py'

#OutputDir
#RemoteOutput
out_dir = 'output/'
#OutputBackup
out_old = 'output_old/'

# Train on local
is_local_train = True
# is_local_train = False

os.chdir(os.path.dirname(os.path.abspath(__file__))) #set current dir

def printexec(cmdstring, paramstring):
    print( cmdstring + ' ' + paramstring)
    call( [cmdstring] + paramstring.strip().split(' ')  )

def printexec_in_new_terminal(cmdstring, paramstring):
    print( cmdstring + ' ' + paramstring)
    new_cmdstring = 'start'
    new_paramstring = cmdstring + ' ' + paramstring
    call( [new_cmdstring] + new_paramstring.strip().split(' ') , shell=True)

def execlocal(cmd, paramstring):
    printexec('c:/cygwin64/bin/' + cmd, paramstring)

def execremote(cmd, paramstring):
    printexec('c:/cygwin64/bin/ssh', 
                '-i ' + remote_key + ' ' + remote_server + ' ' + cmd + ' ' + paramstring)

def copy_file_to_remote(filename, dir1='./', dir2=remote_dir):
    execlocal('scp', '-i ' + remote_key + ' ' + dir1 + filename + ' ' + remote_server + ':' + dir2)

# def copy_file_from_remote(filename, dir1=remote_dir + out_dir, dir2):
#     printexec('c:/cygwin64/bin/scp', 
#                 '-i ' + remote_key + ' ' + remote_server + ':' + dir1 + filename + ' ' + dir2)
def copy_folder_from_remote(foldername, dir1=remote_dir, dir2='./'):
    execlocal('scp', '-r -i ' + remote_key + ' ' + remote_server + ':' + dir1 + foldername + ' ' + dir2)

def rename(oldname, newname):
    execlocal('mv', oldname + ' ' + newname)

def run_local(filename, *params):
    paramstring = ''
    for param in params:
        paramstring += str(param) + ' '
    printexec('python', filename + ' ' + paramstring)

def run_local_in_new_terminal(filename, *params):
    paramstring = ''
    for param in params:
        paramstring += str(param) + ' '
    printexec_in_new_terminal('python', filename + ' ' + paramstring)

def run_remote(filename, *params):
    paramstring = ''
    for param in params:
        paramstring += str(param) + ' '
    execremote('/usr/bin/python', remote_dir + filename + ' ' + paramstring)

def run_train(is_model_exist, dir_name, param1, param2, param3):
        if is_local_train:
                run_local(train_file_name, dir_name, int(param1), param2, param3)
        else:
                if not is_model_exist:
                        execremote('rm', '-r ' + remote_dir + out_old + out_dir)
                        execremote('mv', remote_dir + out_dir + ' ' + remote_dir + out_old)
                        
                        execlocal('rm', '-r ' + out_old + out_dir)
                        execlocal('mv', out_dir + ' ' + out_old)
                
                run_remote(train_file_name, dir_name, int(param1), param2, param3)

                if not is_model_exist:
                        copy_folder_from_remote(out_dir)
                        rename(out_dir, dir_name) 
                else:
                        copy_folder_from_remote(out_dir)
                        execlocal('cp', '-r ' + out_dir + 'model' + ' ' + dir_name)
                        execlocal('cp', out_dir + 'model-final.hdf5' + ' ' + dir_name)
                        execlocal('cp', out_dir + 'training.log' + ' ' + dir_name)

def run_predict(dir_name, epoch, param):
        if is_local_train:
                run_local(predict_file_name, dir_name, epoch, param)
        else:
                # run_local_in_new_terminal(predict_file_name, dir_name, epoch, param)
                run_local(predict_file_name, dir_name, epoch, param)

def train_and_predict(is_model_exist, out_local, epoch_num, parameter):
        run_train(is_model_exist, out_local, is_model_exist, epoch_num, parameter)
        run_predict(out_local, epoch_num, parameter)

def shutdown(time=60):
        # printexec('c:/cygwin64/bin/ssh', 
        #         '-i ' + remote_key + ' ' + remote_server + ' sudo shutdown -r')
        execlocal('c:/cygwin64/bin/shutdown', '-s -t ' + str(time))

'''
ARGV
1: is model exists (is not train start)
2: epoch num
3: parameter
'''

#LocalOutput
out_local_name = 'output_drop='

#epoch
start_epoch_num = 100
list_epoch = [200, 500, 1000]

#Parameter
scaling = 100
dropout = 0.12

#Parameter List
list_noparam = [0]
list_scaling = [100, 1]
list_dropout = [0, 20, 40]
list_augment = [1, 0]

def main():
        #変更したファイルをリモートにコピー
        if not is_local_train:
                copy_file_to_remote(train_file_name)
                copy_file_to_remote(common_file_name)
                # copy_file_to_remote(predict_file_name)


        #Train & predict
        list_param = list_dropout
        for param in list_param:
                out_local = out_local_name + str(param)

                is_model_exist = False
                train_and_predict(is_model_exist, out_local, start_epoch_num, param)

                is_model_exist = True
                for epoch_num in list_epoch:
                        train_and_predict(is_model_exist, out_local, epoch_num, param)

        # shutdown(60)

if __name__ == '__main__':
        main()