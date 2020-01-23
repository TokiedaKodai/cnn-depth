import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# root_dir = 'C:/Users/b19.tokieda/Desktop/cnn-depth_remote/local-dir/output_archive/200123/'
root_dir = './output_archive/'
data_dir = root_dir + '200123/'
output_dir = 'output_'
pred_dir = '/predict_1000'

def get_index_depth(dirname, index=[], typ_name='test', error='RMSE'):
    df = pd.read_csv(dirname + '/error_compare.txt')
    df = df[df['typ_name']==typ_name]
    if len(index) is not 0:
        df = df.loc[index]
    index = df['index'].astyp_name(str).values
    depth = np.array(df['{} depth'.format(error)])
    mean_depth = np.mean(depth)
    depth = np.append(depth, mean_depth)
    index = np.append(index, 'Avg')
    return index, depth

def get_predict(dirname, index=[], typ_name='test', error='RMSE'):
    df = pd.read_csv(dirname + '/error_compare.txt')
    df = df[df['typ_name']==typ_name]
    if len(index) is not 0:
        df = df.loc[index]
    predict = np.array(df['{} predict'.format(error)])
    mean_predict = np.mean(predict)
    predict = np.append(predict, mean_predict)
    return predict

def get_list_dir(list_compares, data_dir, output_dir, pred_dir):
    list_dir = []
    for dir_name in list_compares:
        list_dir.append(data_dir + output_dir + dir_name + pred_dir)
    return list_dir

def get_list_pred(list_dir):
    list_pred = []
    for directory in list_dir:
        pred = get_predict(directory)
        list_pred.append(pred)
    return list_pred

def gen_graph(label, depth, list_pred, list_compares, comp_name, typ_name='test'):
    list_color = ['blue', 'orange', 'lightgreen', 'lightblue', 'red']
    list_bar = []
    list_legend = ['depth']
    list_legend.extend(list_compares)

    if comp_name is not '':
        comp_name = '_' + comp_name
    
    idx = np.array(range(len(label)))
    width = 0.8 / len(list_legend)

    plt.figure()
    list_bar.append(plt.bar(idx-width, depth, width=width, align='edge', tick_label=label, color=list_color[0]))
    for i, pred in enumerate(list_pred):
        list_bar.append(plt.bar(idx+width*i, pred, width=width, align='edge', tick_label=label, color=list_color[i+1]))
    plt.legend(list_bar, list_legend)
    plt.title('Error Comparison')
    # plt.title('No-fake data learning')
    plt.xlabel(typ_name + ' data')
    # plt.xlabel('Fake test data')
    plt.ylabel('RMSE [m]')
    plt.tick_params(labelsize=7)
    plt.savefig('errs_cmp{}_{}.pdf'.format(comp_name, typ_name))

def compare_error(dir_name, error='RMSE'):
    for typ_name in ['train', 'test']:
        label, depth = get_index_depth(dir_name, typ_name=typ_name)
        pred = get_predict(dir_name typ_name=typ_name)
        gen_graph(label, depth, pred, 'predict', typ_name=typ_name)

def compare_errors(list_compares, comp_name='', data_dir=data_dir, output_dir=output_dir, pred_dir=pred_dir):
    list_dir = get_list_dir(list_compares, data_dir, output_dir, pred_dir)
    label, depth = get_index_depth(list_dir[0])
    list_pred = get_list_pred(list_dir)
    gen_graph(label, depth, list_pred, list_compares, comp_name)


def main():
    list_compares = ['no-aug', 'aug_no-zoom', 'aug']
    compare_errors(list_compares, 'no-fake')


    dir1 = root_dir + '200122/output_aug/predict_1000/'
    dir2 = root_dir + '200123/output_aug/predict_1000/'
    index=[44, 45, 46, 47]
    label, depth = get_index_depth(dir1, index)
    pred1 = get_predict(dir1, index)
    pred2 = get_predict(dir2, index)
    list_pred = [pred1, pred2]
    gen_graph(label, depth, list_pred, ['fake data learn', 'no-fake data learn'], 'FakeLearn')

if __name__ == '__main__':
        main()