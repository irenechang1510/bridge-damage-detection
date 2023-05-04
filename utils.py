import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# import os

def get_instance(file_name, label):
    '''
    Purpose: Read in the dataset + Carry out SVD on the full dataset to obtain the damage 
                indicators (first vector of the left signular matrix)
    '''
    dataset = pd.read_csv(file_name, header=None, sep=' ', prefix='node_')
    return svd(dataset, label)

def svd(df, label):
    L, _, _ = np.linalg.svd(df.T)
    phi_1 = L[:, 0]
    res = np.append(phi_1, [label])
    return pd.DataFrame(res.reshape(1, -1), columns = ['d{}'.format(i) for i in range(1, 10)] + ['class'])

def make_file_list(beam, damage, loc):
    '''
    Purpose: Get all the necessary filenames of a beam that share the same damage level and location
    '''
    load_history = range(1, 201)

    file_list = []
    if damage != 0:
        print("Getting the EI {}0 list...".format(damage))
        for j in load_history:
            file_list.append('Data/Beam_{}_M{}/Acceleration/D{}/D{}.Accel_EI{}0_M{}_LH{}.txt'.format(beam, loc, damage, damage, damage, loc, j))
        
    else:
        print("Getting the healthy list...")
        for j in load_history:
            file_list.append('Data/Beam_{}_M{}/Acceleration/UN/UN.Accel_EI100_M{}_LH{}.txt'.format(beam, loc, loc, j))

    return file_list

def make_class_dataset(class_label, beam, loc):
    '''
    Purpose: make a dataframe for a given beam, with a given damage level, at a given damage location
    Parameters:
        class_label: "0" denotes the healthy class, "1" denotes EI=10% beams, etc
        loc: the location of the damage
    '''
    file_list = make_file_list(beam = beam, damage = class_label, loc = loc)
    
    res_table = pd.DataFrame([], columns = ['d{}'.format(i) for i in range(1, 10)] + ['class'])
    for file in file_list:
        if class_label == 0:
            sample = get_instance(file, 0)
        else:
            sample = get_instance(file, 1)
        res_table = res_table.append(sample, ignore_index=True)
    return res_table

def load_data(source_beam, target_beam, damage_levels, loc):
    '''
    Parameters:
        damage levels: a list of all the damage levels in consideration, should match the number of classes
    '''
    # create the dataset for the source domain
    source_dfs = []
    for d in damage_levels:
        S = make_class_dataset(d, source_beam, loc)
        assert len(S) == 200
        source_dfs.append(S)

    # create the dataset for the target domain
    target_dfs = []
    for d in damage_levels:
        T_full = make_class_dataset(d, target_beam, loc)
        assert len(T_full) == 200
        target_dfs.append(T_full)
    
    Source = pd.concat(source_dfs, ignore_index=True)
    Target = pd.concat(target_dfs, ignore_index=True)

    return Source, Target

def plot_analysis(data, pred, dim):
    '''
    Purpose: plot the false/true pos/neg labels from the model on 2d spaces (one selected dim versus the other 8 dims in
    the dataset)
    Parameters: 
        data: the post-SVD dataset (contains 9 features)
        pred: vector of the predicted labels
        dim: the selected dim that will be plotted against the other 8 dims in the dataset
    '''
    idx_fp = data['class'][(pred == 1) & (data['class'] == 0)].index
    idx_fn =  data['class'][(pred == 0) & (data['class'] == 1)].index
    idx_tp = data['class'][(pred == 1) & (data['class'] == 1)].index
    idx_tn = data['class'][(pred == 0) & (data['class'] == 0)].index
    target_fp = data.loc[idx_fp, :]
    target_fn = data.loc[idx_fn, :]
    target_tp = data.loc[idx_tp, :]
    target_tn = data.loc[idx_tn, :]
    fig, ax = plt.subplots(2, 4, figsize=(18, 8))
    count = 1
    for i in range(2):
        for j in range(4):
            if dim == "d"+str(count):
                count += 1
            sns.scatterplot(data = target_tn, x=dim, y="d"+str(count), color='green', alpha=0.5, ax=ax[i][j])
            sns.scatterplot(data = target_tp, x=dim, y="d"+str(count), color='yellow', alpha=0.5,ax=ax[i][j])
            sns.scatterplot(data = target_fp, x=dim, y="d"+str(count), color='red', alpha=0.5, ax=ax[i][j])
            sns.scatterplot(data = target_fn, x=dim, y="d"+str(count), color='blue', alpha=0.5, ax=ax[i][j])
            
            ax[i][j].set_xlim(data[dim].min() - 0.005, data[dim].max() + 0.005)
            ax[i][j].set_ylim(data["d"+str(count)].min()- 0.005, data["d"+str(count)].max() + 0.005)
            count += 1
    plt.legend(loc='upper left', labels=['TN', 'TP', 'FP', 'FN'])
    plt.tight_layout()
    plt.show()


def bootstrapping(mod, Xtest, ytest):
    list_recall0 = []
    list_recall1 = []
    list_acc = []
    for _ in range(20):
        sample_X = Xtest.sample(n=500, replace=True)
        sample_y = ytest[sample_X.index]
        pred, sc = mod.predict(sample_X, sample_y)
        list_acc.append(sc)

        tn, fp, fn, tp = confusion_matrix(sample_y, pred).ravel()
        # print("True pos: ", tp, "False pos: ", fp, "True neg: ", tn, "False neg: ", fn)

        if fn == tn == 0:
            list_recall0.append(0)
        else:
            list_recall0.append(tn/(tn + fn))

        if tp == fp == 0:
            list_recall1.append(0)
        else:
            list_recall1.append(tp/(tp + fp))
    return np.mean(list_acc), np.mean(list_recall0), np.mean(list_recall1)
