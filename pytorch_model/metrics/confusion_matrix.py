import os
import copy
import numpy as np
import pandas as pd
import warnings
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from .plot_functions import *

def plot_confusion_matrix(y_true, y_pred, original_dataset,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                         show_result=True,
                         override_label=[]):
    """
    This function prints and plots the confusion matrix with Matthews correlation coefficient.
    Normalization can be applied by setting `normalize=True`.
    original_dataset must have 'label' and 'meaning' columns, where label is the predicted/true value,
                    meaning is the corresponding string label. Unique values are automatically evaluated and sorted
    override_label: override diagonal element of confusion matrix with label in override_label
    """
    
    # original dataset is used to sort the labels and get the corresponding meaning
    unique_val = original_dataset[['label', 'meaning']].drop_duplicates().sort_values(by=['meaning'])
    labels = unique_val['label'].values.tolist()
    target_names = unique_val['meaning'].values.tolist()
    
    if title is None:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]

    # Compute Matthews correlation coefficient
    warnings.filterwarnings("ignore")
    mcc = matthews_corrcoef(y_true, y_pred)
    warnings.resetwarnings()
    
    # override diagonal element of confusion matrix for fictitious values (y_true has only 1 class)
    if len(override_label) > 0:
        for x in override_label:
            override_ind=np.argwhere([x==y for y in labels])
            conf_mat[(override_ind, override_ind)]=0
    
    # render confusion matrix
    fig = render_conf_mat(conf_mat, mcc, target_names, normalize, title, cmap)        

    if show_result:
        plt.show()
        print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0))
    else:
        plt.close(fig)
    class_report_dict = classification_report(y_true, y_pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0)
    
    fun_options = {'normalize': normalize, 'cmap': cmap, 'target_names': target_names, 'labels': labels}
    
    return fig, conf_mat, mcc, class_report_dict, fun_options

def render_conf_mat(conf_mat, mcc, target_names, normalize=False, title='', cmap=plt.cm.Blues, add_conf_elements=None):
    '''
    Render confusion matrix in plot_confusion_matrix
    - add_conf_elements: additional text annotation for each element of conf_mat. Same shape of conf_mat. E.g. used to add
                        standard deviation when evaluating average confusion matrix for CV folds.
    '''
    fig, ax = plt.subplots()
    im = ax.imshow(conf_mat, interpolation='nearest', cmap=cmap)
#     ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(conf_mat.shape[1]),
           yticks=np.arange(conf_mat.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=target_names, yticklabels=target_names,
           title=title+('\nMatthews Corr Coef: '+str(round(mcc, 3)) if mcc != None else ''),
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = conf_mat.max() / 2.
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            if add_conf_elements is None:
                ax.text(j, i, format(conf_mat[i, j], fmt),
                        ha="center", va="center",
                        color="white" if conf_mat[i, j] > thresh else "black")
            else:
                ax.text(j, i, str(format(conf_mat[i, j], fmt)) + str(add_conf_elements[i, j]),
                    ha="center", va="center",
                    color="white" if conf_mat[i, j] > thresh else "black")
            
    fig.tight_layout()
    
    return fig

def render_conf_mat_boxplot(conf_mat_list, mcc_list, target_names, title=None, cmap=plt.cm.Blues, fig_size=5):

    fig, ax = plt.subplots(len(target_names), len(target_names), figsize=(fig_size, fig_size), sharey=True, sharex=False)
    ax = ax.flatten()

    conf_mat_list_avg = np.mean(np.array(conf_mat_list), axis=0).round().astype(int)
    conf_mat_list_std = np.std(np.array(conf_mat_list), axis=0).round().astype(str)
    mcc_mean = np.mean(np.array(mcc_list)).round(3)
    mcc_std = np.std(np.array(mcc_list)).round(3)
    color_thresh = conf_mat_list_avg.max() / 2.
    tot_class = len(target_names)
    cc=0
    for i in range(tot_class):   # ax.flatten works row by row
        for j in range(tot_class):

            box_data = [mat[i,j] for mat in conf_mat_list]
            
            x_span = max(box_data)-min(box_data)
            xlim = [min(box_data)-0.1*x_span, max(box_data)+0.1*x_span]
            line_color="white" if conf_mat_list_avg[i, j] > color_thresh else "black"
            back_color=cmap(conf_mat_list_avg[i, j] / conf_mat_list_avg.max())
            if min(xlim) != max(xlim):
                ax[cc].set_xlim(xlim)
            ax[cc].set_facecolor(back_color)
            ax[cc].boxplot(box_data, notch=0, vert=0, patch_artist=False, showmeans=True,meanline=True,
                           boxprops=dict(color=line_color),
                           whiskerprops=dict(color=line_color),
                           capprops=dict(color=line_color), # end of whiskers
                           flierprops=dict(marker='X', markerfacecolor='r', markeredgecolor='k', markersize=12),
                           medianprops=dict(color=line_color, linewidth=0),
                           meanprops=dict(color=line_color))
            ax[cc].scatter(box_data, np.repeat(0.85, len(box_data)), color=line_color, marker='x')
            ax[cc].set_xticks([])
            ax[cc].set_yticks([])
            # annotate box_data values and split
            for split, p in enumerate(box_data):
                ax[cc].annotate(str(p), xy=(p, 0.67), ha='center', color=line_color, rotation=45)
                ax[cc].annotate('('+str(split+1)+')', xy=(p, 0.57), ha='center', color=line_color)
            # annotate mean and std
            ax[cc].annotate(str(conf_mat_list_avg[i,j]), xy=(conf_mat_list_avg[i,j], 1.1), ha='center',
                            color=line_color)
            ax[cc].annotate(str(conf_mat_list_avg[i,j])+'±'+conf_mat_list_std[i,j],
                            xy=(np.mean(xlim), 1.3), ha='center', color=line_color, size=fig_size*3)
            if i == (tot_class-1):
                ax[cc].set_xlabel(target_names[j], rotation=45, size = fig_size*2.5)
            if j == 0:
                ax[cc].set_ylabel(target_names[i], rotation=0, ha="right", size = fig_size*2.5)

            cc += 1
    fig.text(0.55, -0.05, 'Predicted label', ha='center', size = fig_size*2.5)
    fig.text(-0.05, 0.55, 'True label', va='center', rotation='vertical', size = fig_size*2.5)
    fig.suptitle(title+'\nMatthews Corr Coef: '+str(mcc_mean)+'±'+str(mcc_std), size = fig_size*3)
    fig.tight_layout(rect=[0, 0.03, 1, 0.9])
    
    return fig

def evaluate_prediction(set_df, binary_thresh=0.5, multi_class=False, multi_class_label=[], multi_class_mode='max'):
    '''
    Returns predicted class 'pred_label' according to probability 'pred_perc'
    
    - set_df: pandas.df with column 'pred' and 'true' ('true' is used only for output dtype)
    - binary_thresh: threshold for binary classification on predicted values
    - multi_class: True if multiclass task
    - multi_class_label: list of labels for argmax in multiclass task
    - multi_class_mode: how to set predicted class given the probabilities array. 'max': takes highest probability
    '''
    
    get_index = set_df.index
    set_true = set_df.true.values
    set_pred = set_df.pred.values
    if multi_class:
        if multi_class_mode == 'max':
            pred_lab = np.array([multi_class_label[np.argmax(x)] for x in set_pred])
        pred_perc = [int(set_pred[i][int(pred_lab[i])]*100) for i in range(len(pred_lab))]
    else:
        pred_lab = (set_pred > binary_thresh)
        pred_perc = (set_pred * 100).astype(float).round().astype(int)

    pred_lab = pred_lab.astype(set_true.dtype)
    out_df = pd.DataFrame({'pred_label': pred_lab, 'pred_perc': pred_perc}).set_index(get_index)
        
    return out_df

def evaluate_confusion_matrix(original_dataset, train_out, binary_thresh=0.5, save_path='', save_ID='',
                              show_result=True, add_title='', subset_iloc=None, normalize=None, cmap=plt.cm.Blues,
                              multi_class=False, multi_class_mode='max'):
    '''
    Args:
        - original_dataset: dataset used to train the model, needed for plot_confusion_matrix and subsetting
        - train_out: output of trained model, used to take true and predicted values for both train and validation
        - binary_thresh: threshold for binary classification on predicted values
        - add_title: additional title used as super title
        - subset_iloc: list of index to be used for subsetting
        - normalize: normalize confusion matrix entries
        - cmap: colormap to be used
        - multi_class: True if multiclass task
        - multi_class_mode: how to set predicted class given the probabilities array. 'max': takes highest probability
    '''
    
    train = train_out['train_preds']
    valid = train_out['valid_preds']
    original_combination = original_dataset[['label', 'meaning']].drop_duplicates().sort_values(by=['meaning'])
#     original_combination.meaning='-'
    avail_label=original_combination.label.values.astype(train.true.values.dtype)

    if subset_iloc is not None:
        original_dataset=original_dataset.iloc[subset_iloc]
        subset_ind = original_dataset.index
        train=train.loc[train.index.intersection(subset_ind)]
        valid=valid.loc[valid.index.intersection(subset_ind)]

    train_true = train.true.values
    valid_true = valid.true.values
    train_eval_pred = evaluate_prediction(train, binary_thresh, multi_class, multi_class_label=avail_label, multi_class_mode=multi_class_mode)
    train_pred = train_eval_pred['pred_label']
    valid_eval_pred = evaluate_prediction(valid, binary_thresh, multi_class, multi_class_label=avail_label, multi_class_mode=multi_class_mode)
    valid_pred = valid_eval_pred['pred_label']
        
    # if true contains only one class, add a fictitious element with missing class.
    # It will be equal also for pred and have 'meaning' = '-'
    missing_label_train=[]
    missing_label_valid=[]
    if len(np.unique(train_true)) == 1:
        missing_label_train = avail_label[avail_label != np.unique(train_true)]
        train_true = np.append(train_true, missing_label_train)
        train_pred = np.append(train_pred, missing_label_train)
        fict_row = copy.deepcopy(original_dataset.iloc[0:len(missing_label_train)]).drop(columns='meaning')
        fict_row['label']=missing_label_train
        fict_row = pd.merge(fict_row, original_combination, how='inner', on=['label'])
        original_dataset = original_dataset.append(fict_row)
    if len(np.unique(valid_true)) == 1:
        missing_label_valid = avail_label[avail_label != np.unique(valid_true)]
        valid_true = np.append(valid_true, missing_label_valid)
        valid_pred = np.append(valid_pred, missing_label_valid)
        fict_row = copy.deepcopy(original_dataset.iloc[0:len(missing_label_valid)]).drop(columns='meaning')
        fict_row['label']=missing_label_valid
        fict_row = pd.merge(fict_row, original_combination, how='inner', on=['label'])
        original_dataset = original_dataset.append(fict_row)

    train_fig, train_conf_mat, train_mcc,  train_class_report, train_opt = plot_confusion_matrix(train_true, train_pred,
                                                                                                 original_dataset=original_dataset,
                                                                                                 title='Train', show_result=False,
                                                                                                 override_label=missing_label_train,
                                                                                                 normalize=normalize, cmap=cmap)
    valid_fig, valid_conf_mat, valid_mcc, valid_class_report, valid_opt = plot_confusion_matrix(valid_true, valid_pred,
                                                                                                 original_dataset=original_dataset,
                                                                                                 title='Validation', show_result=False,
                                                                                                 override_label=missing_label_valid,
                                                                                                 normalize=normalize, cmap=cmap)
    merge_class_report = pd.concat([class_report_to_df(train_class_report, top_column_index='Train'),
                                       class_report_to_df(valid_class_report, top_column_index='Validation')], axis=1)
    
    # save a merged plot of both confusion matrices
    if save_path != '':
        train_path=os.path.join('results', save_path, save_ID+'_conf_mat_train.png')
        valid_path=os.path.join('results', save_path, save_ID+'_conf_mat_valid.png')
        merge_path=os.path.join('results', save_path, save_ID+'_conf_mat.png')
        train_fig.savefig(train_path)
        valid_fig.savefig(valid_path)
        get_concat_h([Image.open(train_path), Image.open(valid_path)], add_main_title=add_title, font_size=20).save(merge_path)
        os.remove(train_path)
        os.remove(valid_path)
        
    if show_result:
        display(Image.open(merge_path))
        print('\n')
        display(merge_class_report)
    
    out = {'train_conf_mat': train_conf_mat,
           'train_mcc': train_mcc,
           'train_class_report': train_class_report,          
           'valid_conf_mat': valid_conf_mat,
           'valid_mcc': valid_mcc,
           'valid_class_report': valid_class_report,
           'merge_class_report': merge_class_report,
           'options': {'train_opt': train_opt,
                       'valid_opt': valid_opt}}
    
    return out

def class_report_to_df(class_report_dict, top_column_index=None):
    '''
    class_report_dict: classification_report(output_dict=True)
    top_column_index: additional label for top column name (usually is 'Train'/'Validation'/'Test')
    '''
    class_report_df=pd.DataFrame(class_report_dict).T.round(2)
    class_report_df.support=class_report_df.support.astype(int)
    class_report_df.loc['accuracy','f1-score']=round(class_report_dict['accuracy'], 2)
    class_report_df.loc['accuracy','support']=class_report_dict['macro avg']['support']
    class_report_df.loc['accuracy',['precision', 'recall']]=''
    if top_column_index is not None:
        class_report_df.columns= [[top_column_index, '', '', ''],class_report_df.columns]

    return class_report_df

