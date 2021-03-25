import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.ticker as mtick
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import math
import matplotlib.cm as cm
from PIL import Image
import cv2
import random
import numpy as np
import pandas as pd
import os
import warnings

def show_sample(df, save_name, image_size, sample_size_for_label_source, ncol):
    '''
    df: must contain:
        - image: path to image
        - label: label of image e.g. 1, 0
        - meaning: meaning of each label
        - source: dataset source
    save_name: name to save figure    
    image_size: size of each sample in pixels
    sample_size_for_label_source: different subplots will be gerated according to all combination of label+source
    ncol: number of images per row
    '''
    
    df['title'] = df.source + ' - ' + df.meaning
    for un in df.label.unique():
        list_to_sample = df[df.label == un].index.tolist()
        rand_ind = random.sample(list_to_sample, min(sample_size_for_label_source, len(list_to_sample)))
        sample_df = df.loc[rand_ind].reset_index()
        main_title = sample_df.title.unique()
        if len(main_title) > 1:
            print('\n #### warning: multiple unique values in main_title:', main_title)

        nrow = math.ceil(sample_df.shape[0] / ncol)
        fig, ax = plt.subplots(nrow, ncol, figsize=(15, 3*nrow), sharey=True, sharex=True)
        ax = ax.flatten()
        for i, row in sample_df.iterrows():
            im = row.image
#             im_name = os.path.basename(im)
#             n=15
#             im_name='\n'.join([im_name[i:i+n] for i in range(0, len(im_name), n)])
            im_ref = np.where(df.image == im)[0][0]
            ax[i].imshow(cv2.resize(np.array(Image.open(im)), (image_size, image_size), interpolation=cv2.INTER_AREA), cmap=cm.gray, vmin=0, vmax=255)
#             ax[i].set_title('df iloc: '+str(im_ref)+'\n'+im_name)
            ax[i].set_title('df iloc: '+str(im_ref))
        fig.suptitle(main_title[0] + ' ' + str(sample_df.shape[0]) + ' out of ' + str(len(list_to_sample)))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig('./results/'+save_name+'_'+sample_df.meaning[0]+'_sample_images.png')
        plt.show()

def plot_performance(df, meas_list, super_title, set_type='split', show_fig=True):
    '''
    Args:
        - df: train_history
        - meas_list: measure(s) we want to plot.
                   It automatically takes all columns ending with meas and discriminate between "Valid" and "Train"
        - set_type: 'split' for Cross-Validation, 'full' for full dataset
    '''

    df.columns = map(str.lower, df.columns)
    main_title = 'Full Dataset' if set_type == 'full' else ('Cross-Validation on ' + str(df.split.max()) + ' splits')
    main_title = super_title + '\n' + main_title
    fig, ax = plt.subplots(len(meas_list), 1, figsize=(6, 3 * len(meas_list)), sharey=False, sharex=False)
    ax = ax.flatten()

    for i, meas in enumerate(meas_list):
        meas = meas.lower()
        if set_type=='split':
            meas_to_plot = df[[col for col in df.columns if meas in col] + ['split', 'epoc']] \
                .groupby('epoc', as_index=False) \
                .agg(
                {
                    'valid_'+meas: ['min', 'max', 'mean'],
                    'train_'+meas: ['min', 'max', 'mean']
                }) \
                .sort_values(by=['epoc'])
            train_avg = meas_to_plot[('train_'+meas, 'mean')].values
            train_min = meas_to_plot[('train_'+meas, 'min')].values
            train_max = meas_to_plot[('train_'+meas, 'max')].values
            valid_avg = meas_to_plot[('valid_'+meas, 'mean')].values
            valid_min = meas_to_plot[('valid_'+meas, 'min')].values
            valid_max = meas_to_plot[('valid_'+meas, 'max')].values
            ax[i].plot(train_avg, linewidth=2, color='b', label='Train')
            ax[i].fill_between(range(len(train_avg)), train_min, train_max, color='b', alpha=.1)
            ax[i].plot(valid_avg, linewidth=2, color='y', label='Validation')
            ax[i].fill_between(range(len(train_avg)), valid_min, valid_max, color='y', alpha=.1)
            ax[i].set_title(meas.upper(), size = 17)
        elif set_type=='full':
            train_avg=df.sort_values(by=['epoc'])['train_'+meas].values
            valid_avg=df.sort_values(by=['epoc'])['valid_'+meas].values
            ax[i].plot(train_avg, linewidth=2, color='b', label='Train')
            ax[i].plot(valid_avg, linewidth=2, color='y', label='Validation')
            ax[i].set_title(meas.upper(), size = 17)
            
        ax[i].set_xlabel('epochs', size = 12)
        ax[i].grid(which='major', axis='y')
        ax[i].set_xticks(ticks=range(1, len(train_avg)+1))

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    if set_type=='split':
        legend_elements = [Line2D([0], [0], color='b', lw=2, label='Train avg'),
                           Patch(facecolor='b', edgecolor='b', alpha=.1, label='Train min-max'),
                           Line2D([0], [0], color='y', lw=2, label='Validation avg'),
                       Patch(facecolor='y', edgecolor='y', alpha=.1, label='Validation min-max')]
    elif set_type=='full':
        legend_elements = [Line2D([0], [0], color='b', lw=2, label='Train'),
                       Line2D([0], [0], color='y', lw=2, label='Validation')]
    
    
    fig.legend(handles=legend_elements, bbox_to_anchor=(1.2, 0.5), loc='center', fontsize = 14)
    # fig.legend(lines, labels, bbox_to_anchor=(1.07, 0.5), loc='center')
    fig.suptitle(main_title, fontsize=20)
    fig.tight_layout(rect=[0, 0.03, 1, 0.9])
    if show_fig:
        fig.show()
    else:
        plt.close(fig)
    
    return fig

def plot_prediction_distribution(set_dict, original_dataset, main_title_prefix = '', session_ID = '', save_path = ''):
    '''
    Plot distribution of predicted percentage for both train and validation
    
    - set_dict: dictionary with 'train_preds' and 'valid_preds' keys, which are pandas.df with
        'pred', 'true', 'pred_class', 'pred_perc' usually output of evaluate_prediction()
    - original_dataset: dataset used to train the model, needed for available label and meaning
    '''

    train_pred = set_dict['train_preds']
    valid_pred = set_dict['valid_preds']
    label_meaning = original_dataset[['label', 'meaning']].drop_duplicates().sort_values(by=['meaning'])

    warnings.filterwarnings("ignore")
    for set_lab, set_preds in zip(['Train', 'Validation'], [train_pred, valid_pred]):

        set_preds = pd.merge(set_preds, label_meaning.rename(columns={'label': 'true', 'meaning': 'true_meaning'}),
                             left_on='true', right_on='true')
        set_preds = pd.merge(set_preds, label_meaning.rename(columns={'label': 'pred_class', 'meaning': 'pred_meaning'}),
                             left_on='pred_class', right_on='pred_class')
        set_preds['correct_class'] = np.where(set_preds.true_meaning == set_preds.pred_meaning, 'Correct', 'Wrong')
        avail_true_meaning = set_preds.true_meaning.unique()

        ncol = len(avail_true_meaning)
        fig, ax = plt.subplots(1, ncol, figsize=(4*ncol, 5), sharey=True, sharex=False)
        ax = ax.flatten()
        newPal   = {'Correct': 'blue', 'Wrong': 'red'}
        for i, true_meaning in enumerate(avail_true_meaning):
            
            true_label = ' ['+str(int(label_meaning[label_meaning.meaning == true_meaning].label.values))+']' if len(label_meaning) == 2 else ''
            meaning_df = set_preds[set_preds.true_meaning == true_meaning]

            sns.swarmplot(x="pred_meaning", y="pred_perc", hue="correct_class", data=meaning_df, palette=newPal, ax = ax[i])
            ax[i].set_xlabel('Predicted Class', size = 13)
            ax[i].tick_params(axis = 'both', which = 'major', labelsize = 12)
            ax[i].set_ylabel('Predicted Percentage' if i==0 else '', size = 13) 
            ax[i].set_title('True Class: ' + true_meaning + true_label, size = 15)
            ax[i].yaxis.set_major_formatter(mtick.PercentFormatter(100))
            ax[i].get_legend().remove()

        legend_elements = [plt.plot([],[], marker="o", ms=10, ls="", color='blue', lw=2, label='Correct')[0],
                           plt.plot([],[], marker="o", ms=10, ls="", color='red', lw=2, label='Wrong')[0]]


        lg = fig.legend(handles=legend_elements, bbox_to_anchor=(1.1, 0.5), loc='center', fontsize = 12, title='Prediction')
        title = lg.get_title()
        title.set_fontsize(15)
        fig.suptitle(set_lab, fontsize=18)
        fig.tight_layout(rect=[0, 0.03, 1, 0.9])
        plt.close(fig)
        fig.savefig(os.path.join(save_path, set_lab+'_prediction_distr.png'), bbox_inches='tight')
    warnings.resetwarnings()
    
    final_path = os.path.join(save_path, session_ID+'_Prediction_distribution.png')
    get_concat_v([Image.open(os.path.join(save_path, x+'_prediction_distr.png')) for x in ['Train', 'Validation']],
                 offset=20, font_size=20, add_main_title=main_title_prefix).save(final_path)
    _=[os.remove(os.path.join(save_path, x+'_prediction_distr.png')) for x in ['Train', 'Validation']]
    
    return final_path

def get_concat_v(img_list, offset=0, add_main_title=None, font_size=20, initial_v_offset=0):
    '''
    images in img_list must have same width
    e.g. get_concat_v([Image.open(im1), Image.open(im2)])
    '''
    
    title_space = 0 if add_main_title is None else font_size+30
    height_list=[x.height for x in img_list]
    dst = Image.new('RGB', (img_list[0].width, sum(height_list) + offset*(len(height_list)-1)+title_space+initial_v_offset), "WHITE")
    
    if add_main_title is not None:
        title_img = centered_text(dst.width, title_space, add_main_title, font_size)
        dst.paste(title_img, (0, initial_v_offset))
        
    dst.paste(img_list[0], (0, initial_v_offset+title_space))
    for i in range(1, len(img_list)):
        dst.paste(img_list[i], (0, initial_v_offset+title_space + sum(height_list[0:i]) + offset))

    return dst

def get_concat_h(img_list, offset=0, add_main_title=None, font_size=20, initial_v_offset=0):
    '''
    images in img_list must have same height
    e.g. get_concat_h([Image.open(im1), Image.open(im2)])
    '''
    
    title_space = 0 if add_main_title is None else font_size+30
    width_list=[x.width for x in img_list]
    dst = Image.new('RGB', (sum(width_list) + offset*(len(width_list)-1), img_list[0].height+title_space+initial_v_offset), "WHITE")
    
    if add_main_title is not None:
        title_img = centered_text(dst.width, title_space, add_main_title, font_size)
        dst.paste(title_img, (0, 0))
        
    dst.paste(img_list[0], (0, initial_v_offset+title_space))
    for i in range(1, len(img_list)):
        dst.paste(img_list[i], (sum(width_list[0:i]) + offset, initial_v_offset+title_space))

    return dst

def centered_text(W, H, msg, font_size=20):
    '''
    font must be in main directory
    '''
    
    img = Image.new('RGB', (W, H), color = 'WHITE')

    fnt = ImageFont.truetype('arial.ttf', font_size)
    d = ImageDraw.Draw(img)
    w, h = d.textsize(msg, fnt)
    d.text(((W-w)/2,(H-h)/2-font_size/10), msg, fill="black", font=fnt)

    return img