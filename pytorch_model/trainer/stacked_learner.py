import copy
import os
import pickle
import glob
import warnings
import random
import numpy as np
import pandas as pd
from timeit import default_timer as timer
import datetime
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from metrics.confusion_matrix import *
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from hyperopt import fmin, tpe, atpe, hp, STATUS_OK, STATUS_FAIL, Trials
from hyperopt.pyll import scope
import hyperopt.pyll
from functools import partial
from dataloaders import SubsetRandomDataLoader

class Stacking:
    def __init__(self, dataset, model_ID_to_stack = [], model_ID = '', plot_distribution = True):
        '''
        Args:
            - dataset: torch.utils.data.Dataset used to train input models
            - model_ID_to_stack: list of input models to be stacked. Use reload_model_ID (e.g. 'AkineticNoDiff_Resnet34_BCE')
                                corresponding .pth and .pickle are expected to be in ./checkpoints
            - model_ID: ID for the stacked model, used as prefix and to create folder in ./results
            - plot_distribution: plot distribution of input predicted probabilities and correlation matrix
            
        Relevant Outputs (in self):
            - reloaded_model_split_dict: main keys: model_ID_to_stack
                                         second level keys: 'split_1', ... , 'split_k'
                                         third level keys: 'ind', 'preds', 'binary_thresh', 'multi_class_mode', 'split_path':
                                                - 'ind': list, validation indices of split
                                                - 'preds': pandas df ['pred', 'true'], validation predicted probabilities
                                                - 'binary_thresh': binary threshold used to get labels
                                                - 'multi_class_mode': multi_class_mode used to get labels if multi_class
                                                - 'invert_class': True if inverted labels have been used to train the model.
                                                                  Inverted in Trainer.run() and then back to normal.
                                                                  With Evaluator.train_predict() predicted prob must be inverted.
                                                - 'multi_class': if True multi class task
                                                - 'split_path': path to reload *splitK_final_model.pth
        '''
        
        _=os.makedirs(os.path.join('results', model_ID, 'tuning_results'), exist_ok=True)
        reloaded_model_split_dict = {}
        tot_splits = []
        multi_class = []

        # reload models and check same number of splits and train/validation indices
        for mod in model_ID_to_stack:
            avail_splits_path = sorted(glob.glob('./checkpoints/'+mod+'_split*_final_model.pth'))
            avail_splits = sorted([int(x.replace('./checkpoints/'+mod+'_split', '')[0]) for x in avail_splits_path])
            if list(range(1, len(avail_splits)+1)) != avail_splits:
                raise ValueError('missing splits for '+mod+' - expected:'+list(range(1, len(avail_splits)+1))+' - found:'+avail_splits)
            else:
                tot_splits.append(len(avail_splits))

            # reload binary_threshold and multi_class_mode
            with open('checkpoints/'+mod+'_full_train_result.pickle', 'rb') as handle:
                model_pickle = pickle.load(handle)
            binary_thresh = model_pickle['options']['binary_thresh']
            multi_class_mode = model_pickle['options']['multi_class_mode']

            # reload split predictions
            split_dict = {}
            for s, split in enumerate(avail_splits_path):
                reload = torch.load(split)
                multi_class.append(reload['train_out']['options']['multi_class'])
                split_valid_ind = reload['train_out']['valid_idx']
                split_valid_preds = reload['train_out']['valid_preds']
                invert_class = reload['train_out']['options']['invert_class']
                split_dict['split_'+str(s+1)] = {'ind': sorted(split_valid_ind),
                                                 'preds': split_valid_preds,
                                                 'binary_thresh': binary_thresh,
                                                 'multi_class_mode': multi_class_mode,
                                                 'invert_class': invert_class,
                                                 'multi_class': reload['train_out']['options']['multi_class'],
                                                 'split_path': split}
            reloaded_model_split_dict[mod] = split_dict

        all_valid_ind = [[v['ind'] for v in split_dict.values()] for split_dict in reloaded_model_split_dict.values()]
        if len(np.unique(np.array(tot_splits))) != 1:
            display(pd.DataFrame({'model': model_ID_to_stack, 'Splits': tot_splits}))        
            raise ValueError('different number of split found')
        if sum([all_valid_ind[0] == all_valid_ind[i] for i in range(len(all_valid_ind))]) != len(model_ID_to_stack):
            raise ValueError('different split validation indices found')
        if (np.arange(dataset.df.shape[0]) != np.unique(np.array([y for x in all_valid_ind[0] for y in x ]))).any():
            raise ValueError('total observations in each split validation don\'t match total dataset observation')
        if len(np.unique(np.array(multi_class))) > 1:
            raise ValueError('different values for multi_class option.', multi_class)
        else:
            multi_class = np.unique(np.array(multi_class))[0]

        # prepare input dataset for 2nd level learner
        final_dataset = pd.concat([pd.concat([x['preds'] for x in mod.values()]).sort_index()\
                                   .drop(columns='true').rename(columns={"pred": mod_name})
                                   for mod_name, mod in reloaded_model_split_dict.items()], axis=1, join = 'inner')\
                        .merge(dataset.df[['label', 'meaning']].rename(columns={"label": 'true'}), left_index=True, right_index=True)

        avail_class = dataset.df[['label', 'meaning']].drop_duplicates().sort_values(by=['meaning']).reset_index(drop=True)
        avail_label = avail_class.label.values.astype(final_dataset.true.values.dtype)

        # plot distribution of input predictions and their correlation matrix
        if plot_distribution: Stacking.plot_input_pred_distribution(final_dataset, reloaded_model_split_dict, avail_class, model_ID)
        
        self.model_ID = model_ID
        self.model_ID_to_stack = model_ID_to_stack
        self.dataset = dataset
        self.reloaded_model_split_dict = reloaded_model_split_dict
        self.multi_class = multi_class
        self.avail_label = avail_label
        self.avail_class = avail_class
        self.final_dataset = final_dataset
        
    
    def perform_stacking(self, binary_thresh_1st_level = 'best',
                         multi_class_mode_1st_level = 'max', learner_2nd_level_set = [], n_splits_2nd_level = 5,
                         binary_thresh_2nd_level = 'best', perf_metric = {},
                         multi_class_mode_2nd_level = 'max', split_seed_2nd_level = 66, optim_max_iter = 20):
        '''
        Perform stacking based on learner_2nd_level_set
        
        Args:
            - binary_thresh_1st_level: binary threshold used when learner_2nd_level_set in ['average', 'majority_vote_by_label', 'median'].
                                        See averaging_method()
            - multi_class_mode_1st_level: used when learner_2nd_level_set in ['average', 'majority_vote_by_label', 'median'].
                                        See averaging_method()
            - learner_2nd_level_set: list of learner to be trained. See tune_learner() for available learners.
            - n_splits_2nd_level: number of fold to Cross-Validate hyperparameters tuning for 2nd level learner
            - binary_thresh_2nd_level: binary threshold to evaluate predicted labels when learner_2nd_level_set
                                    NOT in ['average', 'majority_vote_by_label', 'median']. See tune_learner()
            - perf_metric: dict of {'name': string
                                    'metric': function(true, pred), metric to evaluate average performance on validation sets
                                    'minimize': bool, if True perf_metric is minimized, maximized otherwise}
                           used to evaluate performance and to select best hyperparameters
            - multi_class_mode_2nd_level: used when learner_2nd_level_set
                                    NOT in ['average', 'majority_vote_by_label', 'median']. See tune_learner()
            - split_seed_2nd_level: seed for cross-validation on 2nd level learner model
            - optim_max_iter: maximum number of iteration for Bayesian optimizer in hyperparameters tuning
        '''
        
        if learner_2nd_level_set == []:
            learner_2nd_level_set = ['average', 'majority_vote_by_label', 'median',
                                     'GBM', 'RandomForest', 'KNeighbors', 'SVM', 'passive_aggressive', 'perceptron_elastic_net',
                                     'polynomial_regression', 'logistic_regression', 'elastic_net', 'MLP']
        
        # reload input model prediction and evaluate performance to be compared with ensemble
        tuning_results = []
        for mod_name, mod in self.reloaded_model_split_dict.items():

            split_perf = []
            final_pred_label = []
            for split in mod.values():
                pred_label = evaluate_prediction(split['preds'], binary_thresh=split['binary_thresh'],
                                                 multi_class=self.multi_class, multi_class_label=self.avail_label,
                                                 multi_class_mode=split['multi_class_mode']).drop(columns='pred_perc')
                split_perf.append(perf_metric['metric'](split['preds'].true, pred_label))
                if len(final_pred_label) == 0:
                    final_pred_label = pred_label
                else:
                    final_pred_label = final_pred_label.append(pred_label)
                final_pred_label = final_pred_label.sort_index()

            out = pd.DataFrame({'learner': mod_name,
                                'n_splits': len(split_perf),
                                'perf_metric': perf_metric['name'] if perf_metric != {} else '',
                                'avg_fold_perf': np.mean(split_perf),
                                'full_set_perf': perf_metric['metric'](self.final_dataset.true, final_pred_label.pred_label),
                                'best_binary_thresh': split['binary_thresh'],
                                'time': '0:00:00',
                                'size': '',
                                'models_path': ''}, index = [0])

            if len(tuning_results) == 0:
                tuning_results = out
            else:
                tuning_results = tuning_results.append(out)
        
        # perform stacking
        final_dataset_dict = {}
        for learner_2nd_level in learner_2nd_level_set:

            out = Stacking._perform_stacking(self.final_dataset, self.model_ID_to_stack, self.model_ID, self.avail_label,
                                             self.multi_class, binary_thresh_1st_level, multi_class_mode_1st_level,
                                             learner_2nd_level, n_splits_2nd_level, binary_thresh_2nd_level, perf_metric,
                                             multi_class_mode_2nd_level, split_seed_2nd_level, optim_max_iter)
            tuning_results = tuning_results.append(out['performance_row'])
            final_dataset_dict[learner_2nd_level] = out['final_dataset']
        tuning_results = tuning_results.reset_index(drop=True)
        
        print('\n\nStacked learner results:')
        display(tuning_results.drop(columns='models_path'))
        
        self.tuning_results = tuning_results
        self.final_dataset_dict = final_dataset_dict
        self.perf_metric = perf_metric
        
    def select_best_learner(self, sort_by = 'avg_fold_perf'):
        '''
        sort_by: 'avg_fold_perf' for average fold performance or 'full_set_perf' for performance on full dataset
        '''
        
        best_learner_row = self.tuning_results[~self.tuning_results.learner.isin(self.model_ID_to_stack)].sort_values(by=['avg_fold_perf'],
                       ascending=(True if self.perf_metric['minimize'] else False)).iloc[0]
        best_learner = best_learner_row.learner
        best_binary_thresh = best_learner_row.best_binary_thresh
        best_cv_final_dataset = self.final_dataset_dict[best_learner]
        self.tuning_results.insert(0, 'best', np.where(self.tuning_results.learner == best_learner, 'x', ''))
        self.tuning_results.drop(columns='models_path')\
                            .to_csv(os.path.join('results', self.model_ID, self.model_ID+'_learner_summary.csv'), index=False, sep=',')
        
        # plot stacking results
        Stacking.plot_stacked_results(self.tuning_results, best_learner_row, self.perf_metric, self.model_ID_to_stack, self.model_ID)
        
        # plot confusion matrix for best learner
        if best_learner_row.models_path != '':
            with open(best_learner_row.models_path, 'rb') as handle:
                model_pickle = pickle.load(handle)

            binary_thresh = model_pickle['binary_thresh']
            if binary_thresh != best_learner_row.best_binary_thresh:
                print('\n\n######## warning: binary_thresh mismatch for best learner')

            conf_mat_list_train = []
            conf_mat_list_valid = []
            mcc_list_train = []
            mcc_list_valid = []
            for split in model_pickle['split_models']:
                train_pred = evaluate_prediction(split['train_pred'], binary_thresh=binary_thresh).pred_label
                valid_pred = evaluate_prediction(split['valid_pred'], binary_thresh=binary_thresh).pred_label
                conf_mat_list_train.append(confusion_matrix(split['train_pred'].true, train_pred, labels=self.avail_label))
                conf_mat_list_valid.append(confusion_matrix(split['valid_pred'].true, valid_pred, labels=self.avail_label))
                warnings.filterwarnings("ignore")
                mcc_list_train.append(matthews_corrcoef(split['train_pred'].true, train_pred))
                mcc_list_valid.append(matthews_corrcoef(split['valid_pred'].true, valid_pred))
                warnings.resetwarnings()

            # render box plot confusion matrix
            conf_train_split = render_conf_mat_boxplot(conf_mat_list_train, mcc_list_train, self.avail_class.meaning, title='Train')
            plt.close(conf_train_split)
            conf_valid_split = render_conf_mat_boxplot(conf_mat_list_valid, mcc_list_valid, self.avail_class.meaning, title='Validation')
            plt.close(conf_valid_split)
            train_path=os.path.join('results', self.model_ID, 'avg_split_conf_mat_train.png')
            valid_path=os.path.join('results', self.model_ID, 'avg_split_conf_mat_valid.png')
            merge_path=os.path.join('results', self.model_ID, 'best_learner_conf_mat.png')
            conf_train_split.savefig(train_path, bbox_inches="tight", pad_inches=0.5)
            conf_valid_split.savefig(valid_path, bbox_inches="tight", pad_inches=0.5)
            get_concat_h([Image.open(train_path), Image.open(valid_path)],
                         add_main_title='Split Average for best learner: '+best_learner_row.learner,
                         font_size=20).save(merge_path)
            os.remove(train_path)
            os.remove(valid_path)
            display(Image.open(merge_path))
        else:
            conf_mat, _, _, _, _ = plot_confusion_matrix(best_cv_final_dataset.true, best_cv_final_dataset.pred_label, self.avail_class,
                                  title='Full set for best learner: '+best_learner_row.learner)
            conf_mat.savefig(os.path.join('results', self.model_ID, 'best_learner_conf_mat.png'))

    def predict(self, new_prediction_path = [], stacked_learner = '', stacked_learner_path = '',
                input_split_averaging = 'average', final_output_averaging = 'average', final_binary_thresh=''):
        '''
        Predict on new images

        Args:
            - new_prediction_path: list of str paths to images to be predicted. Same format of 'image' column of DataLoader.df
            - stacked_learner: str of stacked learner to be used
            - stacked_learner_path: path to stacked learner pickle.
                                    If '' 'checkpoints/'+self.model_ID+'_final_'+stacked_learner+'.pickle' is loaded
            - input_split_averaging: averaging method to average split probabilities for each input model. See averaging_methods()
            - final_output_averaging: averaging method to average split probabilities for final stacked_learner. See averaging_methods()
            - final_binary_thresh: if not '' will override the best binary threshold provided with stacked_learner

        Output:
            - dict of:
                - 'final_input_prediction': averaged probabilities from input models. Pandas df ['model_1', ..., 'model_n']
                - 'final_output_prediction': averaged probabilities from stacked learner.
                                             Pandas df ['split_1', ..., 'split_k', 'pred', 'binary_threshold', 'average_method', 'pred_label']
                - 'prediction': final average probabilities and corresponding labels.
                                Pandas df ['image', 'pred', 'binary_threshold', 'average_method', 'pred_label']
        '''

        if input_split_averaging not in ['average', 'majority_vote_by_label', 'median']:
                raise ValueError('Please provide valid input_split_averaging in [\'average\', \'majority_vote_by_label\', \'median\']. Current is', input_split_averaging)
        if final_output_averaging not in ['average', 'majority_vote_by_label', 'median']:
            raise ValueError('Please provide valid final_output_averaging in [\'average\', \'majority_vote_by_label\', \'median\']. Current is', final_output_averaging)
        if stacked_learner not in ['GBM', 'RandomForest', 'KNeighbors', 'SVM', 'passive_aggressive', 'perceptron_elastic_net',
                                  'polynomial_regression', 'logistic_regression', 'elastic_net', 'MLP']:
            raise ValueError('\''+stacked_learner+'\' not supported. See docs for implemented learners.')

        # input models
        print('\n --- Evaluating input models predictions ... ', end='')
        start = timer()

        # set DataLoader
        pred_dataset = copy.deepcopy(self.dataset)
        pred_dataset.df = pd.DataFrame({'image': new_prediction_path,
                                        'label': np.repeat(self.dataset.df.label[0], len(new_prediction_path))})
        pred_loader = SubsetRandomDataLoader( dataset = pred_dataset, indexes = np.array(pred_dataset.df.index), batch_size = 1)

        # get predictions for each split, average with input_split_averaging, for each input model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        final_input_prediction = pred_dataset.df.drop(columns='image').rename(columns={'label': 'true'})  # true is fictitious and dropped after
        for mod_name, mod in self.reloaded_model_split_dict.items():

            mod_prediction = pred_dataset.df.drop(columns='image').rename(columns={'label': 'true'})  # create input for averaging_methods()
            split_columns = []
            for split_name, split in mod.items():
                split_columns.append(split_name)
                split_binary_thresh = split['binary_thresh']
                split_multi_class_mode = split['multi_class_mode']
                split_invert_class = split['invert_class']
                split_multi_class = split['multi_class']
                split_reload = torch.load(split['split_path'])
                split_model = split_reload['model']
                split_model = split_model.to(device)
                split_model.load_state_dict(split_reload['model_state_dict'])
                split_model.eval()

                split_prediction = pd.DataFrame(columns=['index', split_name])
                for i, (mod_input, label, index) in enumerate(pred_loader):
                    mod_input = mod_input.to(device)
                    pred = split_model(mod_input)
                    pred = 1 - pred if split_invert_class else pred
                    split_prediction = split_prediction.append(pd.DataFrame({'index': int(index),
                                                                             split_name: pred.detach().cpu().numpy().ravel()}, index = [0]))
                split_prediction = split_prediction.set_index('index').sort_index()
                mod_prediction = mod_prediction.merge(split_prediction, left_index=True, right_index=True)

            # evaluate model final prediction with averaging_methods()
            out = Stacking.averaging_methods(mod_prediction, average_method = input_split_averaging, binary_thresh = split_binary_thresh,
                                             multi_class = split_multi_class, multi_class_label = self.avail_class,
                                             multi_class_mode = split_multi_class_mode, perf_metric = {}, model_ID = '', show_info = False)

            final_input_prediction = final_input_prediction.merge(out['final_dataset'].drop(columns=['true', 'pred_label'] + split_columns)\
                                                                  .rename(columns={'pred': mod_name}), left_index=True, right_index=True)
        final_input_prediction = final_input_prediction.drop(columns='true')
        print('Done in', str(datetime.timedelta(seconds=round(timer()-start))))

        # stacked learner
        print('\n --- Evaluating stacked learner predictions ... ', end='')
        start = timer()

        # reload stacked_learner
        stacked_learner_path = 'checkpoints/'+self.model_ID+'_final_'+stacked_learner+'.pickle' if stacked_learner_path == '' else stacked_learner_path
        with open(stacked_learner_path, 'rb') as handle:
            learner = pickle.load(handle)

        # predict on each split
        if final_binary_thresh == '': final_binary_thresh = learner['binary_thresh']
        final_output_prediction = pred_dataset.df.drop(columns='image').rename(columns={'label': 'true'})  # true is fictitious and dropped later
        for s, split in enumerate(learner['split_models']):
            split_learner = split['learner']
            if 'predict_proba' in dir(split_learner):
                split_predicted_prob = split_learner.predict_proba(final_input_prediction)[:, 1]
            else:
                split_predicted_prob = split_learner.predict(final_input_prediction).astype(float)
                final_binary_thresh = 0.5  # should be already set to 0.5

            final_output_prediction = final_output_prediction.merge(pd.DataFrame({stacked_learner+'_split_'+str(s+1): split_predicted_prob}),
                                                                    left_index=True, right_index=True)
        final_output_prediction = Stacking.averaging_methods(final_output_prediction, average_method = final_output_averaging,
                                                             binary_thresh = final_binary_thresh, multi_class_label = self.avail_class,
                                                             show_info = False)['final_dataset']
        final_output_prediction = final_output_prediction.drop(columns='true')
        prediction = pred_dataset.df.merge(final_output_prediction[['pred', 'pred_label']],
                                           left_index=True, right_index=True).drop(columns='label')
        prediction.insert(2, 'binary_threshold', final_binary_thresh)
        prediction.insert(3, 'average_method', final_output_averaging)
        final_output_prediction.insert(s+2, 'binary_threshold', final_binary_thresh)
        final_output_prediction.insert(s+3, 'average_method', final_output_averaging)
        print('Done in', str(datetime.timedelta(seconds=round(timer()-start))))
        
        return {'final_input_prediction': final_input_prediction,
                'final_output_prediction': final_output_prediction,
                'prediction': prediction}
            
            
    @staticmethod
    def plot_input_pred_distribution(final_dataset, reloaded_model_split_dict, avail_class, model_ID):
        '''
        Plot distribution of input predictions and their correlation matrix
        '''

        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        del_path = []

        # plot prediction distributions
        fig, ax = plt.subplots(1, len(avail_class), figsize=(12,7), sharey=False)
        ax.flatten()
        for cl, class_row in avail_class.iterrows():
            class_meaning = class_row.meaning
            class_label = class_row.label
            legend_elements = []
            for i, (mod_name, mod) in enumerate(reloaded_model_split_dict.items()):
                pred_class_dataset = pd.concat([x['preds'][x['preds'].true == class_label] for x in mod.values()])\
                                    .sort_index().reset_index(drop=True)
#                 class_meaning = 'Akinetic' if class_label == 1 else 'Normokinetic'
#                 if mod_name == 'AkineticNoDiff_Resnet34_BCE': mod_name = 'BCE'
#                 if mod_name == 'AkineticNoDiff_Resnet34_BCE_inv_class': mod_name = 'BCE - Inv Class'
#                 if mod_name == 'AkineticNoDiff_Resnet34_FocLos': mod_name = 'Focal Loss'
#                 if mod_name == 'AkineticNoDiff_Resnet34_FocLos_inv_class': mod_name = 'Focal Loss - Inv Class'
                ax[cl].scatter(pred_class_dataset.index, pred_class_dataset.pred, c=colors[i], alpha=0.5)
                ax[cl].set_title('True class: '+ class_meaning +
                                 (' ['+str(int(class_label * 100))+'%]' if len(avail_class) == 2 else ''), size = 20)
                ax[cl].set_xticks([])
                ax[cl].scatter([0,0], [0,1], c='white', alpha=0)  # transparent points just to ensure [0,1] y-range to be displayed
                if cl == 0: ax[cl].set_ylabel('Predicted Probability', size = 15)
                legend_elements.append(plt.plot([],[], marker="o", ms=10, ls="", color=colors[i], lw=2, label=mod_name)[0]) 

        lg = fig.legend(handles=legend_elements, bbox_to_anchor=(0.5, 0.15), loc='center', fontsize = 12, title='Model', ncol=3)
        title = lg.get_title()
        title.set_fontsize(15)
        fig.suptitle('Predicted values for each input model', fontsize=23)
        fig.tight_layout(rect=[0, 0.3, 1, 0.9])
        plt.show()
        fig.savefig(os.path.join('results', model_ID, model_ID+'_00_prediction_distr.png'), bbox_inches='tight')
        del_path.append(os.path.join('results', model_ID, model_ID+'_00_prediction_distr.png'))

        # plot correlation matrix
        df = final_dataset.drop(columns = ['true', 'meaning'])
        corr_mat = df.corr().to_numpy()
        for i in range(len(corr_mat)):
            corr_mat[i, i]=None

        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(corr_mat, cmap='bwr', vmax = 1, vmin = -1)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(corr_mat.shape[1]),
               yticks=np.arange(corr_mat.shape[0]),
               xticklabels=df.columns, yticklabels=df.columns,
               title='Correlation Matrix')
        # ax.xaxis.set_label_position('top') 
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", va='center',
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(corr_mat.shape[0]):
            for j in range(corr_mat.shape[1]):
                if i != j: ax.text(j, i, format(corr_mat[i, j], '.2f'), ha="center", va="center")
        fig.tight_layout()
        plt.show()
        fig.savefig(os.path.join('results', model_ID, model_ID+'_00_corr_matrix.png'), bbox_inches='tight')
        del_path.append(os.path.join('results', model_ID, model_ID+'_00_corr_matrix.png'))

        get_concat_v([Image.open(p) for p in del_path]).save(os.path.join('results', model_ID, model_ID+'_Input_models_predicted_values.png'))
        _ = [os.remove(p) for p in del_path]
     
    @staticmethod
    def plot_stacked_results(tuning_results, best_learner_row, perf_metric, model_ID_to_stack, model_ID):

        x = np.arange(0, len(tuning_results) + 1)
        y = np.hstack([tuning_results[tuning_results.learner.isin(model_ID_to_stack)].avg_fold_perf, 0,
                     tuning_results[~tuning_results.learner.isin(model_ID_to_stack)].avg_fold_perf]).flatten()
        label = list(tuning_results[tuning_results.learner.isin(model_ID_to_stack)].learner) + [''] +\
                list(tuning_results[~tuning_results.learner.isin(model_ID_to_stack)].learner)
        stack_colors = list(np.repeat('gray', len(tuning_results) - len(model_ID_to_stack)))
        stack_colors[tuning_results[tuning_results.learner == best_learner_row.learner].index[0] - len(model_ID_to_stack)] = 'blue'
        colors = list(np.repeat('red', len(model_ID_to_stack) + 1)) + stack_colors

        fig, ax = plt.subplots(figsize = (13, 7))
        ax.bar(x, height=y, color=colors)
        ax.set_xlabel('Input model and stacked learner', size = 18)
        ax.set_ylabel(perf_metric['name'], size = 18)
        ytick, _ = plt.yticks()
        ax.set_xticks(x)
        ax.set_xticklabels(label, size = 13)
        ax.set_title(model_ID + ': '+perf_metric['name']+ ' for input model and stacked learner', size = 23)
        ax.set_yticklabels([str(int(y*100))+'%' for y in ytick], size = 13)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", va='center',
                 rotation_mode="anchor")
        # values on bars
        for i, v in enumerate(y):
            plt.text(x[i] - 0.45, v + 0.01, str(round(v*100, 2)) if v != 0 else '', size=12)
        legend_elements = [Patch(facecolor='red', edgecolor='red', label='Input Models'),
                           Patch(facecolor='gray', edgecolor='gray', label='Stacked Learners'),
                           Patch(facecolor='blue', edgecolor='blue', label='Best Stacked Learners')]
        lg = fig.legend(handles=legend_elements, bbox_to_anchor=(1.05, 0.9), loc='center', fontsize = 14, ncol=1)
        plt.show()
        fig.savefig(os.path.join('results', model_ID, model_ID+'_Stacked_models_performance.png'), bbox_inches='tight')
    
    
    @staticmethod
    def averaging_methods(final_dataset, average_method = '', binary_thresh = 0.5, multi_class = False, multi_class_label = [],
                          multi_class_mode = 'max', perf_metric = {}, model_ID = '', show_info = True):
        '''
        Args:
            - final_dataset: dataset with prediction from each model with columns ['feature_1', ..., 'feature_p', 'true']
            - average_method: method to average prediction from all models. Available:
                                - 'average': simple mean of prediction and final label (e.g. 0 or 1) evaluated with evaluate_prediction()
                                - 'majority_vote_by_label': evaluate label for each model with evaluate_prediction() and then
                                                            take mode of labels. Better with more than 2 models
                                - 'median': take median of predictions and then evaluate label with evaluate_prediction()
            - binary_thresh: threshold for binary classification on predicted values
                             Use 'best' to automatically find threshold that maximize (or minimize) perf_metric
            - multi_class: True if multiclass task
            - multi_class_label: list of labels for argmax in multiclass task
            - multi_class_mode: how to set predicted class given the probabilities array. 'max': takes highest probability
            - perf_metric: dict of {'name': string
                                    'metric': function(true, pred), metric to evaluate average performance on validation sets
                                    'minimize': bool, if True perf_metric is minimized, maximized otherwise}
                           if available, performance on final_dataset 'true' and 'pred_label' are evaluated
            - model_ID: string used as prefix and folder for and csv
            - show_info: if True print info

        Output:
            - dict with:
                'performance_row': pandas df with ['learner'=average_method, 'n_splits'=0, 'perf_metric', 'best_binary_thresh',
                                'avg_fold_perf', 'full_set_perf', 'time', 'size', 'models_path'] for best parameters found
                'final_dataset': pandas df, same columns of work_set plus ['pred', 'pred_label']
                                final prediction and evaluated label, respectively
        '''

        if average_method not in ['average', 'majority_vote_by_label', 'median']:
            raise ValueError('Please provide valid average_method in [\'average\', \'majority_vote_by_label\', \'median\']. Current is', average_method)

        if binary_thresh == 'best' and perf_metric == {}:
            raise ValueError('If binary_thresh = \'best\' perf_metric must be provided')

        if show_info: print('\n --- Evaluating '+average_method+' ... ', end='')
        start = timer()
        avail_models = final_dataset.drop(columns = ['true']).columns
        best_binary_thresh = binary_thresh

        if average_method == 'average':
            final_dataset['pred'] = final_dataset[avail_models].mean(axis=1)

        if average_method == 'median':
            if len(avail_models) == 2: warnings.warn('''\n\nusing \'majority_vote\' for 2 models only may not be the best choice,
                                                                in case of draw lowest label will be taken \n''')

            final_dataset['pred'] = final_dataset[avail_models].median(axis = 1)

        if binary_thresh != 'best' and average_method != 'majority_vote_by_label':
            final_dataset = final_dataset.merge(evaluate_prediction(final_dataset, binary_thresh=binary_thresh,
                                                                    multi_class=multi_class, multi_class_label=multi_class_label,
                                                                    multi_class_mode=multi_class_mode).drop(columns='pred_perc'),
                                                left_index=True, right_index=True)

        # binary_thresh tuning for 'average' and 'median'
        if binary_thresh == 'best' and average_method != 'majority_vote_by_label':
            best_binary_results = pd.DataFrame(columns = ['threshold', 'full_set_perf_metric'])
            for thresh in np.arange(0, 1, 0.01).round(3):
                pred_label = evaluate_prediction(final_dataset, binary_thresh=thresh).pred_label.ravel()
                best_binary_results = best_binary_results.append(
                                            pd.DataFrame({'threshold': thresh,
                                                          'full_set_perf_metric': perf_metric['metric'](final_dataset.true, pred_label)},
                                                         index=[0]))
            best_row = best_binary_results.sort_values(by=['full_set_perf_metric'],
                                                        ascending=(True if perf_metric['minimize'] else False)).iloc[0]
            best_binary_results.insert(0, 'best', np.where(best_binary_results.threshold == best_row.threshold, 'x', ''))
            best_binary_thresh = best_row.threshold
            final_dataset = final_dataset.merge(evaluate_prediction(final_dataset, binary_thresh=best_binary_thresh).drop(columns='pred_perc'),
                                                left_index=True, right_index=True)

        if average_method == 'majority_vote_by_label':
            if len(avail_models) == 2: warnings.warn('''\n\nusing \'majority_vote\' for 2 models only may not be the best choice,
                                                                in case of draw lowest label will be taken \n''')

            binary_thresh_set = np.arange(0, 1, 0.01).round(3) if binary_thresh == 'best' else [binary_thresh]
            best_binary_results = pd.DataFrame(columns = ['threshold', 'full_set_perf_metric'])
            vote_dict = {}
            for thresh in binary_thresh_set:
                vote = final_dataset.drop(columns=avail_models)
                for mod in avail_models:
                    single_vote = final_dataset[[mod, 'true']].rename(columns={mod: 'pred'})
                    single_vote = single_vote.merge(evaluate_prediction(single_vote, binary_thresh=thresh).drop(columns='pred_perc'),
                                                    left_index=True, right_index=True)
                    single_vote = single_vote.rename(columns={'pred_label': mod}).drop(columns=['pred', 'true'])
                    vote = vote.merge(single_vote, left_index=True, right_index=True)
                vote['pred_label'] = vote[avail_models].mode(axis = 1).iloc[:, 0].astype(vote[avail_models[0]].dtype)
                vote_dict[str(thresh)] = vote
                if binary_thresh == 'best':
                    best_binary_results = best_binary_results.append(
                                                pd.DataFrame({'threshold': thresh,
                                                              'full_set_perf_metric': perf_metric['metric'](vote.true, vote.pred_label)},
                                                            index=[0]))
            if binary_thresh == 'best':
                best_row = best_binary_results.sort_values(by=['full_set_perf_metric'],
                                                            ascending=(True if perf_metric['minimize'] else False)).iloc[0]
                best_binary_results.insert(0, 'best', np.where(best_binary_results.threshold == best_row.threshold, 'x', ''))
                best_binary_thresh = best_row.threshold
                vote = vote_dict[str(best_binary_thresh)]
            final_dataset['pred_label'] = vote[avail_models].mode(axis = 1).iloc[:, 0].astype(vote[avail_models[0]].dtype)
            final_dataset['pred'] = final_dataset['pred_label'].astype(final_dataset[avail_models[0]].dtype)

        if show_info: print('Done in', str(datetime.timedelta(seconds=round(timer()-start))))
        if perf_metric != {}:
            full_set_perf = round(perf_metric['metric'](final_dataset.true, final_dataset.pred_label), 5)
            if show_info: print('     '+perf_metric['name']+': '+str(full_set_perf)+ ' (full set)')
        else:
            full_set_perf = 0
        binary_thresh_path = os.path.join('results', model_ID, 'tuning_results', model_ID+'_'+average_method+'_binary_thresh_tuning.csv')
        if binary_thresh == 'best':
            best_binary_results.to_csv(binary_thresh_path, index=False, sep=',')
            if show_info: print('\n     '+'binary threshold tuning log saved in '+binary_thresh_path)

        out = {'performance_row': pd.DataFrame({'learner': average_method,
                                                'n_splits': 0,
                                                'perf_metric': perf_metric['name'] if perf_metric != {} else '',
                                                'avg_fold_perf': full_set_perf,
                                                'full_set_perf': full_set_perf,
                                                'best_binary_thresh': best_binary_thresh,
                                                'time': str(datetime.timedelta(seconds=round(timer()-start))),
                                                'size': '',
                                                'models_path': ''}, index = [0]),
               'final_dataset': final_dataset}

        return out

    @staticmethod
    def tune_binary_threshold_with_cv(split_list, perf_metric, perf_metric_minimize = False):
        '''
        Find best binary threshold optimizing perf_metric on validation set AVERAGE

        Args:
            - split_list: list of dictionaries:
                                            - 'train_pred': pandas df with ['pred', 'true'] for train set
                                            - 'valid_pred': ['pred', 'true'] for validation set
                          for each split
            - perf_metric: performance metric function with inputs (true_label, predicted_label)
            - perf_metric_minimize: if False best threshold maximises perf_metric, minimize otherwise

        Output:
            - dict:
                - 'best_binary_results': pandas df of all tested 'threshold' with corresponding
                                        'valid_avg_perf_metric' AVERAGE perf_metric on validation set and
                                        'split_train_perf', 'split_valid_perf' with all values used to evaluate
                                        average performance
                - 'best_row': best_binary_results row corresponding to best threshold.
                              Used to extract 'split_train_perf', 'split_valid_perf' and best 'threshold'
        '''

        best_binary_results = pd.DataFrame(columns = ['threshold', 'valid_avg_perf_metric', 'split_train_perf', 'split_valid_perf'])
        for thresh in np.arange(0, 1, 0.01).round(3):
            split_train_perf = []
            split_valid_perf = []
            for split_set in split_list:
                split_train_label = evaluate_prediction(set_df = split_set['train_pred'], binary_thresh = thresh).pred_label.ravel()
                split_valid_label = evaluate_prediction(set_df = split_set['valid_pred'], binary_thresh = thresh).pred_label.ravel()
                split_train_perf.append(perf_metric(split_set['train_pred'].true, split_train_label))
                split_valid_perf.append(perf_metric(split_set['valid_pred'].true, split_valid_label))
            avg_valid_perf = np.mean(split_valid_perf)
            best_binary_results = best_binary_results.append(pd.DataFrame({'threshold': thresh,
                                                                           'valid_avg_perf_metric': avg_valid_perf,
                                                                           'split_train_perf': [split_train_perf],
                                                                           'split_valid_perf': [split_valid_perf]}))
        best_row = best_binary_results.sort_values(by=['valid_avg_perf_metric'],
                                                                ascending=(True if perf_metric_minimize else False)).iloc[0]
        best_binary_results.insert(0, 'best', np.where(best_binary_results.threshold == best_row.threshold, 'x', ''))

        return {'best_binary_results': best_binary_results,
                'best_row': best_row}

    @staticmethod
    @ignore_warnings(category=ConvergenceWarning)
    def cross_validated_perf(learner, work_set, perf_metric, n_splits = 5, random_seed = 66, binary_thresh = 0.5,
                             multi_class = False, multi_class_label = [], multi_class_mode = 'max', save_split_models = False):
        '''
        Args:
            - work_set: pandas df of probabilities with columns ['feature_1', ..., 'feature_p', 'true']
            - learner: learner. If learner doesn't allow to predict probabilities,
                                only 1.0 and 0.0 are assigned and binary_thresh is set to 0.5
            - perf_metric: dict of {'metric': function(true, pred), metric to evaluate average performance on validation sets
                                    'minimize': bool, if True perf_metric is minimized, maximized otherwise}
            - n_splits: number of fold to Cross-Validate performance
            - random_seed: seed for StratifiedKFold
            - binary_thresh: threshold for binary classification on predicted values.
                             Use 'best' to automatically find threshold that maximize (or minimize) AVERAGE perf_metric on VALIDATION set
            - multi_class: True if multiclass task
            - multi_class_label: list of labels for argmax in multiclass task
            - multi_class_mode: how to set predicted class given the probabilities array. 'max': takes highest probability
            - save_split_models: if True save fitted learner for each split

        Output:
            dict of:
                - 'train_perf': list of perf_metric evaluated on train set for each split
                - 'valid_perf': list of perf_metric evaluated on validation set for each split
                - 'binary_thresh': binary threshold used. If binary_thresh = 'best', best threshold found
                - 'split_models': list of dict {'learner', 'train_pred', 'valid_pred'} of fitted learner for each split and
                                pandas df with ['pred', 'true'] for both train and validation set
                - 'binary_thresh_results': list of average performance for each threshold tested, if binary_thresh = 'best'
        '''

        perf_metric_minimize = perf_metric['minimize']
        perf_metric = perf_metric['metric']
        train_perf = []
        valid_perf = []
        split_models = []
        best_binary_list = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        for train_idx, validation_idx in skf.split(work_set, work_set.true):

            split_learner = copy.deepcopy(learner)
            X_split_train = work_set.iloc[train_idx].drop(columns = 'true')
            Y_split_train = work_set.iloc[train_idx]['true'].ravel()
            X_split_valid = work_set.iloc[validation_idx].drop(columns = 'true')
            Y_split_valid = work_set.iloc[validation_idx]['true'].ravel()

            # train and predict on train set and predict on valid set
            split_learner.fit(X_split_train, Y_split_train)
            if 'predict_proba' in dir(learner):
                Y_train_predicted_prob = split_learner.predict_proba(X_split_train)[:, 1]
                Y_valid_predicted_prob = split_learner.predict_proba(X_split_valid)[:, 1]
            else:
                # predict labels and convert to float as probabilities
                Y_train_predicted_prob = split_learner.predict(X_split_train).astype(float)
                Y_valid_predicted_prob = split_learner.predict(X_split_valid).astype(float)
                binary_thresh = 0.5
            if binary_thresh != 'best':
                Y_train_predicted_label = evaluate_prediction(set_df = pd.DataFrame({'pred': Y_train_predicted_prob,
                                                                                     'true': Y_split_train}),
                                                              binary_thresh=binary_thresh, multi_class=multi_class,
                                                              multi_class_label=multi_class_label,
                                                              multi_class_mode=multi_class_mode).pred_label.ravel()
                train_perf.append(perf_metric(Y_split_train, Y_train_predicted_label))

                Y_valid_predicted_label = evaluate_prediction(set_df = pd.DataFrame({'pred': Y_valid_predicted_prob,
                                                                                     'true': Y_split_valid}),
                                                              binary_thresh=binary_thresh, multi_class=multi_class,
                                                              multi_class_label=multi_class_label,
                                                              multi_class_mode=multi_class_mode).pred_label.ravel()
                valid_perf.append(perf_metric(Y_split_valid, Y_valid_predicted_label))

            split_models.append({'learner': split_learner,
                                 'train_pred': pd.DataFrame({'pred': Y_train_predicted_prob,
                                                             'true': Y_split_train}, index = train_idx),
                                 'valid_pred': pd.DataFrame({'pred': Y_valid_predicted_prob,
                                                             'true': Y_split_valid}, index = validation_idx)})

        if binary_thresh == 'best':
            thresh_tune = Stacking.tune_binary_threshold_with_cv(split_list = split_models, perf_metric = perf_metric,
                                                                 perf_metric_minimize = perf_metric_minimize)

            # save best threshold and update corresponding train and validation performance
            train_perf = thresh_tune['best_row'].split_train_perf
            valid_perf = thresh_tune['best_row'].split_valid_perf

        return {'train_perf': train_perf,
                'valid_perf': valid_perf,
                'binary_thresh': thresh_tune['best_row'].threshold if binary_thresh == 'best' else binary_thresh,
                'split_models': split_models if save_split_models else [],
                'binary_thresh_results': thresh_tune['best_binary_results'] if binary_thresh == 'best' else []}

    @staticmethod
    def objective_function(params, add_dict):
        '''
        Objective function for Bayesian Optimization.

        Args:
            - params: learner parameters
            - add_dict: additional "fixed" input:
                        - 'work_set': pandas df of probabilities with columns ['feature_1', ..., 'feature_p', 'true']
                        - 'learner_name': learner name
                        - 'perf_metric': dict of {'metric': function(true, pred), metric to evaluate average performance on validation sets
                                                  'minimize': bool, if True perf_metric is minimized, maximized otherwise}
                        optional:
                        - 'param_formatting': dict of {'param_nam': format} to force formatting
                        - 'pipeline_key': if learner is Pipeline, define key to which apply parameters
                        - 'cov_mat': covariance matrix if needed for some learners
                        - 'n_splits': 5, number of fold to Cross-Validate performance
                        - 'random_seed': 66, seed for StratifiedKFold
                        - 'binary_thresh': 0.5, threshold for binary classification on predicted values
                                            Use 'best' to automatically find threshold that maximize perf_metric
                        - 'multi_class': False, True if multiclass task
                        - 'multi_class_label': [], list of labels for argmax in multiclass task
                        - 'multi_class_mode': 'max', how to set predicted class given the probabilities array. 'max': takes highest probability
                        - 'model_ID': folder reference for error log

        Output:
            - dict with 'loss' to be minimized and 'status' for optimization successful completion
        '''

        # get add_dict values
        work_set = add_dict['work_set']
        perf_metric = add_dict['perf_metric']['metric']
        perf_metric_minimize = add_dict['perf_metric']['minimize']
        param_formatting = add_dict['param_formatting'] if 'param_formatting' in add_dict else {}
        learner_name = add_dict['learner_name'] if 'learner_name' in add_dict else 'learner'
        pipeline_key = add_dict['pipeline_key'] if 'pipeline_key' in add_dict else ''
        cov_mat = add_dict['cov_mat'] if 'cov_mat' in add_dict else None
        n_features = len(work_set.columns) - 1
        if (cov_mat != np.ones((n_features, n_features)) * -99).all() and params['metric'] == 'mahalanobis':
            params['metric_params'] = {'V': cov_mat}
        random_seed = add_dict['random_seed'] if 'random_seed' in add_dict else 66
        n_splits = add_dict['n_splits'] if 'n_splits' in add_dict else 5
        binary_thresh = add_dict['binary_thresh'] if 'binary_thresh' in add_dict else 0.5
        multi_class = add_dict['multi_class'] if 'multi_class' in add_dict else False
        multi_class_label = add_dict['multi_class_label'] if 'multi_class_label' in add_dict else []
        multi_class_mode = add_dict['multi_class_mode'] if 'multi_class_mode' in add_dict else 'max'
        model_ID = add_dict['model_ID'] if 'model_ID' in add_dict else ''

        # instantiate raw learner
        if learner_name == 'elastic_net':
            learner = SGDClassifier(loss = 'log', penalty = 'elasticnet', shuffle = True, random_state = 666, eta0 = 0.1,
                               learning_rate = 'adaptive', early_stopping = True, class_weight = 'balanced', n_jobs = -1)
        if learner_name == 'logistic_regression':
            learner = LogisticRegression(penalty = 'none', class_weight = 'balanced', max_iter = 2000,
                                     random_state = 666, solver = 'saga', n_jobs = -1)
        if learner_name == 'polynomial_regression':
            learner = Pipeline([('poly', PolynomialFeatures(degree=3)),
                            ('linear', LogisticRegression(penalty = 'none', class_weight = 'balanced', max_iter = 2000,
                                                          random_state = 666, fit_intercept=False, solver = 'saga', n_jobs = -1))])
        if learner_name == 'perceptron_elastic_net':
            learner = Perceptron(penalty = 'elasticnet', alpha = 0.0001, shuffle = True, early_stopping = True, 
                             class_weight = 'balanced', random_state = 666, n_jobs = -1)
        if learner_name == 'passive_aggressive':
            learner = PassiveAggressiveClassifier(early_stopping = True, shuffle = True,
                                              class_weight = 'balanced', random_state = 666, n_jobs = -1)
        if learner_name == 'SVM':
            learner = SVC(probability = True, shrinking = True, cache_size = 500,
                        class_weight = 'balanced', random_state = 666)
        if learner_name == 'KNeighbors':
            learner = KNeighborsClassifier(algorithm = 'brute', n_jobs = -1)
        if learner_name == 'RandomForest':
            learner = RandomForestClassifier(bootstrap = False,  class_weight = 'balanced',
                                         random_state = 666, n_jobs = -1)
        if learner_name == 'GBM':
            learner = GradientBoostingClassifier(loss = 'deviance', criterion = 'friedman_mse', random_state = 666)
        if learner_name == 'MLP':
            learner = MLPClassifier(solver = 'adam', shuffle = True, early_stopping = True, random_state = 666, max_iter=1000)

        raw_learner = copy.deepcopy(learner)
        
        # change parameters format if needed
        for k, v in param_formatting.items():
            params[k] = v(params[k])

        # apply params to learner, keeping the set ones (provided with learner)
        if pipeline_key == '':
            [setattr(learner, k, v) for k, v in params.items()]
        else:
            [setattr(learner[pipeline_key], k, v) for k, v in params.items()]
           
        # cross-validate
        with open(os.path.join('results', model_ID, 'error_'+learner_name+'.txt'), 'a') as fh:
            try:
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                cv_res = Stacking.cross_validated_perf(copy.deepcopy(learner), work_set, add_dict['perf_metric'], n_splits,
                                                       random_seed, binary_thresh, multi_class, multi_class_label,
                                                       multi_class_mode, save_split_models = False)
                warnings.resetwarnings()

                train_avg_perf = np.mean(cv_res['train_perf'])
                valid_avg_perf = np.mean(cv_res['valid_perf'])
                best_binary_thresh = cv_res['binary_thresh']
                binary_thresh_results = cv_res['binary_thresh_results']
                status = STATUS_OK
                loss = valid_avg_perf if perf_metric_minimize else -valid_avg_perf
            except Exception as e:
                best_binary_thresh = -1
                binary_thresh_results = []
                status = STATUS_FAIL
                loss = -1e9 if perf_metric_minimize else 1e9
                print(e, file=fh)

        return {'loss': loss, 'status': status, 'best_binary_thresh': best_binary_thresh, 'learner_model': learner_name,
                'binary_thresh_tuning': [binary_thresh_results], 'raw_learner': [raw_learner]}

    @scope.define
    def hidden_layer(n_layers, max_neuron_per_layer = 200, seed = 99):
        min_neuron_per_layer = 20
        if max_neuron_per_layer <= max_neuron_per_layer: max_neuron_per_layer = min_neuron_per_layer + 1
        np.random.seed(int(seed))
        arr = np.random.choice(np.arange(int(min_neuron_per_layer), int(max_neuron_per_layer)), int(n_layers), replace=True)
        return tuple(arr)
    
    @staticmethod
    def tune_learner(work_set, learner_name = '', perf_metric = {}, n_splits = 5, optim_max_iter = 100,
                     binary_thresh = 'best', multi_class = False, multi_class_label = [],
                     multi_class_mode = 'max', final_model_split_seed = 66, model_ID = ''):

        '''
        Tune hyperparameter with Bayesian Optimization for selected learner and returns cross-validated models (one for each split)
        for best learner, selected according to optimal average performance metric on validation splits
        
        https://github.com/hyperopt/hyperopt/wiki/FMin
        https://medium.com/vantageai/bringing-back-the-time-spent-on-hyperparameter-tuning-with-bayesian-optimisation-2e21a3198afb

        Args:
            - work_set: pandas df of probabilities with columns ['feature_1', ..., 'feature_p', 'true']
            - learner_name: one of 'GBM', 'RandomForest', 'KNeighbors', 'SVM', 'passive_aggressive', 'perceptron_elastic_net',
                            'polynomial_regression', 'logistic_regression', 'elastic_net', 'MLP'
            - perf_metric: dict of {'name': string
                                    'metric': function(true, pred), metric to evaluate average performance on validation sets
                                    'minimize': bool, if True perf_metric is minimized, maximized otherwise}
            - n_splits: number of split for both tuning and final models
            - optim_max_iter: maximum number of iteration for Bayesian optimizer
            - binary_thresh: if int, threshold for binary classification on predicted values.
                             if 'best', best threshold is selected optimizing average perf_metric on validation splits
            - multi_class: True if multiclass task
            - multi_class_label: list of labels for argmax in multiclass task
            - multi_class_mode: how to set predicted class given the probabilities array. 'max': takes highest probability
            - final_model_split_seed: seed for cross-validation on final model (to be different from the one used for single input model)
            - model_ID: string used as prefix and folder for pickle and csv

        Output:
            - dict with:
                'performance_row': pandas df with ['learner', 'n_splits', 'perf_metric', 'best_binary_thresh',
                                'avg_fold_perf', 'full_set_perf', 'time', 'size', 'models_path'] for best parameters found
                'final_dataset': pandas df, same columns of work_set plus ['pred', 'pred_label']
                                of final model Cross-Validated predicted probabilities and corresponding evaluated labels
            * learner is saved in a pickle file as dict{'train_perf': performance metric on train splits
                                                        'valid_perf': performance metric on validation splits
                                                        'binary_thresh': best binary_thresh
                                                        'split_models': list of [{'learner', 'train_pred', 'valid_pred'}] for each split
                                                        'binary_thresh_results': summary of threshold tested for optimal one (if any)}
        '''

        if learner_name not in ['GBM', 'RandomForest', 'KNeighbors', 'SVM', 'passive_aggressive', 'perceptron_elastic_net',
                              'polynomial_regression', 'logistic_regression', 'elastic_net', 'MLP']:
            raise ValueError('\''+learner_name+'\' not supported. See docs for implemented learners.')

            
        # define parameters space for learner to be tuned
        '''
        - 'param_space': dictionary of parameters to be optimized
        - 'param_choice': dict of {'param': [choices list]} for all params to be searched with hp.choice.
                            needed because hp.choice returns the index of the list and not the value (for best_param_dict)
        - 'param_formatting': dict of {'param': format} used to format parameters before being passed to learner
        - 'scope_func_setting': dict of {'function': scope.function, 'function_tuned_args': list of str, 'out_param': str}.
                                With custom scope function optimizer returns the corresponding tuned arguments (input of scope function)
                                for each iteration, so final parameter 'out_param' needs to be re-evaluated with the same function.
                                Used only outside objective_function
        '''
        
        n_features = len(work_set.columns) - 1
        pipeline_key = ''
        cov_mat = np.ones((n_features, n_features)) * -99
        param_choice = {}
        param_formatting = {}
        scope_func_setting = {}
        if learner_name == 'elastic_net':
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier
            # for hp.choice parameters optimizer return the corresponding index and not value
            param_choice = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 25, 50, 100]}
            param_space = {'eta0': hp.loguniform('eta0', np.log(0.01), np.log(1)),
                          'l1_ratio': hp.quniform('l1_ratio', 0, 1.0, 0.1),
                          'alpha': hp.choice('alpha', param_choice['alpha'])}

        if learner_name == 'logistic_regression':
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
            # for hp.choice parameters optimizer return the corresponding index and not value
            param_choice = {'random_state': [666]}
            param_space = {  # fictitious parameter to be tuned in order to run code as for other learner
                'random_state': hp.choice('random_state', param_choice['random_state'])}
            optim_max_iter = 1

        if learner_name == 'polynomial_regression':
            # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html#sklearn.preprocessing.PolynomialFeatures
            pipeline_key = 'poly'
            # for hp.choice parameters optimizer return the corresponding index and not value
            param_choice = {'degree': [2, 3, 4, 5],
                            'interaction_only': [True, False]}
            param_space = {'degree': hp.choice('degree', param_choice['degree']),
                          'interaction_only': hp.choice('interaction_only', param_choice['interaction_only'])}

        if learner_name == 'perceptron_elastic_net':
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html#sklearn.linear_model.Perceptron
            # for hp.choice parameters optimizer return the corresponding index and not value
            param_choice = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 25, 50, 100]}
            param_space = {'eta0': hp.loguniform('eta0', np.log(0.01), np.log(1)),
                          'alpha': hp.choice('alpha', param_choice['alpha'])}

        if learner_name == 'passive_aggressive':
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html#sklearn.linear_model.PassiveAggressiveClassifier
            # for hp.choice parameters optimizer return the corresponding index and not value
            param_choice = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 25, 50, 100],
                            'loss': ['hinge', 'squared_hinge']}
            param_space = {'loss': hp.choice('loss', param_choice['loss']),
                          'C': hp.choice('C', param_choice['C'])}

        if learner_name == 'SVM':
            # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
            # for hp.choice parameters optimizer return the corresponding index and not value
            param_choice = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 25, 50, 100],
                            'degree': [2, 3, 4, 5],
                            'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1,
                                      1 / n_features, # 'auto'
                                      1 / (n_features * work_set.drop(columns='true').unstack().var())], # 'scale'
                            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
            param_space = {'C': hp.choice('C', param_choice['C']),
                          'degree': hp.choice('degree', param_choice['degree']),
                          'gamma': hp.choice('gamma', param_choice['gamma']),
                          'kernel': hp.choice('kernel', param_choice['kernel'])}

        if learner_name == 'KNeighbors':
            # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
            # for hp.choice parameters optimizer return the corresponding index and not value
            param_choice = {'weights': ['uniform', 'distance'],
                            'metric': ['euclidean', 'manhattan', 'mahalanobis']}
            param_formatting = {'n_neighbors': int}
            param_space = {'n_neighbors': hp.quniform('n_neighbors', 1, round(len(work_set) / 4), 1),
                          'weights': hp.choice('weights', param_choice['weights']),
                          'metric': hp.choice('metric', param_choice['metric'])}
            cov_mat = work_set.drop(columns = 'true').corr().to_numpy()

        if learner_name == 'RandomForest':
            # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#
            # for hp.choice parameters optimizer return the corresponding index and not value
            param_formatting = {'min_samples_split': int,
                                'n_estimators': int}
            param_space = {'n_estimators': hp.loguniform('n_estimators', np.log(10), np.log(1000)),
                          'min_samples_split': scope.int(hp.loguniform('min_samples_split', np.log(2), np.log(50))),
                          'max_features': hp.loguniform('max_features', np.log(0.1), np.log(1))}
        if learner_name == 'GBM':
            # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
            # for hp.choice parameters optimizer return the corresponding index and not value
            param_formatting = {'min_samples_split': int,
                                'n_estimators': int}
            param_space = {'n_estimators': hp.loguniform('n_estimators', np.log(10), np.log(1000)),
                          'min_samples_split': scope.int(hp.loguniform('min_samples_split', np.log(2), np.log(50))),
                          'max_features': hp.loguniform('max_features', np.log(0.1), np.log(1)),
                          'subsample': hp.loguniform('subsample', np.log(0.1), np.log(1))}
        
        if learner_name == 'MLP':
            # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
            # for hp.choice parameters optimizer return the corresponding index and not value
            param_choice = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 25, 50, 100],
                            'activation': ['identity', 'logistic', 'tanh', 'relu'],
                            'batch_size': [16, 32, 64]}
            scope_func_setting = {'function': scope.hidden_layer,
                                  'function_tuned_args': ['n_layers', 'max_neuron_per_layer', 'seed'],
                                  'out_param': 'hidden_layer_sizes'}
            param_space = {'alpha': hp.choice('alpha', param_choice['alpha']),
                          'hidden_layer_sizes': scope.hidden_layer(hp.loguniform('n_layers', np.log(1), np.log(3)),
                                                            hp.loguniform('max_neuron_per_layer', np.log(20), np.log(200)),
                                                            hp.choice('seed', np.arange(2000))),
                          'activation': hp.choice('activation', param_choice['activation']),
                          'batch_size': hp.choice('batch_size', param_choice['batch_size'])}

        print('\n --- Training '+learner_name+' ... ', end='')
        
        # set input for objective function
        add_dict = {'work_set': work_set,
                    'param_formatting': param_formatting,
                    'learner_name': learner_name,
                    'pipeline_key': pipeline_key,
                    'cov_mat': cov_mat,
                    'perf_metric': perf_metric,
                    'n_splits': n_splits,
                    'binary_thresh': binary_thresh,
                    'multi_class': multi_class,
                    'multi_class_label': multi_class_label,
                    'multi_class_mode': multi_class_mode,
                    'model_ID': model_ID}

        # adapt function to allow add_dict as additional argument
        fmin_objective = partial(Stacking.objective_function, add_dict = add_dict)

        # optimize
        run_opt = True
        opt_algo = atpe.suggest
        while run_opt:
            try:    # try first with ATPE
                trials = Trials()
                start = timer()
                best_param_dict = fmin(fn=fmin_objective, 
                                  space=param_space, 
                                  algo=opt_algo, 
                                  trials=trials,
                                  max_evals = optim_max_iter,
                                  rstate= np.random.RandomState(123),
                                  verbose = False)
                run_opt = False
            except Exception as e:    # if fails, try TPE
                print('\n     ######## ATPE failed. Elapsed: '+str(datetime.timedelta(seconds=round(timer()-start))))
                print('     Error:', e)
                print('\n     Trying with TPE ... ', end='')
                opt_algo = tpe.suggest

        # get results
        results_df = pd.concat([pd.concat([pd.DataFrame(v['misc']['vals'], index=[i]) for i, v in enumerate(trials)]),
                         pd.concat([pd.DataFrame(v, index=[i]) for i, v in enumerate(trials.results)]).rename(columns={'loss': 'fmin_loss'})],
                               axis = 1)
        learner = results_df['raw_learner'][0]
        results_df = results_df.drop(columns='raw_learner')

        # apply custom scope function (if any) to get final corresponding parameter
        if len(scope_func_setting) > 0:
            def scope_fun(**x): return hyperopt.pyll.stochastic.sample(scope_func_setting['function'](**x))
            scope_param = scope_func_setting['out_param']
            scope_input = scope_func_setting['function_tuned_args']
            scope_kwargs = {k: best_param_dict[k] for k in scope_input}
            best_param_dict = {k: v for k, v in best_param_dict.items() if k not in scope_input}
            best_param_dict[scope_param] = hyperopt.pyll.stochastic.sample(scope_fun(**scope_kwargs))  # evaluate final parameter
            results_df.insert(int(np.where(results_df.columns == 'fmin_loss')[0]), scope_param,
                              results_df.apply(lambda row: scope_fun(**row[scope_input]), axis=1))
            results_df = results_df.drop(columns = scope_input)

        # change parameters format if needed
        for k, v in param_formatting.items():
            best_param_dict[k] = v(best_param_dict[k])
            results_df[k] = results_df[k].astype(v)

        if len(param_choice) > 0:   # convert np.choice index to corresponding value (if any)
            for k in param_choice.keys():
                best_param_dict[k] = param_choice[k][best_param_dict[k]]   # best_param_dict[k] contains the index
                results_df[k] = np.array(param_choice[k])[results_df[k]]

        results_df = results_df.reset_index().rename(columns={'index': 'iter'})
        results_df[perf_metric['name']] = results_df.fmin_loss if perf_metric['minimize'] else -results_df.fmin_loss
        best_param_results_df = results_df.sort_values(by=['fmin_loss'], ascending=True).iloc[0] # fmin_loss is always minimized
        results_df = results_df.drop(columns='binary_thresh_tuning')
        results_df.insert(0, 'best', np.where(results_df.index == best_param_results_df.iter, 'x', ''))
        best_binary_thresh = best_param_results_df.best_binary_thresh

        # train the learner with best parameters with Cross-Validation and save each fold model for prediction mode
        best_learner = copy.deepcopy(learner)
        _ = [setattr(best_learner, k, v) for k, v in best_param_dict.items()]
        final_split_models = Stacking.cross_validated_perf(learner = best_learner, work_set = work_set, perf_metric = perf_metric,
                                                           n_splits = n_splits, random_seed = final_model_split_seed,
                                                           binary_thresh = best_binary_thresh,
                                                           multi_class=multi_class, multi_class_label=multi_class_label,
                                                           multi_class_mode = multi_class_mode, save_split_models = True)

        # get cross-validated prediction and evaluate label with best_binary_thresh
        final_pred = pd.DataFrame(columns=['pred', 'true'])
        for split_mod in final_split_models['split_models']:
            final_pred = final_pred.append(split_mod['valid_pred'])
        if sum(final_pred.sort_index().true == work_set.true) != len(work_set):
            raise ValueError('Final prediction true labels don\'t match input labels')
        final_dataset = work_set.merge(final_pred.drop(columns='true'), left_index=True, right_index=True)
        final_dataset = final_dataset.merge(evaluate_prediction(final_dataset, binary_thresh=best_binary_thresh,
                                                                multi_class=multi_class, multi_class_label=multi_class_label,
                                                                multi_class_mode=multi_class_mode).drop(columns='pred_perc'),
                                                    left_index=True, right_index=True)

        # save model and optimization results
        pickle_path = 'checkpoints/'+model_ID+'_final_'+learner_name+'.pickle'
        with open(pickle_path, 'wb') as handle:
                     pickle.dump(final_split_models, handle, protocol=pickle.HIGHEST_PROTOCOL)
        csv_path = os.path.join('results', model_ID, 'tuning_results', model_ID+'_'+learner_name+'_param_tuning.csv')
        results_df.to_csv(csv_path, index=False, sep=',')
        binary_thresh_path = os.path.join('results', model_ID, 'tuning_results', model_ID+'_'+learner_name+'_binary_thresh_tuning.csv')
        if len(best_param_results_df['binary_thresh_tuning']) > 0:
            best_param_results_df['binary_thresh_tuning'].to_csv(binary_thresh_path, index=False, sep=',')

        avg_fold_perf = round(np.mean(final_split_models['valid_perf']), 5)
        full_set_perf = round(perf_metric['metric'](final_dataset.true, final_dataset.pred_label), 5)
        pickle_size = str(round(os.path.getsize(pickle_path) / 2**20, 1))+' MB'
        failed_trials = sum(results_df.status == 'fail')
        if failed_trials == 0: os.remove(os.path.join('results', model_ID, 'error_'+learner_name+'.txt'))
        print('Done in '+str(datetime.timedelta(seconds=round(timer()-start)))+
              ('   #######   ('+str(failed_trials)+'/'+str(optim_max_iter)+' trials failed)' if failed_trials > 0 else ''))
        print('     '+perf_metric['name']+': '+str(avg_fold_perf)+' (validation folds average)  '+str(full_set_perf)+ ' (full set)')
        print('\n     '+'best model (for all split) saved in '+pickle_path+' ('+pickle_size+')')
        print('     '+'tuning log saved in '+csv_path)
        if len(best_param_results_df['binary_thresh_tuning']) > 0: print('     '+'binary threshold tuning log saved in '+binary_thresh_path)

        out = {'performance_row': pd.DataFrame({'learner': learner_name,
                                                'n_splits': n_splits,
                                                'perf_metric': perf_metric['name'],
                                                'avg_fold_perf': avg_fold_perf,
                                                'full_set_perf': full_set_perf,
                                                'best_binary_thresh': best_binary_thresh,
                                                'time': str(datetime.timedelta(seconds=round(timer()-start))),
                                                'size': pickle_size,
                                                'models_path': pickle_path}, index = [0]),
               'final_dataset': final_dataset}

        return out

    @staticmethod
    def _perform_stacking(final_dataset, model_ID_to_stack, model_ID, avail_label, multi_class, binary_thresh_1st_level,
                          multi_class_mode_1st_level, learner_2nd_level, n_splits_2nd_level, binary_thresh_2nd_level,
                          perf_metric, multi_class_mode_2nd_level, split_seed_2nd_level, optim_max_iter):

        work_set = copy.deepcopy(final_dataset)[model_ID_to_stack + ['true']]

        if learner_2nd_level in ['average', 'majority_vote_by_label', 'median']:
            out = Stacking.averaging_methods(final_dataset = work_set, average_method = learner_2nd_level,
                                             binary_thresh = binary_thresh_1st_level, multi_class = multi_class,
                                             multi_class_label = avail_label, multi_class_mode = multi_class_mode_1st_level,
                                             perf_metric = perf_metric, model_ID = model_ID)
        else:
            out = Stacking.tune_learner(work_set = work_set, learner_name = learner_2nd_level, perf_metric = perf_metric,
                                        n_splits = n_splits_2nd_level, optim_max_iter = optim_max_iter,
                                        binary_thresh = binary_thresh_2nd_level, multi_class = multi_class,
                                        multi_class_label = avail_label, multi_class_mode = multi_class_mode_2nd_level,
                                        model_ID = model_ID, final_model_split_seed = split_seed_2nd_level)

        return out