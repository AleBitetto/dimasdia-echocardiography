import math
import copy
import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import warnings
import re
from timeit import default_timer as timer
import datetime
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
import seaborn as sns
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from dataloaders import SubsetRandomDataLoader
from metrics import Accuracy
from loss.focal_loss import *
from metrics.confusion_matrix import *
from metrics.plot_functions import *
from pynvml import *


def get_memory_usage():
    
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
#     print(f'total    : {info.total/ 2**20}')
#     print(f'free     : {info.free/ 2**20}')
#     print(f'used     : {info.used/ 2**20}')
    return round(info.used/ 2**20)

def set_seed(SEED = 6666):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Evaluator:
    def __init__(self, mode = 'train', loss_add_function=None, multi_class=False, **kwargs):
        '''
        Args:
            - mode: 'train', 'predict' or 'feat_vis' for feature visualization
            - loss_add_function: additional function to be applied on prediction ONLY before evaluating the loss
            - multi_class: True for multi class task
            - **kwargs:
                  pass "dataset", "set_idx" and "batch_size" if you want to instantiate SubsetRandomDataLoader
                  pass "set_loader" if already instatiated            
        '''
        
        self.multi_class = multi_class
        self.loss_add_function = loss_add_function
        if mode not in ('train', 'predict', 'feat_vis'):
            raise ValueError('Please provide mode as \'train\' or \'predict\' or \'feat_vis\' - current is: ' + mode)
        else:
            self.mode = mode
        dataset = kwargs.get('dataset')
        set_idx = kwargs.get('set_idx')
        batch_size = kwargs.get('batch_size')
        set_loader = kwargs.get('set_loader')
        if (dataset is not None) and (set_idx is not None) and (batch_size is not None):
            self.set_loader = SubsetRandomDataLoader(dataset, set_idx, batch_size)
        elif set_loader:
            self.set_loader = set_loader
        else:
            raise ValueError('Please provide set_loader or (dataset, batch_size, set_idx)')
      
    def train_predict(self, model, loss_criterion=None, metrics=None, optimizer=None, scheduler=None,
                      show_log=None, start=None, it=None, it_save=None, it_log=None, it_per_epoch=None,
                      trainer_ID=None, save_checkpoint=True):
        '''
        Args:
            - model: model to be trained/evaluated
            - loss_criterion: if not None, losses is evaluated and returned
            - metrics: if not None, metrics are evaluated and returned
            if self.mode == 'train'
                - optimizer, scheduler, show_log, it, it_save, it_log, it_per_epoch, trainer_ID: 
                - save_checkpoint: if True save checkpoints
            
        Return:
            - out: dictionary of
                preds: index, predicted and true values (pd.Dataframe)
                    + loss if loss_criterion not None
                    + eval_metrics if metrics not None
                    + optimizer, scheduler, it, epoch, lr if mode == 'train'
        '''
        
        if self.mode == 'train':
            model.train()
        elif self.mode == 'predict' or self.mode == 'feat_vis':
            model.eval()
            
        set_num = 0
        set_labels = []
        set_preds = []
        set_preds_raw = [] # used only for features visualisation
        set_indices = []
        set_images = []
        losses = []
        loss_weights = []
        out = {}

        for inputs, labels, indices in self.set_loader:
            inputs = inputs.cuda().float()
            if self.multi_class:
                labels = labels.cuda().long().view(-1)
            else:
                labels = labels.cuda().float()
            indices = indices.cuda().float()
            set_num += len(inputs)
            
            # if train
            if self.mode == 'train':
                epoch = (it + 1) / it_per_epoch
                
                lr = scheduler.get_last_lr()[0]

                # checkpoint
                if it % it_save == 0 and it != 0 and save_checkpoint:
                    save(model, optimizer, it, epoch, trainer_ID)
  
                preds = model(inputs)
                if self.loss_add_function is not None:
                    loss = loss_criterion(self.loss_add_function(preds), labels)
                else:
                    loss = loss_criterion(preds, labels)
                
                set_labels.append(labels.cpu())
                set_preds.append(preds.cpu())
                set_indices.append(indices.cpu())
                losses.append(loss.data.cpu().numpy())
                loss_weights.append(len(preds))

                optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                if it % it_log == 0 and show_log == 1:
                    print(
                        "{:5f} {:4d} {:5.1f} |                      |                      | {:6.2f}".format(
                            lr, it, epoch, timer() - start
                        ))

                it += 1
                out['scheduler'] = scheduler   # todo: capisci se servono davvero
                out['optimizer'] = optimizer
                out['it'] = it
                out['epoch'] = epoch
                out['lr'] = lr
                
            # if predict
            if self.mode == 'predict':
                with torch.no_grad():
                    preds = model(inputs)
                    if loss_criterion:
                        loss = loss_criterion(preds, labels)
                        if self.loss_add_function is not None:
                            loss = loss_criterion(self.loss_add_function(preds), labels)
                        else:
                            loss = loss_criterion(preds, labels)
                        losses.append(loss.data.cpu().numpy())
                
                set_labels.append(labels.cpu())
                set_preds.append(preds.cpu())
                set_indices.append(indices.cpu())
                loss_weights.append(len(preds))
                
            # if feature visualization
            if self.mode == 'feat_vis':
                set_images.append(inputs)
                set_indices.append(indices.cpu())
                set_labels.append(labels.cpu())
                preds = model(inputs)  # grad is required for backpropagation
                set_preds_raw.append(preds)
                set_preds.append(preds.cpu())
                
        if set_num != len(self.set_loader.sampler):
            raise ValueError('Total number of batch evaluation is different from set length when: ', self.mode)

        # evaluate loss
        if loss_criterion is not None:
            loss = np.average(losses, weights=loss_weights)
            out['loss'] = loss

        # evaluate metrics
        if metrics is not None:
            with torch.no_grad():
                set_labels = torch.cat(set_labels)
                set_preds = torch.cat(set_preds)
                set_indices = torch.cat(set_indices)
                out['metrics'] = [i(set_labels.cpu(), set_preds.cpu()).item() for i in metrics]
        else:
            set_labels = torch.cat(set_labels)
            set_preds = torch.cat(set_preds)
            set_indices = torch.cat(set_indices)
#             set_images = torch.cat(set_images)
        
        # save output
        if self.multi_class:
            out_preds = pd.DataFrame({'pred': [arr for arr in set_preds.detach().numpy()],
                                      'true': set_labels.numpy().ravel(),
                                      'index': set_indices.numpy().ravel().astype(int)}).set_index('index')
        else:
            out_preds = pd.DataFrame({'pred': set_preds.detach().numpy().ravel(),
                                      'true': set_labels.numpy().ravel(),
                                      'index': set_indices.numpy().ravel().astype(int)}).set_index('index')
        
        if self.mode != 'feat_vis':
            out['preds'] = out_preds
        else:
            out['preds'] = {'img': set_images,
                            'pred_list': set_preds_raw,#torch.cat(set_preds_raw),
                            'pred_table': out_preds}
        
        inputs.cpu()
        labels.cpu()
        indices.cpu()
        del inputs, labels, indices
            
        return out

class Trainer:
    def __init__(self, classifier, dataset, batch_size, train_idx, validation_idx, loss_criterion, metrics,
                 loss_add_function=None, multi_class=False, invert_class=False, trainer_ID='train', show_info=True):
        '''
        Args:
            - classifier: model to trained
            - dataset: dataset to be used (dataloader)
            - batch_size: batch size for training
            - train_idx, validation_idx: index for train and validation set
            - loss_criterion: loss to be used in training
            - metrics: metrics to be used for performances evaluation
            - loss_add_function: additional transformation to be applied before evaluating loss. E.g. for multiclass problem, if final layer is
                                softmax, loss_criterion = nn.NLLLoss() and loss_add_function = torch.log in order to get LogSoftmax loss and
                                leaving the model ready to predict without any final transformation
            - multi_class: True for multi class task
            - invert_class: if multi_class=False, invert (1-p) predicted and true values, train and then convert back to p
                            Original classes are already inverted in the provided dataset.
            - trainer_ID: ID to define current setting. Used as prefix for saved images, model and output
            - show_info: if True print model and dataset info
        '''
        
#         set_seed()
        # invert class 1 <-> 0
        if multi_class: invert_class = False
        if invert_class: dataset.df.label = np.where(dataset.df.label == 0, 1, 0)
        
        self.classifier = classifier
        self.model_ID = classifier.__class__.__name__

        self.loss_criterion = loss_criterion.cuda()
        self.metrics = metrics
        
        train_loader = SubsetRandomDataLoader(dataset, train_idx, batch_size)
        validation_loader = SubsetRandomDataLoader(dataset, validation_idx, batch_size)
        self.train_evaluator = Evaluator(mode='train', loss_add_function=loss_add_function, multi_class=multi_class,
                                         set_loader=train_loader)
        self.train_evaluator_pred = Evaluator(mode='predict', loss_add_function=loss_add_function, multi_class=multi_class,
                                              set_loader=train_loader)
        self.valid_evaluator = Evaluator(mode='predict', loss_add_function=loss_add_function, multi_class=multi_class,
                                         set_loader=validation_loader)

        self.it_per_epoch = math.ceil(len(train_idx) / batch_size)
        self.multi_class = multi_class
        self.invert_class = invert_class
        self.options = {'invert_class': invert_class,    # store model options
                        'multi_class': multi_class,
                        'loss_function': loss_criterion,
                        'loss_add_function': loss_add_function}
        self.train_idx = train_idx
        self.validation_idx = validation_idx
        self.trainer_ID = trainer_ID
        
        if show_info:
            print('Trainer started with classifier: {}\ndataset: {}\nbatch size: {}'.format(classifier.__class__.__name__, dataset, batch_size))
            print('Train set: {}'.format(len(train_idx)))
            print('Validation set: {}'.format(len(validation_idx)))
            print('Training with {} mini-batches per epoch'.format(self.it_per_epoch))
        
    def run(self, max_epochs=10, lr=0.01, show_info=True, show_log=1, save_checkpoint=True, save_final=True):
        '''
        Args:
            - max_epochs: max epochs
            - lr: learning rate
            - show_info: if True print model and dataset info
            - show_log: 1 log at every batch, 2 log at every epoch, 3 off
            - save_checkpoint: if True save checkpoints
            - save_final: if True save final model
            
        Return:
            - out: dictionary of
                train_history: history log of training performances
        '''
        self.classifier = self.classifier.cuda()
        model = self.classifier

        it = 0
        epoch = 0
        it_save = self.it_per_epoch * 5
        it_log = math.ceil(self.it_per_epoch / 5)
        it_smooth = self.it_per_epoch

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 2 * self.it_per_epoch, gamma=0.9)

        if show_info:
            print("Logging performance every {} iter, smoothing every: {} iter, saving checkpoints every {} iter".format(it_log, it_smooth, it_save))
            print("{}'".format(optimizer))
            print("{}'".format(scheduler))
            print("{}'".format(self.loss_criterion))
            print("{}'".format(self.metrics))
        if show_log < 3:
            print('\n* performance with model.train()\n# performance with model.eval()')
            print('                    |         VALID        |        TRAIN         |         ')
            print(' lr     iter  epoch | loss    roc    acc   | loss    roc    acc   |  time   ')
            print('------------------------------------------------------------------------------')

        start = timer()
        train_history = pd.DataFrame(columns = ['lr', 'iter', 'epoc', 'Valid_loss', 'Valid_roc', 'Valid_acc', 'Train_loss', 'Train_roc', 'Train_acc', 'time'],
                                 dtype=float).fillna('')
        while epoch < max_epochs:
            # train
            train_eval = self.train_evaluator.train_predict(model=model, loss_criterion=self.loss_criterion, metrics=self.metrics,
                                                            optimizer=optimizer, scheduler=scheduler, show_log=show_log, start=start,
                                                            it=it, it_save=it_save, it_log=it_log, it_per_epoch=self.it_per_epoch,
                                                           trainer_ID=self.trainer_ID, save_checkpoint=save_checkpoint)

            train_preds_training = train_eval['preds']
            optimizer = train_eval['optimizer']   # should be unnecessary
            scheduler = train_eval['scheduler']   # should be unnecessary
            it = train_eval['it']
            epoch = train_eval['epoch']
            lr = train_eval['lr']
            train_loss_training = train_eval['loss']
            train_m_training = train_eval['metrics']
            train_acc_training, train_roc_training = train_m_training
            # predict the training set without batchnorm or regularitazion
            train_pred_eval = self.train_evaluator_pred.train_predict(model, loss_criterion=self.loss_criterion, metrics=self.metrics)
            train_preds = train_pred_eval['preds']
            train_loss = train_pred_eval['loss']
            train_m = train_pred_eval['metrics']
            train_acc, train_roc = train_m

            # validation
            valid_eval = self.valid_evaluator.train_predict(model, loss_criterion=self.loss_criterion, metrics=self.metrics)
            valid_preds = valid_eval['preds']
            valid_loss = valid_eval['loss']
            valid_m = valid_eval['metrics']
            valid_acc, valid_roc = valid_m
            
            if show_log <= 2:
                print(
                    "{:5f} {:4d} {:5.1f} | {:0.3f}# {:0.3f}  {:0.3f}  | {:0.3f}* {:0.3f}  {:0.3f}  | {:6.2f}".format(
                        lr, it, epoch, valid_loss, valid_roc, valid_acc, train_loss_training, train_roc_training, train_acc_training, timer() - start
                    ))
                print(
                    "                    |                      | {:0.3f}# {:0.3f}  {:0.3f}  | {:6.2f}".format(
                       train_loss, train_roc, train_acc, timer() - start
                    ))
            
            train_history=train_history.append(
                pd.DataFrame(np.array([lr, it, epoch, valid_loss, valid_roc, valid_acc,
                                       train_loss, train_roc, train_acc, timer() - start]).reshape(1,-1),
                             columns = train_history.columns)
                )

        # Save and return
        train_out = {'trainer_ID': self.trainer_ID,
                     'model_ID': self.model_ID,
                     'train_history': train_history,
                     'train_preds_training': 1 - train_preds_training if self.invert_class else train_preds_training,
                     'train_preds': 1 - train_preds if self.invert_class else train_preds,
                     'train_idx': self.train_idx,
                     'valid_preds': 1 - valid_preds if self.invert_class else valid_preds,
                     'valid_idx': self.validation_idx,
                     'options': self.options}
        
        if save_final:
            save(model, optimizer, 'final', epoch, self.trainer_ID, train_out, show_info)
        
        self.classifier.cpu()
        self.loss_criterion.cpu()
        del self.classifier, self.loss_criterion, self.metrics, self.train_evaluator, self.train_evaluator_pred, self.valid_evaluator
        torch.cuda.empty_cache()
        
        return train_out
    
def save(model, optimizer, iter, epoch, trainer_ID, train_out=None, show_info=True):
    '''
    https://pytorch.org/tutorials/beginner/saving_loading_models.html
    '''
    # save model for inference
    torch.save({'model': model,
                'model_state_dict': model.state_dict(),
                'train_out': train_out
               }, "checkpoints/{}_{}_model.pth".format(trainer_ID, iter))
    if iter == 'final':
        if show_info: print('\nModel saved in', "checkpoints/{}_{}_model.pth".format(trainer_ID, iter))
        # used only to export weights for inference, model needs to be imported apart (using settings)
        torch.save({'state_dict': model.state_dict(),
                    'settings': model.model_settings
                   },"checkpoints/{}_{}_state_dict.pth".format(trainer_ID, iter))
    # save general checkpoint for resuming training
    torch.save({
        "optimizer": optimizer.state_dict(),
        "iter": iter,
        "epoch": epoch
    }, "checkpoints/{}_{}_optimizer.pth".format(trainer_ID, iter))
    

class CrossValidationUtils:
    
    @staticmethod    
    def train_predict_CV(mode='train', dataset=None, base_classifier=None, base_classifier_train_out=None, n_splits=5, n_splits_full=5,
                         stratify_lab=None, max_epochs_split=10, max_epochs_full=10, loss_fun='', metrics=[],
                         loss_add_function=None, multi_class=False, multi_class_mode='max', binary_thresh=0.5, invert_class=False,
                         measure_to_plot=[], conf_mat_meas=['f1-score'], batch_size=64, show_info=False, show_log=2, show_perf_plot=False,
                         show_distr_plot=False, save_split_checkpoint=False, save_split_final=True, save_full_checkpoint=False, save_full_final=True,
                         reload_split=False, reload_full=False, trainer_ID='', silent=False, cv_for_full_model_only=False, split_seed=42):
        
        '''
        Train model with Cross-Validation and then train model on full dataset. Use in wrapper tune_with_CV().

        Arg:
            mode: 'train' or 'predict'. Train starting from base classifier or predict on folds and full dataset with base_classifier
            dataset: dataset used
            base_classifier: pretrained classifier to be used for transfer learning or to predict only
            base_classifier_train_out: train out from base_classifier. Used to take column name of train_history
            n_splits: number of folds for Cross-Validation
            n_splits_full: number of folds to predict confusion matrix on full dataset model
            stratify_lab: array to be used for stratified CV
            max_epochs_split: max epoch for each fold
            max_epochs_full: max epoch for model trained on full dataset
            loss_fun: loss function to be used for train. 'BCE' or 'NLL' or 'FocLos' available
            metrics: metrics to be evaluated for train
            loss_add_function: additional transformation to be applied before evaluating loss. E.g. for multiclass problem, if final layer is
                               softmax, loss_criterion = nn.NLLLoss() and loss_add_function = torch.log in order to get LogSoftmax loss and
                               leaving the model ready to predict without any final transformation
            multi_class: True for multi class task
            multi_class_mode: how to set predicted class given the probabilities array. 'max': takes highest probability
            binary_thresh: threshold for binary classification on predicted values
            invert_class: if True the binary label will be inverted when training. Only if multi_class=False
            measure_to_plot: measure to be plotted for performance evolution over epochs. Both CV and full model are provided
            conf_mat_meas: list of measures to be logged for each class, taken from confusion matrix report
            batch_size: batch size for models
            show_info, show_log: see trainer.Trainer.run
            show_perf_plot: if True performance evolutions are plotted
            show_distr_plot: if True predicted probabilities distribution are plotted
            save_split_checkpoint: save model checkpoints for CV models
            save_split_final: save final model for CV models
            save_full_checkpoint: save model checkpoints for full dataset model
            save_full_final: save final model for full dataset model
            reload_split: reload each CV model
            reload_full: reload full dataset model
            trainer_ID: ID to define current setting. Used as prefix for saved images, model and output
            silent: if True suppress all print, display, plot
            cv_for_full_model_only: if True, add confusion matrix predicting full model on splits (with a different seed).
                                    Basically, suppress everything but prediction on splits and confusion matrix
            split_seed: seed for stratified sampling. Changed in cv_for_full_model_only == True

         Return:
             dictionary of:
                 - session_ID
                 - split_log: log for each fold
                 - final_performance: summary for CV and full model
        '''

        main_folder = trainer_ID
        base_classifier_ID = base_classifier_train_out['model_ID']
        # set loss function
        if loss_fun == 'BCE':
            loss_criterion = nn.BCELoss()
        elif loss_fun == 'NLL':
            loss_criterion = nn.NLLLoss()
        elif loss_fun == 'FocLos':
            loss_criterion = FocalLoss(eval_logits=False, multi_class=multi_class)
        elif loss_fun == '':
            pass
        else:
            raise ValueError('Only \'BCE\' or \'NLL\' or \'FocLos\' allowed - current is: ' + loss_fun)
        session_ID = trainer_ID + '_' + base_classifier_ID + '_' + loss_fun

        if mode == 'train' and cv_for_full_model_only:
            raise ValueError('cv_for_full_model_only==True available only with mode==\'train\'')
        
        avail_class = sorted(dataset.df.meaning.unique().tolist())
        conf_mat_log_cols = [x+'_'+y for y in conf_mat_meas for x in avail_class]
        conf_mat_log_cols = [['train_'+x, 'valid_'+x] for x in conf_mat_log_cols]
        conf_mat_log_cols = conf_mat_log_cols + [['train_MCC', 'valid_MCC']]
        conf_mat_log_cols=np.array(conf_mat_log_cols).ravel()

        if multi_class: invert_class = False
        if invert_class:
            session_ID = session_ID + '_inv_class'
        
        if silent:
            show_info = False
            show_log = 3
            show_perf_plot = False
            show_distr_plot = False
        
        
        ##########################################################
        
        #######      train on Cross-Validated dataset      #######
        
        ##########################################################
        
        
        if mode == 'train' and n_splits > 1 and silent == False: print('\nExecuting a {}-fold cross validation'.format(n_splits))
        split_log = []
        split_train_distr=[]
        split_valid_distr=[]
        if n_splits > 1:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
            # display distribution for each split
            split=1
            for train_idx, validation_idx in skf.split(dataset.df, stratify_lab):
                train_d=dataset.df.iloc[train_idx].groupby(['label', 'meaning', 'source']).size()\
                                        .to_frame().add_prefix('X_').rename(columns={'X_0': 'train_'+str(split)})
                valid_d=dataset.df.iloc[validation_idx].groupby(['label', 'meaning', 'source']).size()\
                                        .to_frame().add_prefix('X_').rename(columns={'X_0': 'valid_'+str(split)})
                if len(split_train_distr) == 0:
                    split_train_distr = train_d
                    split_valid_distr = valid_d
                else:
                    split_train_distr = pd.concat([split_train_distr, train_d], axis=1, join='outer')
                    split_valid_distr = pd.concat([split_valid_distr, valid_d], axis=1, join='outer')
                split +=1
            if cv_for_full_model_only == False and silent == False:
                print('\nTrain distribution for each split:')
                display(split_train_distr)
                print('\nValidation distribution for each split:')
                display(split_valid_distr)
            split_log=pd.DataFrame(columns = np.concatenate((['Split'], base_classifier_train_out['train_history'].columns.values,
                                                             conf_mat_log_cols)), dtype=float).fillna('')
            split = 1
            conf_mat_log = {}
            split_prediction_out = {}
            # run training for each fold
            for train_idx, validation_idx in skf.split(dataset.df, stratify_lab):
                if mode == 'train' and silent == False: print('\n=== Split #{} ===\n'.format(split))
                split_ID = session_ID+'_split'+str(split)+('FF' if cv_for_full_model_only else '')
                # Start from base classifier
                classifier = copy.deepcopy(base_classifier)
                reload_path = 'checkpoints/'+session_ID+'_split'+str(split)+'_final_model.pth'
                run_flag=True
                if reload_split:
                    if os.path.exists(reload_path):
                        if silent == False: print('-- Reloading', reload_path)
                        reload = torch.load(reload_path)
                        split_train_out = reload['train_out']
                        run_flag=False
                    else:
                        if silent == False: print('--', reload_path, 'Not found. Proceeding with training\n')
                if run_flag:
                    if mode == 'train':
                        split_trainer = Trainer(classifier, copy.deepcopy(dataset), batch_size, train_idx, validation_idx,
                                                loss_criterion=loss_criterion, metrics=metrics, loss_add_function=loss_add_function,
                                                multi_class=multi_class, invert_class=invert_class, trainer_ID=split_ID, show_info=show_info)
                        split_train_out = split_trainer.run(max_epochs=max_epochs_split, show_info=show_info, show_log=show_log,
                                                            save_checkpoint=save_split_checkpoint, save_final=save_split_final)
                        # add predicted label and percentage
                        split_train_out['train_preds'][['pred_class', 'pred_perc']] = evaluate_prediction(set_df=split_train_out['train_preds'],
                                                                                        binary_thresh=binary_thresh, multi_class=multi_class,
                                                                                        multi_class_label=sorted(dataset.df.label.unique()),
                                                                                        multi_class_mode=multi_class_mode)
                        split_train_out['valid_preds'][['pred_class', 'pred_perc']] = evaluate_prediction(set_df=split_train_out['valid_preds'],
                                                                                        binary_thresh=binary_thresh, multi_class=multi_class,
                                                                                        multi_class_label=sorted(dataset.df.label.unique()),
                                                                                        multi_class_mode=multi_class_mode)
                    elif mode == 'predict':
                        train_pred = CrossValidationUtils.predict_dataset(model_ID=classifier, dataset=copy.deepcopy(dataset),
                                                                          new_index=np.array(train_idx), batch_size=batch_size,
                                                                          binary_thresh=binary_thresh, multi_class=multi_class,
                                                                          multi_class_mode=multi_class_mode, invert_class=invert_class,
                                                                          save_path='', add_title='- Train', save_ID='',
                                                                          show_result=False)
                        valid_pred = CrossValidationUtils.predict_dataset(model_ID=classifier, dataset=copy.deepcopy(dataset),
                                                                          new_index=np.array(validation_idx), batch_size=batch_size,
                                                                          binary_thresh=binary_thresh, multi_class=multi_class,
                                                                          multi_class_mode=multi_class_mode, invert_class=invert_class,
                                                                          save_path='', add_title='- Validation', save_ID='',
                                                                          show_result=False)
                        split_train_out = {'train_preds': train_pred['prediction'],
                                           'valid_preds': valid_pred['prediction'],
                                           'train_history': base_classifier_train_out['train_history']}
                split_prediction_out['split_'+str(split)] = split_train_out
                # save confusion matrix
                split_conf_mat=evaluate_confusion_matrix(original_dataset=dataset.df, train_out=split_train_out,
                                                         binary_thresh=binary_thresh, save_path=main_folder,
                                                         save_ID=split_ID, multi_class=multi_class, multi_class_mode=multi_class_mode,
                                                         show_result=False, add_title=session_ID+' - Split '+str(split)+'\n')          
                conf_mat_log['split_'+str(split)] = {k: v for k, v in split_conf_mat.items() if k in ['train_conf_mat', 'valid_conf_mat',
                                                                                                     'train_mcc', 'valid_mcc']}
                conf_mat_report = []
                for cfm in conf_mat_meas:
                    for cl in avail_class:
                        conf_mat_report += [split_conf_mat['train_class_report'][cl][cfm],
                                            split_conf_mat['valid_class_report'][cl][cfm]]
                conf_mat_report = pd.DataFrame([conf_mat_report+[split_conf_mat['train_mcc'], split_conf_mat['valid_mcc']]], columns = conf_mat_log_cols)
                # update log
                split_log = split_log.append(
                    pd.DataFrame(np.hstack((np.repeat(split, split_train_out['train_history'].shape[0]).reshape(-1,1),
                                            split_train_out['train_history'], pd.concat([conf_mat_report]*split_train_out['train_history'].shape[0]))),
                                 columns=split_log.columns)
                )
                split += 1

            # evaluate and plot average confusion matrix of all splits
            
            target_names_opt = split_conf_mat['options']['train_opt']['target_names'] # takes from last split - always the same
            norm_opt = split_conf_mat['options']['train_opt']['normalize']
            conf_train = [v['train_conf_mat'] for k, v in conf_mat_log.items()]
            mcc_train = [v['train_mcc'] for k, v in conf_mat_log.items()]
            conf_valid = [v['valid_conf_mat'] for k, v in conf_mat_log.items()]
            mcc_valid = [v['valid_mcc'] for k, v in conf_mat_log.items()]
            conf_train_split = render_conf_mat_boxplot(conf_mat_list=conf_train, mcc_list=mcc_train, target_names=target_names_opt,
                                                        title='Train', cmap=plt.cm.Greens, fig_size=5)
            plt.close(conf_train_split)
            conf_valid_split = render_conf_mat_boxplot(conf_mat_list=conf_valid, mcc_list=mcc_valid, target_names=target_names_opt,
                                                        title='Validation', cmap=plt.cm.Greens, fig_size=5)
            plt.close(conf_valid_split)
            train_path=os.path.join('results', main_folder, 'avg_split_conf_mat_train.png')
            valid_path=os.path.join('results', main_folder, 'avg_split_conf_mat_valid.png')
            if mode == 'train':
                merge_path=os.path.join('results', main_folder, 'avg_split_conf_mat.png')
            if mode == 'predict' and cv_for_full_model_only:
                merge_path=os.path.join('results', main_folder, 'avg_split_conf_mat_full.png')
            conf_train_split.savefig(train_path, bbox_inches="tight", pad_inches=0.5)
            conf_valid_split.savefig(valid_path, bbox_inches="tight", pad_inches=0.5)
            get_concat_h([Image.open(train_path), Image.open(valid_path)],
                        add_main_title=session_ID+' - '+str(n_splits)+' Split Average'+
                        (' (different seed) on full dataset model' if cv_for_full_model_only else ''),
                        font_size=20).save(merge_path)
            os.remove(train_path)
            os.remove(valid_path)
            if silent == False:
                display(Image.open(merge_path))

            # evaluate average performance on splits
            split_log[['Split', 'iter', 'epoc', 'time']] = split_log[['Split', 'iter', 'epoc', 'time']].astype(int)
            split_performance = split_log[split_log.epoc == split_log.epoc.max()].groupby('epoc').agg(
                Valid_Loss_Avg=("Valid_loss", lambda x: str(round(x.mean(), 3))+' ± '+str(round(np.std(x), 3))),
                Valid_ROC_Avg=("Valid_roc", lambda x: str(round(x.mean(), 3))+' ± '+str(round(np.std(x), 3))),
                Valid_Acc_Avg=("Valid_acc", lambda x: str(round(x.mean(), 3))+' ± '+str(round(np.std(x), 3))),
                Train_Loss_Avg=("Train_loss", lambda x: str(round(x.mean(), 3))+' ± '+str(round(np.std(x), 3))),
                Train_ROC_Avg=("Train_roc", lambda x: str(round(x.mean(), 3))+' ± '+str(round(np.std(x), 3))),
                Train_Acc_Avg=("Train_acc", lambda x: str(round(x.mean(), 3))+' ± '+str(round(np.std(x), 3)))
                ).reset_index(drop=True)
            for col in conf_mat_log_cols:    # conf_mat_meas for each class, train/valid
                split_performance = pd.concat(
                    [split_performance,
                     split_log[split_log.epoc == split_log.epoc.max()].groupby('epoc').agg(
                            add_col=(col, lambda x: str(round(x.mean(), 3))+' ± '+str(round(np.std(x), 3)))
                     ).reset_index(drop=True).rename(columns={"add_col": col+'_Avg'})], axis=1)

            if mode == 'train' and silent == False:
                print('\n\nAverage Performance:')
                display(split_performance)



        ###############################################################
        
        #######      train on full dataset for final model      #######
        
        ###############################################################
        
        
        if mode == 'train' and silent == False: print('\n\n\n=== Full Dataset ===\n')
        full_distr = []
        classifier = copy.deepcopy(base_classifier)
        train_idx, validation_idx = train_test_split(
            list(range(len(dataset))),
            test_size=0.2,
            stratify=stratify_lab,
            shuffle=True,
            random_state=42
        )
        full_distr=pd.concat([dataset.df.iloc[train_idx].groupby(['label', 'meaning', 'source']).size()\
                                    .to_frame().add_prefix('X_').rename(columns={'X_0': 'train'}),
                              dataset.df.iloc[validation_idx].groupby(['label', 'meaning', 'source']).size()\
                                    .to_frame().add_prefix('X_').rename(columns={'X_0': 'validation'})],
                             axis=1, join='outer')
        if cv_for_full_model_only == False and silent == False:
            print('\nDistribution for full dataset:')
            display(full_distr)
        reload_path = 'checkpoints/'+session_ID+'_full_final_model.pth'
        run_flag=True
        if reload_full:
            if os.path.exists(reload_path):
                if silent == False: print('-- Reloading', reload_path)
                reload = torch.load(reload_path)
                train_out = reload['train_out']
                run_flag=False
            else:
                if silent == False: print('--', reload_path, 'Not found. Proceeding with training\n')
        if run_flag:
            if mode == 'train':
                trainer = Trainer(classifier, copy.deepcopy(dataset), batch_size, train_idx, validation_idx,
                                  loss_criterion=loss_criterion, metrics=metrics, loss_add_function=loss_add_function,
                                  multi_class=multi_class, invert_class=invert_class, trainer_ID=session_ID+'_full', show_info=show_info)
                train_out = trainer.run(max_epochs=max_epochs_full, show_info=show_info, show_log=show_log,
                                       save_checkpoint=save_full_checkpoint, save_final=save_full_final)
                # add predicted label and percentage
                train_out['train_preds'][['pred_class', 'pred_perc']] = evaluate_prediction(set_df=train_out['train_preds'],
                                                                                    binary_thresh=binary_thresh, multi_class=multi_class,
                                                                                    multi_class_label=sorted(dataset.df.label.unique()),
                                                                                    multi_class_mode=multi_class_mode)
                train_out['valid_preds'][['pred_class', 'pred_perc']] = evaluate_prediction(set_df=train_out['valid_preds'],
                                                                                    binary_thresh=binary_thresh, multi_class=multi_class,
                                                                                    multi_class_label=sorted(dataset.df.label.unique()),
                                                                                    multi_class_mode=multi_class_mode)
            elif mode == 'predict' and cv_for_full_model_only == False:
                train_pred = CrossValidationUtils.predict_dataset(model_ID=classifier, dataset=copy.deepcopy(dataset),
                                                                  new_index=np.array(train_idx), batch_size=batch_size,
                                                                  binary_thresh=binary_thresh, multi_class=multi_class,
                                                                  multi_class_mode=multi_class_mode, invert_class=invert_class,
                                                                  save_path='', add_title='- Train', save_ID='', show_result=False)
                valid_pred = CrossValidationUtils.predict_dataset(model_ID=classifier, dataset=copy.deepcopy(dataset),
                                                                  new_index=np.array(validation_idx), batch_size=batch_size,
                                                                  binary_thresh=binary_thresh, multi_class=multi_class,
                                                                  multi_class_mode=multi_class_mode, invert_class=invert_class,
                                                                  save_path='', add_title='- Validation', save_ID='', show_result=False)
                train_out = {'train_preds': train_pred['prediction'],
                             'valid_preds': valid_pred['prediction'],
                             'train_history': base_classifier_train_out['train_history']}
        full_prediction_out = train_out if cv_for_full_model_only == False else None
        
        # plot and save confusion matrix
        if cv_for_full_model_only == False:
            full_conf_mat=evaluate_confusion_matrix(original_dataset=dataset.df, train_out=train_out,
                                                    binary_thresh=binary_thresh, save_path=main_folder,
                                                    save_ID=session_ID+'_full', multi_class=multi_class, multi_class_mode=multi_class_mode,
                                                    show_result=not silent, add_title=session_ID+' - Full Dataset\n', cmap=plt.cm.Reds)
            conf_mat_report = []
            for cfm in conf_mat_meas:
                for cl in avail_class:
                    conf_mat_report += [full_conf_mat['train_class_report'][cl][cfm],
                                        full_conf_mat['valid_class_report'][cl][cfm]]
            conf_mat_report = pd.DataFrame([conf_mat_report+[full_conf_mat['train_mcc'], full_conf_mat['valid_mcc']]], columns = conf_mat_log_cols)
            full_performance = pd.DataFrame(np.hstack((train_out['train_history'].tail(1),conf_mat_report)),
                                             columns=np.concatenate((train_out['train_history'].columns, conf_mat_log_cols)))\
                                             .round(3).drop(columns=['lr', 'iter', 'epoc', 'time'])
            full_performance.columns = map(str.lower, full_performance.columns)
        
        if n_splits > 1:
            final_performance = split_performance
            final_performance.columns = map(str.lower, final_performance.columns.str.replace('_Avg', ''))
        elif n_splits_full == 1:
            final_performance = full_performance
        else:
            final_performance = pd.DataFrame()
            
        if cv_for_full_model_only == False:
            
            # add confusion matrix of split average with full dataset model
            reload_t = torch.load('checkpoints/'+session_ID+'_full_final_model.pth')
            model_full = reload_t['model']
            train_out_full = reload_t['train_out']
            if n_splits_full > 1:
                full_performance=CrossValidationUtils.train_predict_CV(mode='predict',
                                                        dataset=copy.deepcopy(dataset),
                                                        base_classifier=model_full,
                                                        base_classifier_train_out=train_out_full, invert_class=invert_class,
                                                        n_splits=n_splits_full, stratify_lab=dataset.labels,
                                                        batch_size=batch_size, trainer_ID=trainer_ID, conf_mat_meas=conf_mat_meas,
                                                        binary_thresh=binary_thresh, multi_class=multi_class,
                                                        multi_class_mode=multi_class_mode, loss_fun=loss_fun, 
                                                        cv_for_full_model_only=True, split_seed=split_seed**2+1, silent=silent)  
            # perfomance summary for CV and Full
            if not (n_splits == 1 and n_splits_full == 1):
                final_performance = final_performance.append(full_performance)
            final_performance['Set'] = (['Cross-Validation', 'Full'] if n_splits > 1 else 'Full')
            final_performance.set_index('Set', inplace=True)
            if mode == 'train' and silent == False:
                print('\nPerformance summary:')
                display(final_performance)
        else:
            if len(final_performance) > 0:
                for col in ['valid_loss', 'valid_roc', 'valid_acc', 'train_loss', 'train_roc', 'train_acc']:
                    final_performance[col] = final_performance[col].str.replace(' ± 0.0', '') # for full dataset there's no variation

        # plot and save performance merging in single plot
        if mode == 'train':
            split_path=os.path.join('results', main_folder, session_ID+'_split_performance.png')
            full_path=os.path.join('results', main_folder, session_ID+'_full_performance.png')
            merge_path=os.path.join('results', main_folder, session_ID+'_performance.png')
            if n_splits > 1:
                fig_split_performance = plot_performance(df = split_log, meas_list = measure_to_plot, super_title=session_ID,
                                                         set_type='split', show_fig=False)
                fig_split_performance.savefig(split_path, bbox_inches='tight')
            fig_full_performance = plot_performance(df = train_out['train_history'], meas_list = measure_to_plot, super_title=session_ID,
                                                    set_type='full', show_fig=False)
            fig_full_performance.savefig(full_path, bbox_inches='tight')
            if n_splits > 1:
                get_concat_h([Image.open(split_path), Image.open(full_path)], offset=20).save(merge_path)
                os.remove(split_path)
                os.remove(full_path)
            else:
                os.rename(full_path, merge_path)
            if show_perf_plot:
                print('\n\n\n')
                display(Image.open(merge_path))
                
        # plot distribution of predicted probabilities
        if mode == 'train':
            if n_splits > 1:
                # put all split prediction together
                split_train_preds = []
                split_valid_preds = []
                for k, v in split_prediction_out.items():
                    if len(split_train_preds) == 0 and len(split_valid_preds) == 0:
                        split_train_preds = v['train_preds']
                        split_valid_preds = v['valid_preds']
                    else:
                        split_train_preds = pd.concat([split_train_preds, v['train_preds']], axis=0)
                        split_valid_preds = pd.concat([split_valid_preds, v['valid_preds']], axis=0)
                split_path = plot_prediction_distribution(set_dict={'train_preds': split_train_preds, 'valid_preds': split_valid_preds},
                                                         original_dataset=dataset.df, session_ID=session_ID+'_split',
                                                         main_title_prefix='Cross-Validation on ' + str(n_splits) + ' splits',
                                                         save_path=os.path.join('results', main_folder))
            full_path = plot_prediction_distribution(set_dict=full_prediction_out, original_dataset=dataset.df,
                                                     main_title_prefix='Full Dataset', session_ID=session_ID+'_full',
                                                     save_path=os.path.join('results', main_folder))
            merge_path = os.path.join('results', main_folder, session_ID+'_Prediction_distribution.png')
            if n_splits > 1:
                get_concat_v([Image.open(os.path.join('results', main_folder, session_ID+x+'_Prediction_distribution.png')) for x in ['_split', '_full']],
                                 offset=20, font_size=20,
                                 add_main_title='Prediction probabilities distribution: ' + session_ID).save(merge_path)
                _=[os.remove(os.path.join('results', main_folder, session_ID+x+'_Prediction_distribution.png')) for x in ['_split', '_full']]
            else:
                print('\nPrediction probabilities distribution:')
                os.rename(full_path, merge_path)
            if show_distr_plot:
                display(Image.open(merge_path))
            
        if cv_for_full_model_only:
            try:
                [os.remove(os.path.join('results', main_folder, session_ID+'_split'+str(split)+'FF_conf_mat.png'))
                 for split in range(1, n_splits+1)] # remove split confusion matrices
            except:
                pass
        if cv_for_full_model_only == False:
            
            # merge all confusion matrix into single plot - only for saving .png

            get_concat_v(
               ([Image.open(os.path.join('results', main_folder, session_ID+'_split'+str(split)+'_conf_mat.png')) 
                    for split in range(1, n_splits+1)] if n_splits > 1 else []) + 
                ([Image.open(os.path.join('results', main_folder, 'avg_split_conf_mat.png'))] if n_splits > 1 else []) + 
                [Image.open(os.path.join('results', main_folder, session_ID+'_full_conf_mat.png'))] + 
                ([Image.open(os.path.join('results', main_folder, 'avg_split_conf_mat_full.png'))] if n_splits_full > 1 else [])
                ).save(os.path.join('results', main_folder, session_ID+'_conf_mat.png'))
            if n_splits > 1:
                [os.remove(os.path.join('results', main_folder, session_ID+'_split'+str(split)+'_conf_mat.png')) for split in range(1, n_splits+1)]
                os.remove(os.path.join('results', main_folder, 'avg_split_conf_mat.png'))
            os.remove(os.path.join('results', main_folder, session_ID+'_full_conf_mat.png'))
            if n_splits_full > 1: os.remove(os.path.join('results', main_folder, 'avg_split_conf_mat_full.png'))

            final_performance['loss'] = loss_fun
            final_performance['inverted_class'] = 'yes' if invert_class else 'no'
            final_performance['session_ID'] = session_ID
            final_performance['model_ID'] = base_classifier_train_out['model_ID']
            final_performance=final_performance[['session_ID'] + [x for x in final_performance.columns.tolist() if x != 'session_ID']]

            out = {'session_ID': session_ID,
                   'original_dataset': dataset.df,
                   'split_log': split_log,
                   'final_performance': final_performance,
                   'split_distribution': {'train_distr': split_train_distr,
                                          'valid_distr': split_valid_distr},
                   'full_distribution': full_distr,
                   'options': {'loss_add_function': loss_add_function,
                               'invert_class': invert_class,
                               'multi_class': multi_class,
                               'multi_class_mode': multi_class_mode,
                               'binary_thresh': binary_thresh}}

            # save pickle
            with open('checkpoints/'+session_ID+'_full_train_result.pickle', 'wb') as handle:
                pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            out = final_performance
            
        torch.cuda.empty_cache()
        
        return out

    
    @staticmethod
    def best_model_performance(trainer_ID, model_spec, subset_to_evaluate=[]):
        '''
        Args:
            - trainer_ID: trainer_ID identifier for dataset class
            - model_spec: spec for selected model among the trainer_ID family. e.g. 'Resnet34_BCE' or 'Resnet34_FocLos_inv_class'
            - subset_to_evaluate: list of subset to be used to evaluate confusion matrix. Must be taken from original_dataset['source']
        '''
        
        model_ID=trainer_ID+'_'+model_spec
        print('\nBest model ID:', model_ID)
        reload = torch.load('checkpoints/'+model_ID+'_final_model.pth')
        model = reload['model']
        train_out = reload['train_out']
        print('\nLoaded model:', model.__class__.__name__)
        model.load_state_dict(reload['model_state_dict'])   # reload state

        with open('checkpoints/'+model_ID+'_train_result.pickle', 'rb') as handle:
            out = pickle.load(handle)
        model_final_performance = out['final_performance']
        original_dataset = out['original_dataset']
        try:   # make compatible with previous version of pickle file
            multi_class = out['multi_class']
            multi_class_mode = out['multi_class_mode']
            binary_thresh = out['binary_thresh']
        except:   # set to default
            multi_class = False
            multi_class_mode = 'max'
            binary_thresh = 0.5
            
        print('\nFinal performance:')
        display(model_final_performance)
        print('\nDataset distribution:')
        display(out['full_distribution'])

        # full dataset performance
        print('\n\n  ==============================  Full Dataset  ==============================\n')
        _=evaluate_confusion_matrix(original_dataset=original_dataset, train_out=train_out, binary_thresh=binary_thresh,
                                    save_path=trainer_ID,save_ID=model_ID+'_BEST_full', add_title=model_ID+' - Full Dataset\n',
                                    show_result=True, cmap=plt.cm.Reds,
                                    multi_class=multi_class, multi_class_mode=multi_class_mode)

        # subset dataset performance
        for source in subset_to_evaluate:
            print('\n\n  ==============================  '+source+' Dataset  ==============================\n')
            set_ind = original_dataset[original_dataset.source == source].index.tolist()
            _=evaluate_confusion_matrix(original_dataset=original_dataset, train_out=train_out, binary_thresh=binary_thresh,
                                        save_path=trainer_ID,save_ID=model_ID+'_BEST_'+source, add_title=model_ID+' - '+source+'\n',
                                        show_result=True, subset_iloc=set_ind,
                                        multi_class=multi_class, multi_class_mode=multi_class_mode)

            
    @staticmethod
    def tune_with_CV(trainer_ID, dataset, base_classifier_ID, batch_size,
                     n_splits=5, n_splits_full=5, max_epochs_split=20, max_epochs_full=20, stratify_lab=None, loss_set=[],
                     multi_class=False, multi_class_mode='max', binary_thresh=0.5, invert_class_set=False,
                     measure_to_plot=[], metrics=[], conf_mat_meas=['f1-score'], show_info=False, show_log=2,
                     show_perf_plot=True, show_distr_plot=True, save_split_checkpoint=False, save_split_final=True,
                     save_full_checkpoint=False, save_full_final=True, reload_split=False, reload_full=False, silent=False):
        
        '''
        Train model with Cross-Validation. Wrapper for train_predict_CV().

        Arg:
            trainer_ID: ID to define current setting. Used as prefix for saved images, model and output
            dataset: dataset used
            base_classifier_ID: string for pretrained classifier to be reloaded for transfer learning.
                                If passing a raw nn.model, no pretrained model will be loaded
            batch_size: batch size for models
            n_splits: number of folds for Cross-Validation
            n_splits_full: number of folds to predict confusion matrix on full dataset model
            max_epochs_split: max epoch for each fold
            max_epochs_full: max epoch for model trained on full dataset
            stratify_lab: array to be used for stratified CV
            loss_set: set of loss function to be used for train. 'BCE' or 'NLL' or 'FocLos' available
            multi_class: True for multi class task
            multi_class_mode: how to set predicted class given the probabilities array. 'max': takes highest probability
            binary_thresh: threshold for binary classification on predicted values
            invert_class_set: if True both invert_class=True and invert_class=False will be tested. Only if multi_class=False
            measure_to_plot: measure to be plotted for performance evolution over epochs. Both CV and full model are provided
            metrics: metrics to be evaluated for train
            conf_mat_meas: list of measures to be logged for each class, taken from confusion matrix report
            show_info, show_log: see trainer.Trainer.run
            show_perf_plot: if True performance evolutions are plotted
            show_distr_plot: if True predicted probabilities distribution are plotted
            save_split_checkpoint: save model checkpoints for CV models
            save_split_final: save final model for CV models
            save_full_checkpoint: save model checkpoints for full dataset model
            save_full_final: save final model for full dataset model
            reload_split: reload each CV model
            reload_full: reload full dataset model
            silent: if True suppress all print, display, plot
            
        '''
        
#         set_seed()
        # create directories
        _=os.makedirs(os.path.join('results', trainer_ID), exist_ok=True)

        # load base_classifier
        if type(base_classifier_ID) == str:
            reload = torch.load('checkpoints/'+base_classifier_ID+'_final_model.pth')
            base_classifier = reload['model']
            base_classifier_train_out = reload['train_out']
            if silent == False: print('\nLoaded base classifier:', base_classifier_ID, '('+base_classifier.__class__.__name__+')')
            base_classifier.load_state_dict(reload['model_state_dict'])   # reload state
        else:
            base_classifier = base_classifier_ID
            base_classifier_train_out = {'train_history': pd.DataFrame(columns = ['lr', 'iter', 'epoc', 'Valid_loss', 'Valid_roc',
                                                                                  'Valid_acc', 'Train_loss', 'Train_roc', 'Train_acc', 'time'],
                                                                       dtype=float).fillna(''),
                                         'model_ID': base_classifier_ID.__class__.__name__}
            if silent == False: print('\nUsed raw classifier:', base_classifier_ID.__class__.__name__)

        # hyperparameter tuning on loss_set and invert_class_set with Cross-Validation and final model training
        final_log=[]
        out_res={}
        if silent == False: print('\nStart hyperparameter tuning')
        if multi_class: invert_class_set=False
        start = timer()
        for loss_fun in loss_set:
            loss_add_function=None
            if loss_fun == 'NLL':
                loss_add_function=torch.log
            
            for invert_class in [False, True] if invert_class_set else [False]:

                if silent == False: print('\n\n'+'='*30, loss_fun+' - '+('InvertClass:yes' if invert_class else 'InvertClass:no'), '='*30+'\n')

                out = CrossValidationUtils.train_predict_CV('train', copy.deepcopy(dataset), base_classifier, base_classifier_train_out,
                                                            n_splits, n_splits_full, stratify_lab, max_epochs_split, max_epochs_full, loss_fun,
                                                            metrics, loss_add_function, multi_class, multi_class_mode, binary_thresh,
                                                            invert_class, measure_to_plot, conf_mat_meas, batch_size,
                                                            show_info, show_log, show_perf_plot, show_distr_plot, save_split_checkpoint,
                                                            save_split_final, save_full_checkpoint, save_full_final,
                                                            reload_split, reload_full, trainer_ID, silent)
                out_final_performance = out['final_performance']
                out_res[loss_fun+('_inv_class' if invert_class else '')] = out

                if len(final_log) == 0:
                    final_log=pd.DataFrame(columns = out_final_performance.columns, dtype=float).fillna('')
                final_log=final_log.append(out_final_performance)

        final_log.to_csv(os.path.join('results', trainer_ID, trainer_ID+'_final_performance.csv'), index=True, sep=';')
        if silent == False:
            print('\n\n\nFinal Log:')
            display(final_log)
            print('\nTotal elapsed time:', str(datetime.timedelta(seconds=round(timer()-start))))
        
        torch.cuda.empty_cache()
        
        return out_res

        
    @staticmethod
    def predict_dataset(model_ID, dataset, new_index, batch_size, binary_thresh=0.5, multi_class=False,
                        multi_class_mode='max', invert_class=False, conf_mat_meas = ['f1-score', 'precision', 'recall'],
                        metrics = None, show_result=True,
                        save_path='', add_title='', save_ID='', normalize=None, cmap=plt.cm.Blues):
        '''
        Args:
            - model_ID: string for pretrained classifier to be reloaded for prediction.
                        If passing a raw nn.model, no pretrained model will be loaded
            - dataset: dataset to predict from
            - new_index: index of observation to predict
            - batch_size: batch size to be used
            - binary_thresh: threshold for binary classification on predicted values
            - multi_class: True if multiclass task. If model is reloaded, input value is replaced by reloaded one
            - multi_class_mode: how to set predicted class given the probabilities array. 'max': takes highest probability
            - invert_class: invert class in prediction. If model is reloaded, input value is replaced by reloaded one
            - conf_mat_meas: list of confusion matrix metrics to be exported, by each class, in the conf_mat_report dataframe
            - metrics: dictionary of additional name-metrics (overall) to be added to conf_mat_report dataframe
                    {'acc': Accuracy(), 'roc': roc_auc_score} if multi_class=False else {'acc': AccuracyMulti(), 'roc': ROCMulti()}
            - add_title: additional title used as super title
            - normalize: normalize confusion matrix entries
            - cmap: colormap to be used
        '''
        
        # reload model and predict
        if type(model_ID) == str:
            reload = torch.load('checkpoints/'+model_ID+'_final_model.pth')
            model = reload['model']
            invert_class = reload['train_out']['options']['invert_class']
            multi_class = reload['train_out']['options']['multi_class']
        else:
            model = model_ID
        model.cuda()
        
        if metrics is not None:
            metrics_name = list(metrics.keys())
            metrics = list(metrics.values())
        if multi_class: invert_class = False
        ev = Evaluator(mode='predict', dataset=dataset, set_idx=new_index, batch_size=batch_size, multi_class=multi_class)
        ev_out = ev.train_predict(model, metrics=metrics)
        new_preds = ev_out['preds']
        if metrics is not None:
            ev_metrics = ev_out['metrics']
        # invert prediction
        if invert_class:
            new_preds.pred = 1 - new_preds.pred
            if metrics is not None:
                ev_metrics = [i(torch.tensor(new_preds.true.values), torch.tensor(new_preds.pred.values)).item() for i in metrics]

        # prepare input for confusion matrix
        original_dataset = dataset.df
        original_combination = original_dataset[['label', 'meaning']].drop_duplicates().sort_values(by=['meaning'])
        avail_label=original_combination.label.values.astype(new_preds.true.values.dtype)
        missing_label=[]
        new_preds_true = new_preds.true.values
        new_preds_eval = evaluate_prediction(new_preds, multi_class=multi_class, multi_class_label=avail_label, multi_class_mode=multi_class_mode)
        new_preds_pred = new_preds_eval['pred_label']
        new_preds[['pred_class', 'pred_perc']] = new_preds_eval
        if len(np.unique(new_preds_true)) == 1:
            missing_label = avail_label[avail_label != np.unique(new_preds_true)]
            new_preds_true = np.append(new_preds_true, missing_label)
            new_preds_pred = np.append(new_preds_pred, missing_label)
            fict_row = copy.deepcopy(original_dataset.iloc[0:len(missing_label)]).drop(columns='meaning')
            fict_row['label']=missing_label
            fict_row = pd.merge(fict_row, original_combination, how='inner', on=['label'])
            original_dataset = original_dataset.append(fict_row)

        pred_fig, pred_conf_mat, pred_mcc, pred_class_report, pred_opt = plot_confusion_matrix(new_preds_true, new_preds_pred,
                                                                                             original_dataset=original_dataset,
                                                                                             title='Prediction '+add_title, show_result=False,
                                                                                             override_label=missing_label,
                                                                                             normalize=normalize, cmap=cmap)

        class_report = class_report_to_df(pred_class_report, top_column_index='')
        
        # create confusion matrix report (dataframe)
        avail_class = sorted(dataset.df.meaning.unique().tolist())
        conf_mat_log_cols = ['prediction_'+x+'_'+y for y in conf_mat_meas for x in avail_class]
        conf_mat_log_cols = conf_mat_log_cols + ['prediction_mcc']
        conf_mat_log_cols=np.array(conf_mat_log_cols).ravel()
        conf_mat_report = []
        for cfm in conf_mat_meas:
            for cl in avail_class:
                conf_mat_report += [pred_class_report[cl][cfm]]
        conf_mat_report = pd.DataFrame([conf_mat_report+[pred_mcc]], columns = conf_mat_log_cols)
        if metrics is not None:
            conf_mat_report = pd.concat([conf_mat_report, pd.DataFrame([ev_metrics], columns=['prediction_'+x for x in metrics_name])], axis = 1)

        if show_result:
            display(pred_fig)
            display(class_report)

        if save_ID != '':
            pred_fig.savefig(os.path.join('results', save_path, save_ID+'_prediction_conf_mat.png'))

        out = {'prediction': new_preds,
               'pred_conf_mat': pred_conf_mat,
               'pred_mcc': pred_mcc,
               'pred_class_report': pred_class_report,
               'conf_mat_report': conf_mat_report}
        
        model.cpu()
        del model
        torch.cuda.empty_cache()

        return out
    
