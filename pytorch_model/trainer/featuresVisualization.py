import copy
import os
import pickle
import diagnostics
from pathlib import Path
from timeit import default_timer as timer
import datetime
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import cm
import seaborn as sns
import joypy
import cv2
from PIL import Image, ImageDraw, ImageFont
from trainer import Trainer, Evaluator
from metrics.plot_functions import *
from metrics.confusion_matrix import evaluate_prediction
import matplotlib.colors as clr


class FeaturesVisualizationUtils:
    def __init__(self, save_path = '', model_ID = '', image_size = 256):
        '''
        Args:
            - save_path, model_ID: save path for figures and model_ID used as prefix
            - image_size: size of original figures used to resize heatmaps when saving figures
        '''
        
        self.save_path = save_path
        self.model_ID = model_ID
        self.image_size = image_size

    def run_features_visualization(self, dataset, new_index, model, reload_model_ID, frame_to_show=None,
                                   feature_batch_size = 100, gradient_aggregation = 'mean', apply_soft_activation = True,
                                   nn_to_replace = nn.ReLU, nn_replacement = nn.LeakyReLU(inplace=True),
                                   heatmap_transformation='ReLU', ht_transf_add_param=None, original_to_gray=True,
                                   heatmap_mask_alpha=0.8):
        '''
        Main wrapper to evaluate features visualization with Grad-CAM (https://arxiv.org/pdf/1610.02391.pdf)
        https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82

        # todo: check case of inverted class, probably we should invert also heatmap


        Args:
            - dataset: dataset to be used for prediction
            - new_index: index to select images from dataset (must be an np.array)
            - model: raw model with feature visualization hooks enabled (e.g. Resnet34FeatVis())
            - reload_model_ID: ID for the model .pth file to reload. Trained weights will be applied to raw model
            - frame_to_show: frame to be used as background for heatmap. None if images have just one.
            - feature_batch_size: in order not to get out of memory error for GPU new_index is split in batches. Keep below 200.
            - gradient_aggregation: 'mean' or 'sum'. Aggregation method for multiclass problem. Should be consistent with 
                                    reduction method used for the Loss Function
            - apply_soft_activation: if True substitute activation function defined in nn_to_replace with the one in nn_replacement in
                                     order to avoid null gradient. Separate statistics for average heatmap will be produced.
            - nn_to_replace, nn_replacement: see above
            - heatmap_transformation: final tranformation to be applied to heatmap
                                      - 'ReLU': apply ReLU as from Grad-CAM algo
                                      - 'linear': apply linear scaling keeping positive and negative values
                                      - 'exponential': apply exponential scaling keeping positive and negative values
            - ht_transf_add_param: additional parameter for heatmap_transformation (if any) (see _range_scaler())
            - original_to_gray: if True original image (background for heatmap) is converted to grayscale. Input format is BGR
            - heatmap_mask_alpha: coefficient of transparency when superimposing heatmap to original image
            
        '''

        COLOR_MAP_NEG_POS = clr.LinearSegmentedColormap.from_list('red-black-green', ['red', 'black', 'green'], N=256)
        COLOR_MAP_NEG_POS_white = clr.LinearSegmentedColormap.from_list('red-black-green', ['red', 'white', 'green'], N=256)
#         COLOR_MAP_NEG_POS = clr.LinearSegmentedColormap.from_list('red-black-green', ['yellow','black', 'blueviolet'], N=256)
        COLOR_MAP_ZERO_POS = clr.LinearSegmentedColormap.from_list('black-green', ['black', 'green'], N=256)
        COLOR_MAP_ZERO_POS_white = clr.LinearSegmentedColormap.from_list('black-green', ['white', 'green'], N=256)
        
        if heatmap_transformation in ['ReLU']:
            self.cmap = COLOR_MAP_ZERO_POS
            self.cmap_white = COLOR_MAP_ZERO_POS_white
        elif heatmap_transformation in ['linear', 'exponential']:
            self.cmap = COLOR_MAP_NEG_POS
            self.cmap_white = COLOR_MAP_NEG_POS_white
        
        if dataset.df.image.nunique() != dataset.df.shape[0]:
            raise ValueError('Input dataset contains duplicated images')
        if gradient_aggregation not in ('mean', 'sum'):
            raise ValueError('Please provide gradient_aggregation as \'mean\' or \'sum\' - current is: ' + gradient_aggregation)
        if heatmap_transformation not in ('ReLU', 'linear', 'exponential'):
            raise ValueError('Please provide heatmap_transformation as \'ReLU\' or \'linear\' or \'exponential\' - current is: '+
                             heatmap_transformation)
        print('Evaluating heatmap:')
        print('\n -- Gradient aggregation:', gradient_aggregation)
        if apply_soft_activation:
            print(' -- Soft activation is applied to', nn_to_replace, ', replaced with', nn_replacement)
        print('')

        # create directories
        _=os.makedirs(os.path.join('results', self.save_path, 'FeatVis_Single_images'), exist_ok=True)

        # reload model
        reload = torch.load('checkpoints/'+reload_model_ID+'_final_model.pth')
        reload_model_check_ID = reload['model'].model_ID
        invert_class = reload['train_out']['options']['invert_class']
        multi_class = reload['train_out']['options']['multi_class']
        if reload_model_check_ID != model.model_ID:
            raise ValueError('Raw model provided ('+model.model_ID+') is different from reloaded one ('+reload_model_check_ID+')')
        model = model.cuda()
        model.load_state_dict(reload['model_state_dict'])
        
        # loop for each feature_batch_size
        new_index_set=np.array_split(new_index, math.ceil(len(new_index) / feature_batch_size))
        final_log = []
        heatmap_list = []
        start = timer()
        for i, new_index_batch in enumerate(new_index_set):

            print('Processing ' + str(i+1) + '/' + str(len(new_index_set)) +
                  ' batch - Total elapsed time: ' + str(datetime.timedelta(seconds=round(timer()-start))), end='\r')
            new_index_batch = new_index_batch.tolist()

            # evaluate prediction and export images
            ev = Evaluator(mode = 'feat_vis', dataset=dataset, set_idx=new_index_batch, batch_size=1, # keep batch_size = 1
                           multi_class=multi_class)
            new_preds = ev.train_predict(model)

            image_list = new_preds['preds']['img']
            prediction_list = new_preds['preds']['pred_list']
            prediction_table_t = new_preds['preds']['pred_table']
            
            # invert prediction
            if invert_class:
                prediction_list = [1 - x for x in prediction_list]
                prediction_table_t.pred = 1 - prediction_table_t.pred

            # add predicted class
            label_descr = dataset.df[['label', 'meaning']].drop_duplicates().sort_values(by=['meaning'])
            prediction_table_t[['pred_class', 'pred_perc']] = evaluate_prediction(prediction_table_t, multi_class=multi_class,                                                            multi_class_label=label_descr.label.values.astype(prediction_table_t.true.values.dtype))
            indices = np.array(prediction_table_t.index)
            # evaluate heatmap
            heatmap_list_t, final_log_t = FeaturesVisualizationUtils._evaluate_heatmap(model, dataset, invert_class, image_list, prediction_list,
                                                                                      prediction_table_t, label_descr, indices,
                                                                                      self.model_ID, self.save_path, multi_class,
                                                                                      self.image_size, gradient_aggregation,
                                                                                      apply_soft_activation, nn_to_replace, nn_replacement,
                                                                                      new_index=new_index_batch, frame_to_show=frame_to_show,
                                                                                      original_to_gray=original_to_gray,
                                                                                      heatmap_transformation=heatmap_transformation,
                                                                                      ht_transf_add_param=ht_transf_add_param,
                                                                                      heatmap_mask_alpha=heatmap_mask_alpha,
                                                                                      cmap=self.cmap)

            # append results
            if len(final_log) == 0:
                prediction_table = copy.deepcopy(prediction_table_t)
                final_log = copy.deepcopy(final_log_t)
                heatmap_list = copy.deepcopy(heatmap_list_t)
            else:
                prediction_table = pd.concat([prediction_table, copy.deepcopy(prediction_table_t)], axis=0)
                final_log = pd.concat([final_log, copy.deepcopy(final_log_t)], axis=0)
                heatmap_list = np.concatenate([heatmap_list, copy.deepcopy(heatmap_list_t)])
            del final_log_t, heatmap_list_t
        print('\nDone\n\n')
        model = model.cpu()
        del image_list
        torch.cuda.empty_cache()
        
        self.final_log = final_log
        self.heatmap_list = heatmap_list
        self.label_descr = label_descr
        self.heatmap_transformation = heatmap_transformation

        
    def plot_heatmap(self, fig_size = 12, heatmap_size = 256, white_cmap = False, show_fig = True, by_class_title = ''):
        '''
        Args:
            - fig_size: overall figures size (y shape is automatically adjusted)
            - heatmap_size: resize heatmaps before evaluating averages (seems not to affect final results)
            - white_cmap: use white as neutral color instead of black
        '''
        if by_class_title == '':
            by_class_title='Average of features activations by class\nwithout null and soft'
        box_plot_df = FeaturesVisualizationUtils._plot_heatmap(self.heatmap_list, self.final_log, self.label_descr, self.save_path,
                                                               self.model_ID, self.image_size, heatmap_size, fig_size,
                                                               (self.cmap_white if white_cmap else self.cmap),
                                                               self.heatmap_transformation, show_fig, by_class_title)
        self.box_plot_df = box_plot_df
    
    
    def plot_sample_masked(self, fig_size = 12, max_sample_combination = 30, sample_ncol = 4, white_cmap = False, show_fig = True):
        '''
        Args:
            - fig_size: overall figures size (y shape is automatically adjusted)
            - max_sample_combination: max number of sample for each combination (if available) of true class vs predicted class.
                                      Null and soft activation are excluded
            - sample_ncol: number of images per row in masked samples plot
            - white_cmap: use white as neutral color instead of black
        '''
        
        FeaturesVisualizationUtils._plot_sample_masked(self.final_log, self.save_path, self.model_ID, max_sample_combination,
                                                       sample_ncol, fig_size, (self.cmap_white if white_cmap else self.cmap),
                                                       self.heatmap_transformation, show_fig)
        
        
    def plot_heatmap_max_distribution(self, show_fig = True):
        FeaturesVisualizationUtils._plot_heatmap_max_distribution(self.final_log, self.save_path, self.model_ID, show_fig)
    
    
    @staticmethod
    def _range_scaler(x, a, b, xmin = None, xmax = None, mode = 'linear', s = None):
        '''
        Scale input interval into new range
        - a, b: new interval range
        - xmin, xmax: provided if scaling has to be performed from a different input range [x.min, x.max()]
        - mode: 'linear' for linear scaling, 'exponential' for exponential scaling
        - s: if mode == 'exponential' s is used for decay in exponential kernel.
             The higher s the more spiked the decay (leptokurtic)
        '''

        if xmin is None: xmin = x.min()
        if xmax is None: xmax = x.max()

        if mode == 'linear':
            # https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
            out = (b - a) * (x - xmin) / (xmax - xmin) + a

        if mode == 'exponential':
            if s is None: s = 5
            # https://stackoverflow.com/questions/49184033/converting-a-range-of-integers-exponentially-to-another-range
            r = (x - xmin) / (xmax - xmin)
            C = s ** (xmax - xmin)
            out = ((b - a) * C ** r + a * C - b) / (C - 1)

        return out
    
    
    @staticmethod
    def _evaluate_heatmap(model, dataset, invert_class, image_list, prediction_list, prediction_table, label_descr, indices,
                          model_ID, save_path, multi_class, image_size, gradient_aggregation, apply_soft_activation,
                          nn_to_replace, nn_replacement, new_index, frame_to_show, original_to_gray,
                          heatmap_transformation, ht_transf_add_param, heatmap_mask_alpha=0.8, cmap=plt.cm.RdBu):        
        '''
        Evaluates heatmap for features visualization and saves single images. Used in wrapper run_features_visualization().

        Args:
            - see run_features_visualization()
            - cmap: color map for heatmap
        '''
    
        apply_colormap = cm.get_cmap(cmap)
        heatmap_list = []
        final_log=[]
        for ii, ind in enumerate(new_index):

#             print('                        - ' + str(ii) + '/' + str(len(new_index)) + 'images', end = '\r')
            
            # slice informations and prepare output for final_log
            dataset_row = copy.deepcopy(dataset.df.loc[ind])
            prediction_row = copy.deepcopy(prediction_table.loc[ind])
            orig_image_path = str(dataset_row.image)
            masked_image_path = os.path.join('results', save_path, 'FeatVis_Single_images', model_ID + '_loc_' + str(ind) + '.png')
            dataset_row['pred'] = prediction_row['pred']
            dataset_row['pred_label'] = prediction_row['pred_class']
            dataset_row['pred_perc'] = prediction_row['pred_perc']
            dataset_row['pred_meaning'] = label_descr.meaning.values[np.where(label_descr.label == prediction_row['pred_class'])][0]
            dataset_row['masked_image_path'] = masked_image_path
            dataset_row['soft_activation'] = 'No'
            dataset_row['pred_soft'] = None
            dataset_row['pred_soft_label'] = None
            dataset_row['pred_soft_perc'] = None
            dataset_row['pred_soft_meaning'] = None
            dataset_row['null_heatmap'] = 'No'
            dataset_row['pred_soft_diff_L2norm'] = None


            # evaluate everything according to order defined by new_index (indices may result in a shuffling of new_index)
            shuffled_i = np.where(indices == ind)[0][0]
            img = image_list[shuffled_i]
            pred = prediction_list[shuffled_i]

            # evaluate heatmap for feature visualization
            if gradient_aggregation == 'mean':
                pred.mean().backward()
            elif gradient_aggregation == 'sum':
                pred.sum().backward()
            gradients = model.get_activations_gradient()

            # if gradient is zero, replace all activation functions with non-zero output ones
            if gradients.max() == 0 and apply_soft_activation:
                dataset_row['soft_activation'] = 'Yes'
                model_soft = copy.deepcopy(model)
                FeaturesVisualizationUtils._replace_modules(model_soft, nn_to_replace = nn_to_replace, nn_replacement= nn_replacement)
                pred_soft = model_soft(img)
                if invert_class: pred_soft = 1 - pred_soft
                if gradient_aggregation == 'mean':
                    pred_soft.mean().backward()
                elif gradient_aggregation == 'sum':
                    pred_soft.sum().backward()
                gradients = model_soft.get_activations_gradient()
                prediction_row_soft = copy.deepcopy(prediction_row.to_frame().transpose()).drop(columns = 'pred')
                prediction_row_soft['pred'] = [pred_soft.cpu().detach().numpy().ravel()]
                dataset_row['pred_soft'] = prediction_row_soft['pred'].values[0]
                pred_soft = evaluate_prediction(prediction_row_soft, multi_class=multi_class,
                                                multi_class_label=label_descr.label.values.astype(prediction_table.true.values.dtype))
                dataset_row['pred_soft_label'] = pred_soft['pred_label'].values[0]
                dataset_row['pred_soft_perc'] = pred_soft['pred_perc'].values[0]
                dataset_row['pred_soft_meaning'] = label_descr.meaning.values[np.where(label_descr.label.values == prediction_row_soft.pred_class.values)][0]
                dataset_row['null_heatmap'] = 'No'
                dataset_row['pred_soft_diff_L2norm'] = np.linalg.norm((pred.cpu().detach().numpy()-prediction_row_soft['pred'].values))

            # pool the gradients across the channels
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

            # get the activations of the last convolutional layer
            activations = model.get_activations(img).detach()

            # weight the channels by corresponding gradients
            for i in range(activations.shape[1]): # range(512):
                activations[:, i, :, :] *= pooled_gradients[i]

            # average the channels of the activations
            heatmap = torch.mean(activations, dim=1).squeeze()
            
            # save max and min value for distribution
            ht_max = float(heatmap.cpu().max())
            ht_min = float(heatmap.cpu().min())
            dataset_row['heatmap_max'] = ht_max
            dataset_row['heatmap_min'] = ht_min
            
            # check if heatmap is null, otherwise apply heatmap_transformation
            if abs(ht_max) <1e-9 and abs(ht_min) <1e-9:
                dataset_row['null_heatmap'] = 'Yes'
                if heatmap_transformation == 'ReLU':
                    heatmap = heatmap.cpu().numpy()
                if heatmap_transformation in ['linear', 'exponential']:
                    heatmap = np.ones(heatmap.shape) * 0.5
            else:
                
                # apply heatmap_transformation
                if heatmap_transformation == 'ReLU':
                    # relu on top of the heatmap
                    # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
                    heatmap = np.maximum(heatmap.cpu(), 0)
                    # normalize the heatmap
                    heatmap /= torch.max(heatmap)
                    heatmap = heatmap.numpy()
                
                if heatmap_transformation in ['linear', 'exponential']:
                    # scale positive [0.5,1] and negative in [0,0.5] according to heatmap_transformation
                    scaler = FeaturesVisualizationUtils._range_scaler
                    heatmap_t = heatmap.cpu().numpy()
                    # translate the magnitude to 1 so that the scaling parameters are consistent for every heatmap
                    ht_abs_max = abs(heatmap_t).max()
                    heatmap_t = heatmap_t / (10**round(math.log10(ht_abs_max)))
                    ht_abs_max = abs(heatmap_t).max()
                    heatmap = np.zeros(heatmap_t.shape)
                    mask_pos = heatmap_t >=0
                    mask_neg = heatmap_t <=0
                    heatmap[mask_pos] = scaler(x=heatmap_t[mask_pos], a=0.5, b=1, xmin=0, xmax=ht_abs_max,
                                               mode=heatmap_transformation, s=ht_transf_add_param)
                    # symmetrize the negative values to be scaled in the same way as positive (exponential with base <1 may cause asymmetry)
                    heatmap[mask_neg] = 0.5-scaler(x=abs(heatmap_t[mask_neg]), a=0, b=0.5, xmin=0, xmax=ht_abs_max,
                                               mode=heatmap_transformation, s=ht_transf_add_param)
                if heatmap.min() < -1e-6 or heatmap.max() > 1+1e-6:
                    print('\nii:', ii, 'ind:', ind, 'heatmap outside [0,1]', heatmap.min(), heatmap.max())
                heatmap = np.clip(heatmap, 0, 1) # ensure approx error below 0 and above 1 are clipped

                # draw the heatmap
            #     plt.matshow(heatmap.squeeze())

                # load original image - handle pickle files
                if orig_image_path.endswith('.pickle') or orig_image_path.endswith('.pkl'):
                    with open(orig_image_path, 'rb') as handle:
                        original_image = pickle.load(handle)
                else:
                    original_image = cv2.imread(orig_image_path)
                # select single frame
                if frame_to_show is not None:
                    original_image = original_image[frame_to_show]
                # convert grayscale to RGB (=BGR) - just make the image with 3 channels
                if len(original_image.shape) == 2:
                    original_image = cv2.cvtColor(original_image,cv2.COLOR_GRAY2BGR)
                # convert original BGR to grayscale
                if original_to_gray:
                    original_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
                    original_image = cv2.cvtColor(original_image,cv2.COLOR_GRAY2BGR)  # just make the image with 3 channels
                original_image = cv2.resize(np.array(original_image), (image_size, image_size), interpolation=cv2.INTER_AREA)
                heatmap_mask = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_CUBIC)
                # ensure min=0 and max=1 are always present, for correct scaling of colormap (when saving with cv2.imwrite)
                heatmap_mask[0, 0] = 0
                heatmap_mask[-1, 0] = 1
                if (heatmap_mask.min() > 0.01 and heatmap_mask.max() < 0.99) or (heatmap_mask.min() > 0.01 and heatmap_mask.max() < 0.99):
                    print('ii:', ii, 'ind:', ind, 'heatmap_mask not properly scaled', heatmap_mask.min(), heatmap_mask.max())
                np_rgba = apply_colormap(heatmap_mask)  # apply colormap and extract RGBA np.array
                heatmap_mask = np.uint8(np_rgba[:,:,:3]*255)  # remove alpha channel (always=1)
                masked_image = heatmap_mask_alpha*heatmap_mask + original_image
                masked_image = masked_image[:,:, [2, 1, 0]]  # RGB to BGR
                cv2.imwrite(masked_image_path, masked_image)
                
            heatmap_list.append(heatmap)

            # append final_log
            if len(final_log) == 0:
                final_log=pd.DataFrame(columns = dataset_row.index, dtype=float).fillna('')
            final_log=final_log.append(dataset_row)

        heatmap_list = np.asarray(heatmap_list)
        torch.cuda.empty_cache()

        return heatmap_list, final_log


    @staticmethod
    def _plot_heatmap(heatmap_list, final_log, label_descr, save_path, model_ID, image_size,
                      heatmap_size = 256, fig_size = 12, cmap=plt.cm.RdBu, heatmap_transformation='ReLU', show_fig=True,
                     by_class_title='Average of features activations by class\nwithout null and soft'):
        '''
        Evaluate 4 average heatmaps (overall, all without null gradient, all without null and soft activation, only soft activation without null).
        Add bar plot and box plot for differences of new predicted classes with soft activation.
        Evaluate average heatmaps (no soft, no null) for each class comparing correct vs incorrect prediction in a confusion matrix shape plot.
        
        Args:
            - heatmap_list, final_log: heatmaps list and log produced in _evaluate_heatmap
            - label_descr: class label description
            - save_path, model_ID: save path for figures and model_ID used as prefix
            - image_size: size of original figures used to resize heatmaps when saving figures
            - heatmap_size: resize heatmaps before evaluating averages (seems not to affect final results)
            - fig_size: overall figures size (y shape is automatically adjusted)
            - cmap: colormap to be used
            - heatmap_transformation: final tranformation to be applied to heatmap. Used to define ticks of colorbar only
            - by_class_title: title for confusion matrix shape plot
        '''

        apply_colormap = cm.get_cmap(cmap)
        if heatmap_transformation == 'ReLU':
            ticks_label = ['0', '0.25', '0.5', '0.75', '1']
        if heatmap_transformation in ['linear', 'exponential']:
            ticks_label = ['-1', '-0.5', '0', '0.5', '1']
       
        
        ### create average and class heatmaps

        # resize all heatmaps
        heatmap_list_work = []
        _=[heatmap_list_work.append(cv2.resize(heatmap_list[i,:,:], (heatmap_size, heatmap_size),
                                               interpolation=cv2.INTER_CUBIC)) for i in range(heatmap_list.shape[0])]
        heatmap_list_work= np.asarray(heatmap_list_work)

        # average heatmap for all images (including null heatmap)
        heatmap_overall = heatmap_list_work.mean(axis=0)
        null_count = sum(final_log.null_heatmap == 'Yes')
        soft_count = sum(final_log.soft_activation == 'Yes')
        soft_and_null_count = sum((final_log.soft_activation == 'Yes') & (final_log.null_heatmap == 'Yes'))

        # average heatmap for all images excluding null heatmap
        null_index = np.where(final_log.null_heatmap == 'No')[0]
        heatmap_overall_not_null = heatmap_list_work[null_index].mean(axis=0) if len(null_index) > 0 else np.zeros(heatmap_overall.shape)

        # average heatmap for all images excluding null heatmap and soft activation
        soft_and_null_index = np.where((final_log.soft_activation == 'No') & (final_log.null_heatmap == 'No'))[0]
        heatmap_no_soft_not_null = heatmap_list_work[soft_and_null_index].mean(axis=0) if len(soft_and_null_index) > 0 else np.zeros(heatmap_overall.shape)

        # average heatmap for soft activation images excluding null heatmap
        soft_index = np.where((final_log.soft_activation == 'Yes') & (final_log.null_heatmap == 'No'))[0]
        heatmap_soft = heatmap_list_work[soft_index].mean(axis=0) if len(soft_index) > 0 else np.zeros(heatmap_overall.shape)
        total_images = len(heatmap_list_work)

        # average heatmap for each predicted class (correct vs incorrect), excluding null and soft activation
        class_df = final_log[(final_log.soft_activation == 'No') & (final_log.null_heatmap == 'No')]
        class_heatmap_list = []
        for pred_mean in label_descr.meaning:
            pred_row = []
            hm_count = []
            true_label = []
            for true_mean in label_descr.meaning:
                meaning_df = class_df[(class_df.pred_meaning == pred_mean) & (class_df.meaning == true_mean)]
                iloc_index = [np.where([x == y for x in final_log.index])[0][0] for y in meaning_df.index]
                hm_to_save = heatmap_list_work[iloc_index].mean(axis=0) if len(iloc_index) > 0 else np.zeros(heatmap_list_work[0].shape)
                pred_row.append(hm_to_save)
                hm_count.append(len(iloc_index))
                true_label.append(true_mean)
            class_heatmap_list.append({'pred_meaning': pred_mean,
                                       'true_meaning': true_label,
                                       'heatmap_list': pred_row,
                                       'heatmap_count': hm_count})
        if sum([sum(x['heatmap_count']) for x in class_heatmap_list]) != class_df.shape[0]:
            print('\n\n\n ###### error in class_heatmap_list')



        ### check differences in prediction for images with soft activation and prepare data for barplot and boxplot

        soft_dataset = copy.deepcopy(final_log[final_log.soft_activation == 'Yes'])
        soft_dataset.pred_label = soft_dataset.pred_label.astype(int)
        soft_dataset.pred_soft_label = soft_dataset.pred_soft_label.astype(int)

        box_plot_df = []
        misclass_ref = pd.DataFrame(label_descr.meaning).reset_index(drop=True).set_index('meaning')
        for meaning in soft_dataset.pred_meaning.unique():
            meaning_df = soft_dataset[soft_dataset.pred_meaning == meaning]
            L2diff_list = meaning_df.pred_soft_diff_L2norm.values
            row_to_add = pd.DataFrame({'meaning': meaning,
                                       'count': meaning_df.shape[0],
                                      'avg_L2_norm_diff': np.mean(L2diff_list),
                                      'L2diff_list': [L2diff_list]}, index=[0])
            misclass_df = meaning_df.pred_soft_meaning.value_counts().to_frame()
            misclass_df.index.name = 'meaning'
            misclass_final = pd.merge(misclass_ref, misclass_df, on='meaning', how='outer').fillna(0).transpose().reset_index(drop=True)
            row_to_add = pd.concat([row_to_add, misclass_final], axis=1)

            if len(box_plot_df) == 0:
                box_plot_df=pd.DataFrame(columns = row_to_add.columns, dtype=int).fillna('')
            box_plot_df=box_plot_df.append(row_to_add)



        ### create and save plot

        # plot average heatmap
        nrow = 3 if len(box_plot_df) > 0 else 2
        fig_size_y = fig_size*1.3 if len(box_plot_df) > 0 else fig_size
        fig, ax = plt.subplots(nrow, 2, figsize=(fig_size, fig_size_y), sharey=False, sharex=False)
        ax = ax.flatten()
        hm_list = [heatmap_overall, heatmap_overall_not_null, heatmap_no_soft_not_null, heatmap_soft]
        title_list = ['ALL IMAGES ('+str(total_images)+')\nNull gradient: '+str(null_count)+'\nSoft Activation: '+
                      str(soft_count)+'\nSoft and Null: '+str(soft_and_null_count),
                     'ALL IMAGES WITHOUT NULL ('+str(total_images-null_count)+')\nRemoved: '+str(null_count)+'\n\n',
                     'WITHOUT NULL AND SOFT ('+str(total_images-(null_count+soft_count-soft_and_null_count))+
                      ')\nRemoved: '+str(null_count+soft_count-soft_and_null_count),
                     'ONLY SOFT ACTIVATION WITHOUT NULL ('+str(len(soft_index))+')\nRemoved: '+str(total_images-len(soft_index))]

        for i in range(len(hm_list)):
            hm_mask = cv2.resize(hm_list[i], (image_size, image_size))
            
#             hm_mask_to_save = np.uint8(255 * hm_mask)
#             hm_mask_to_save = cv2.applyColorMap(hm_mask_to_save, cv2.COLORMAP_JET)
# #             cv2.imwrite(os.path.join('results', save_path, 'FeatVis_Single_images', model_ID + '_0FeatVis_' + str(i) + '.png'), hm_mask_to_save)
#             # in rendering colors are inverted
#             hm_mask = 255-np.uint8(255 * hm_mask)
#             hm_mask = cv2.applyColorMap(hm_mask, cv2.COLORMAP_JET)
            
#             np_rgba = apply_colormap(hm_mask)  # apply colormap and extract RGBA np.array
#             hm_mask = np.uint8(np_rgba[:,:,:3]*255)  # remove alpha channel (always=1)
#             hm_mask = hm_mask[:,:, [2, 1, 0]]  # RGB to BGR
#             cv2.imwrite(masked_image_path, masked_image)

            if len(np.unique(hm_mask)) > 1:
                pp=ax[i].matshow(hm_mask, vmin=0, vmax=1, cmap=cmap)
                cbar=fig.colorbar(pp, ax=ax[i], ticks=[0, 0.25, 0.5, 0.75, 1])
                cbar.ax.set_yticklabels(ticks_label)
            else:
                ax[i].plot()
                ax[i].axis("off")
            ax[i].set_title(title_list[i])
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            

        if len(box_plot_df) > 0:
            
            # stacked bar plot for changes in predicted label when using soft activation
            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
                      'tab:gray', 'tab:olive', 'tab:cyan']#plt.get_cmap('T10').colors
            patch_handles = []
            # left alignment of data starts at zero
            left = np.zeros(box_plot_df.shape[0])
            y_pos = np.arange(box_plot_df.shape[0])
            y_lab = box_plot_df.meaning.tolist()
            for i, d in enumerate(y_lab):
                patch_handles.append(ax[4].barh(y_pos, box_plot_df[d].values, 
                  color=colors[i], align='center', left=left))
                left += box_plot_df[d].values
            # search all of the bar segments and annotate
            text_label = box_plot_df[y_lab].to_numpy().astype(int)
            for j in range(len(patch_handles)):
                for i, patch in enumerate(patch_handles[j].get_children()):
                    bl = patch.get_xy()
                    x = 0.5*patch.get_width() + bl[0]
                    y = 0.5*patch.get_height() + bl[1]
                    if text_label[i,j] > 0:
                        ax[4].text(x,y, str(text_label[i,j]), ha='center')
            ax[4].legend(patch_handles[::-1], y_lab[::-1], bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=len(y_pos) // 2, fontsize = 14)
            ax[4].set_title('Predicted class with soft activation\n', size = 20)
            ax[4].set_xticks([])
            ax[4].set_xlabel('Soft activation class', size = 14)
            ax[4].set_ylabel('Original activation class', size = 14)
            ax[4].set_yticks(y_pos)
            y_lab_split = [x[:9]+'\n'+x[9:] if len(x) > 9 else x for x in y_lab]
            ax[4].set_yticklabels(y_lab_split, size = 14, fontweight = 'bold')
            for i, tick in enumerate(ax[4].yaxis.get_major_ticks()):
                tick.label1.set_color(colors[::-1][len(colors)-i-1]) #set the color property
            ax[4].set_xlim([0, box_plot_df['count'].max()+5])

            # boxplot of L2 norm for predictions probabilities differences
            data_to_plot = [box_plot_df[box_plot_df.meaning == x].L2diff_list.values[0] for x in y_lab]
            bp = ax[5].boxplot(data_to_plot, vert = False, showmeans=True, meanline=True,patch_artist=True,
                           medianprops=dict(color='black'),
                           meanprops=dict(color='black'))
            for i, box in enumerate(bp['boxes']):
                box.set(facecolor = colors[i] )
            ax[5].set_title('Differences in predicted probabilities\n original vs soft activation', size = 20)
            ax[5].set_xlabel('L2 norm difference', size = 14)
            ax[5].set_yticks(y_pos+1)
            ax[5].set_yticklabels(y_lab_split, size = 14, fontweight = 'bold')
            for i, tick in enumerate(ax[5].yaxis.get_major_ticks()):
                tick.label1.set_color(colors[::-1][len(colors)-i-1])
                ax[5].annotate(str(np.mean(data_to_plot[i]).round(3)[0])+'±'+str(np.std(data_to_plot[i]).round(3)[0]),
                               xy=(np.mean(data_to_plot[i]), i+1.3), ha='center', color=colors[::-1][len(colors)-i-1])

        fig.suptitle('Average of features activations\n', size = 30)
        plt.tight_layout(rect=[0, 0.03, 1, 0.9])
        fig.savefig(os.path.join('results', save_path, model_ID + '_FeatVis_avg.png'))
        if show_fig:
            plt.show()
        else:
            plt.close()
        
        # plot heatmap by class, excluding null and soft activation
        fig, ax = plt.subplots(len(class_heatmap_list), len(class_heatmap_list[0]['true_meaning']),
                               figsize=(fig_size, fig_size), sharey=True, sharex=True)
        ax = ax.flatten()
        ax_i = 0
        for i in range(len(class_heatmap_list)):

            pred_meaning = class_heatmap_list[i]['pred_meaning']
            for j in range(len(class_heatmap_list[0]['true_meaning'])):

                true_meaning = class_heatmap_list[i]['true_meaning'][j]
                images_count = class_heatmap_list[i]['heatmap_count'][j]
                img = class_heatmap_list[i]['heatmap_list'][j]

                hm_mask = cv2.resize(img, (image_size, image_size))
#                 hm_mask_to_save = np.uint8(255 * hm_mask)
#                 hm_mask_to_save = cv2.applyColorMap(hm_mask_to_save, cv2.COLORMAP_JET)
# #                 cv2.imwrite(os.path.join('results', save_path, 'FeatVis_Single_images', model_ID + '_0FeatVis_' +
# #                                          str(i) + 'vs' + str(j) + '.png'), hm_mask_to_save)
#                 # in rendering colors are inverted
#                 hm_mask = 255-np.uint8(255 * hm_mask)
#                 hm_mask = cv2.applyColorMap(hm_mask, cv2.COLORMAP_JET)
                if images_count > 0:
                    pp=ax[ax_i].matshow(hm_mask, vmin=0, vmax=1, cmap=cmap)
                    cbar=fig.colorbar(pp, ax=ax[ax_i], ticks=[0, 0.25, 0.5, 0.75, 1])
                    cbar.ax.set_yticklabels(ticks_label)
                else:
                    ax[ax_i].plot()
                    ax[ax_i].axis("off")
                ax[ax_i].set_title(('True:\n' + true_meaning + '\n' if i == 0 else '')+ '\n(' + str(images_count) + ')', size=fig_size*1.7)
                if j == 0:
                    ax[ax_i].set_ylabel('Predicted:\n' + pred_meaning + '\n\n(' + str(sum(class_heatmap_list[i]['heatmap_count'])) + ')', size=fig_size*1.7)
                ax[ax_i].set_xticks([])
                ax[ax_i].set_yticks([])
                
                ax_i += 1

        fig.suptitle(by_class_title, size = fig_size*2.6)
        plt.tight_layout(rect=[0, 0.03, 1, 0.9])
        fig.savefig(os.path.join('results', save_path, model_ID + '_FeatVis_by_class.png'))
        if show_fig:
            plt.show()
        else:
            plt.close()

        if len(box_plot_df) > 0:
            print('Data used for bar and box plot')
            display(box_plot_df)

        return box_plot_df
    
    
    @staticmethod
    def _plot_sample_masked(final_log, save_path, model_ID, max_sample_combination = 10, ncol = 4,
                            fig_size=12, cmap=plt.cm.RdBu, heatmap_transformation='ReLU', show_fig=True):
        '''
        Plot samples of masked images, taking all possible combination of true class vs predicted class. Null and soft activation are excluded

        Args:
            - final_log: output of _evaluate_heatmap()
            - save_path, model_ID: save path for figures and model_ID used as prefix
            - max_sample_combination: max number of sample for each combination (if available)
            - ncol: number of images per row
            - fig_size: overall figures size (y shape is automatically adjusted)
            - cmap: colormap used for colorbar
            - heatmap_transformation: final tranformation to be applied to heatmap. Used to define ticks of colorbar only
        '''
        
        if heatmap_transformation == 'ReLU':
            ticks_label = ['0', '0.25', '0.5', '0.75', '1']
        if heatmap_transformation in ['linear', 'exponential']:
            ticks_label = ['-1', '-0.5', '0', '0.5', '1']
            
        plot_df = final_log[(final_log.null_heatmap == 'No') & (final_log.soft_activation == 'No')]\
                    .groupby(['label', 'pred_label']).head(max_sample_combination).sort_values(by=['label', 'pred_label'])
        del_path = []
        multiclass = True if plot_df.meaning.nunique() > 2 else False
        for meaning in plot_df.meaning.unique():
            meaning_df = pd.concat([plot_df[(plot_df.meaning == meaning) & (plot_df.pred_meaning == meaning)],
                                 plot_df[(plot_df.meaning == meaning) & (plot_df.pred_meaning != meaning)]], axis=0).reset_index()
            true_label = meaning_df.label.unique()

            nrow = math.ceil(meaning_df.shape[0] / ncol)
            fig, ax = plt.subplots(nrow, ncol, figsize=(fig_size, fig_size*nrow/ncol), sharey=True, sharex=True)
            ax = ax.flatten()
            last_i = meaning_df.shape[0]-1 
            for i, row in meaning_df.iterrows():
                pred_label = row['pred_meaning']
                image_path = row['masked_image_path']
                image_name = os.path.split(image_path)[1].replace(model_ID+'_','').split('.')[0]
                pp=ax[i].imshow(plt.imread(image_path), vmin=0, vmax=1, cmap=cmap)
                ax[i].set_title('Pred:' + pred_label + ' [' + str(int(row.pred_perc)) + '%]\n' + image_name,
                                color='red' if pred_label!=meaning else 'black')
                ax[i].set_xticks([])
                ax[i].set_yticks([])
                if (i % ncol == (ncol-1)) or (i == last_i):
                    cbar=fig.colorbar(pp, ax=ax[i], ticks=[0, 0.25, 0.5, 0.75, 1])
                    cbar.ax.set_yticklabels(ticks_label)
            # delete empy odd subplot
            if last_i < nrow*ncol:
                [fig.delaxes(ax[x]) for x in range(last_i+1, nrow*ncol)]
            fig.suptitle('True class: ' + meaning + (' ['+str(int(true_label*100))+'%]' if not multiclass else ''), size = 30)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if show_fig:
                plt.show()
            else:
                plt.close()
            fig_path = os.path.join('results', save_path, 'FeatVis_Single_images', model_ID + '_0sample_'+meaning+'.png')
            fig.savefig(fig_path)
            del_path.append(fig_path)

        # concatenate images for each "meaning" (true class) and delete the single ones
        get_concat_v([Image.open(x) for x in del_path]).save(os.path.join('results', save_path, model_ID + '_FeatVis_sample.png'))
        _=[os.remove(x) for x in del_path]
        
    
    @staticmethod
    def _plot_heatmap_max_distribution(final_log, save_path, model_ID, ncol = 2, fig_size = 10, show_fig=True):
        '''
        Plot distribution for maximum value of each heatmap
        '''
        def format_print(x, digits=3):
            magn = math.log10(abs(x))
            round_val=int(-np.sign(magn)*math.ceil(abs(math.log10(abs(x))))+digits)
            return str(round(x, round_val))

        del_path = []
        for heatmap_type in ['max', 'min']:
            
            heatmap_distr = final_log.groupby(['source', 'pred_meaning', 'soft_activation', 'null_heatmap']).agg(
                heatmap_Avg=('heatmap_'+heatmap_type, 'mean'),
                heatmap_Min=('heatmap_'+heatmap_type, 'min'),
                heatmap_Max=('heatmap_'+heatmap_type, 'max'),
                heatmap_5_quantile=('heatmap_'+heatmap_type, lambda x: np.quantile(x, 0.05)),
                heatmap_95_quantile=('heatmap_'+heatmap_type, lambda x: np.quantile(x, 0.95)),
                count=('heatmap_'+heatmap_type, lambda x: len(x))
            )

            heatmap_distr_plot = heatmap_distr.reset_index(drop=False)
            heatmap_distr_plot = heatmap_distr_plot[(heatmap_distr_plot.null_heatmap == 'No') & (heatmap_distr_plot.count != 0)]

            nrow = math.ceil(heatmap_distr_plot.shape[0] / ncol)
            fig, ax = plt.subplots(nrow, ncol, figsize=(fig_size, fig_size * nrow / ncol), sharey=True, sharex=True)
            ax = ax.flatten()

            for i in range(heatmap_distr_plot.shape[0]):
                names = heatmap_distr_plot.iloc[i]
                source = names.source
                pred_meaning = names.pred_meaning
                soft_activation = names.soft_activation
                avg = names['heatmap_Avg']
                vmin = names['heatmap_Min']
                vmax = names['heatmap_Max']
                v5 = names['heatmap_5_quantile']
                v95 = names['heatmap_95_quantile']
                count = names['count']

                bar_data=final_log[(final_log.source == source) & (final_log.pred_meaning == pred_meaning) &(final_log.soft_activation == soft_activation) & (final_log.null_heatmap == 'No')]['heatmap_'+heatmap_type].values

                if max(abs(bar_data)) > 0:
                    n, bins, patches = ax[i].hist(bar_data, len(bar_data), density = True)
                    ax[i].tick_params(axis='x', labelrotation=45)
                    ax[i].set_xlabel('Heatmap '+heatmap_type) 
                    ax[i].set_ylabel('Density') 
                    ax[i].set_title('Source:'+source+'\nPrediction:'+pred_meaning+' ('+str(int(count))+')'+
                                    '\nSoft Activation:'+soft_activation+
                                    '\nAverage:'+format_print(avg)+'\nMin-Max:'+format_print(vmin)+' - '+format_print(vmax)+
                                   '\n5%-95%:'+format_print(v5)+' - '+format_print(v95))

            fig.suptitle('Distribution of '+heatmap_type+' value for each heatmap', size = 30)
            plt.tight_layout(rect=[0, 0.03, 1, 0.9])
            if show_fig:
                plt.show()
            else:
                plt.close()
            fig_path = os.path.join('results', save_path, model_ID + '_FeatVis_max_heatmap_distr_'+heatmap_type+'.png')
            fig.savefig(fig_path)
            del_path.append(fig_path)
        get_concat_v([Image.open(x) for x in del_path]).save(os.path.join('results', save_path, model_ID + '_FeatVis_max_heatmap_distr.png'))
        _=[os.remove(x) for x in del_path]    

    
    @staticmethod
    def _replace_modules(model, nn_to_replace = nn.ReLU, nn_replacement= nn.LeakyReLU()):
        '''
        replaces all modules in model
        '''
        
        for child_name, child in model.named_children():
            if isinstance(child, nn_to_replace):
                setattr(model, child_name, nn_replacement)
            else:
                FeaturesVisualizationUtils._replace_modules(child)
                
                
class FilterVisualizationUtils:
    def __init__(self, save_path = '', model_ID = ''):
        '''
        Args:
            - save_path, model_ID: save path for figures and model_ID used as prefix
        '''
        
        self.save_path = save_path
        self.model_ID = model_ID
        
        
    def extract_filters(self, model, filter_type = nn.Conv2d):
        '''
        Extract and print list of filters with filter_type instance
        '''
        
        self.weight_list = []
        print('Filters found (1st dim is # of filter, 2nd is # of channels, 3rd-4th is filter size):\n')
        for i, (mod_name, mod) in enumerate(model.named_modules()):

            if isinstance(mod, filter_type):   # same of if (type(mod) == filter_type):
                print(i, mod_name, mod.weight.data.shape)
                self.weight_list.append({'layer': mod_name,
                                    'weight': mod.weight.data.cpu()})
     
    
    def plot_filters(self, minimum_filter_size = 5, fig_size = 12, ncol = 12, color_map = plt.cm.Greys):
        '''
        Plot filter for each layer. If number of channels is 3, RGB plot is added
        
        Args:
            - minimum_filter_size: minimum filter size (one of the 2) to select filters to plot
            - fig_size: overall figures size (y shape is automatically adjusted)
            - ncol: number of images per row
            - color_map: color map to be used for single channel
        '''
        
        # take only weights with size >= minimum_filter_size (1st dim is # of filter, 2nd is # of channels, 3rd-4th is filter size)
        final_list = [x for x in self.weight_list if max(x['weight'].size()[2:4]) >= minimum_filter_size]
        print('\nTotal filters with size >= ' + str(minimum_filter_size) + ':', len(final_list), '\n')
        del_path = []
        for i, filt in enumerate(final_list):

            print('Evaluating '+str(i+1)+'/'+str(len(final_list))+'...', end='')
            weight = filt['weight']
            layer_name = filt['layer']
            num_filt = weight.size(0)
            num_chann = weight.size(1)
            total_filt = num_filt * (num_chann + 1 if num_chann == 3 else num_chann) # add also RBG composition when 3 channels

            nrow = math.ceil(total_filt / ncol)
            fig, ax = plt.subplots(nrow, ncol, figsize=(fig_size, fig_size*nrow/ncol), sharey=True, sharex=True)
            ax = ax.flatten()
            ax_c = 0
            for fi in range(num_filt):

                # if num_chann == 3, plot RGB filter first
                if num_chann == 3:
                    npimg = np.array(weight[fi].numpy(), np.float32)
                    #standardize the numpy image
                    npimg = (npimg - np.mean(npimg)) / np.std(npimg)
                    npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
                    npimg = npimg.transpose((1, 2, 0))
                    ax[ax_c].imshow(npimg, cmap=color_map)
                    ax[ax_c].axis('off')
                    ax[ax_c].set_title(str(fi))
                    ax[ax_c].set_xticklabels([])
                    ax[ax_c].set_yticklabels([])
                    ax_c += 1

                # plot single channels
                for ch in range(num_chann):

                    npimg = np.array(weight[fi, ch].numpy(), np.float32)
                    #standardize the numpy image
                    npimg = (npimg - np.mean(npimg)) / np.std(npimg)
                    npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
                    ax[ax_c].imshow(npimg, cmap=color_map)
                    ax[ax_c].axis('off')
                    ax[ax_c].set_title(str(fi) + ',' + str(ch))
                    ax[ax_c].set_xticklabels([])
                    ax[ax_c].set_yticklabels([])
                    ax_c += 1

            if total_filt % ncol > 0:
                [fig.delaxes(ax[x]) for x in range(ax_c, nrow*ncol)]   
            fig.suptitle('Layer: ' + layer_name, size = 25)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.close(fig)
            fig_path = os.path.join('results', self.save_path, self.model_ID + '_Filter_'+ str(i)+'.png')
            fig.savefig(fig_path)
            del_path.append(fig_path)
            print('Done')

        # concatenate images for each "meaning" (true class) and delete the single ones
        merge_path = os.path.join('results', self.save_path, self.model_ID + '_FilterVis.png')
        get_concat_v([Image.open(x) for x in del_path]).save(merge_path)
        _=[os.remove(x) for x in del_path]
        display(Image.open(merge_path))
        
        
    def evaluate_weight_distribution(self, ref_model, model, add_title='', n_bins = 40,
                                      quantile_cutoff = 0.05, zero_toll = 1e-8):
        '''
        Evaluate weight % changes distribution with respect to a reference model (both of the same structure).
        % change = ((weight - ref_weight) / ref_weight * 100)
        
        Args:
            - ref_model: reference model
            - model: model to compare with
            - add_title: additional title on top of plots
            - n_bins: number of bins to evaluate in histogram. A cap of 20% of total count is applied
            - quantile_cutoff: exclude outliers from plots with a cut-off on both sides
            - zero_toll: weights that became "zero" are also displayed. Weight is considered zero if abs(weight) < zero_toll
        '''
        
        self.add_title = add_title
        self.n_bins = n_bins
        self.quantile_cutoff = quantile_cutoff
        self.zero_toll = zero_toll
                
        # get weights
        model_params = []
        for name, param in model.named_parameters():
            if np.prod(param.size()) > 1:
                model_params.append({'layer': name,
                                     'weight': param.data})
        ref_model_params = []
        for name, param in ref_model.named_parameters():
            if np.prod(param.size()) > 1:
                ref_model_params.append({'layer': name,
                                         'weight': param.data})

        # check for consistency
        model_names = [(x['layer'], x['weight'].size()) for x in model_params]
        ref_model_names = [(x['layer'], x['weight'].size()) for x in ref_model_params]
        if (model_names != ref_model_names):
            raise ValueError('Model and Reference Model structures don\'t match')

        data_joy_df = pd.DataFrame(columns = ['layer', 'bin_count'], dtype=float).fillna('')
        data_bubble_df = pd.DataFrame(columns = ['layer', 'bin', 'count'], dtype=float).fillna('')
        data_bubble_stats = pd.DataFrame(columns = ['layer', 'min', 'max', 'avg', 'low_quantile', 'up_quantile', 'tot_count'], dtype=float).fillna('')

        for i in range(len(model_params)):

            weight = model_params[i]['weight'].cpu().numpy().ravel()
            ref_weight = ref_model_params[i]['weight'].cpu().numpy().ravel()
            with np.errstate(divide='ignore'):
                perc_diff = ((weight - ref_weight) / ref_weight * 100)#.ravel()
            perc_diff[np.isinf(perc_diff)] = 100 # replace x/0 with 100%
            _=np.nan_to_num(perc_diff, nan=0,copy=False) # replace 0/0 with 0%

            # take full distribution for bubble
        #     dist=sns.distplot(x=perc_diff, bins=min((n_bins, round(len(perc_diff) * 0.2))), # get counts
        #                    hist=True, kde=False)
        #     plt.close()
            data_bubble_stats = data_bubble_stats.append(
                pd.DataFrame({'layer': i,
                              'min': perc_diff.min(),
                              'max': perc_diff.max(),
                              'avg': perc_diff.mean(),
                              'low_quantile': np.quantile(perc_diff, quantile_cutoff),
                              'up_quantile': np.quantile(perc_diff, 1-quantile_cutoff),
                              'tot_count': len(perc_diff)}, index=[0])
            )

            # cut-off outliers for joyplot
            quantile_cut = np.quantile(perc_diff, [quantile_cutoff, 1-quantile_cutoff])
            quantile_mask = (perc_diff >= quantile_cut[0]) & (perc_diff <= quantile_cut[1])
            perc_diff_quant = perc_diff[quantile_mask]
            weight_quant = weight[quantile_mask]
            ref_weight_quant = ref_weight[quantile_mask]
            dist_quant=sns.distplot(x=perc_diff_quant, bins=min((n_bins, round(len(perc_diff_quant) * 0.2))), # get counts
                           hist=True, kde=False)
            plt.close()

            # extract bins and normalize counts in order to prepare data for joyplot
            # each frequency is repeated proportionally to itself for a total of 100 points (apart from approx),
            # so that corresponding frequency value will have the right proportion in the final histogram/distribution
            data_joy_df = data_joy_df.append(
                pd.DataFrame({'layer': i,
                              'bin_count': np.concatenate([np.repeat(x.get_x()+ x.get_width()/2,
                                                                     math.ceil(x.get_height() / len(perc_diff_quant) * 100)) for x in dist_quant.patches]).ravel()})
            )
            # extract bins and count for bubble plot. Add count of weights that where abs()>zero_toll and became abs()<zero_toll
            cut_point=[x.get_x() for x in dist_quant.patches]
            zero_count = []
            for p in range(len(cut_point)):
                if p == 0:
                    ind = np.where(perc_diff_quant <= cut_point[p])
                elif p == len(cut_point)-1:
                    ind = np.where(perc_diff_quant > cut_point[p])
                else:
                    ind = np.where((perc_diff_quant <= cut_point[p]) & (perc_diff_quant > cut_point[p-1]))
                zero_weight = weight_quant[ind]
                zero_ref_weight = ref_weight_quant[ind]
                zero_count.append(sum((abs(zero_ref_weight) > zero_toll) & (abs(zero_weight) < zero_toll)))
            data_bubble_df = data_bubble_df.append(
                pd.DataFrame({'layer': i,
                              'bin': [x.get_x()+ x.get_width()/2 for x in dist_quant.patches],
                              'count': [x.get_height() for x in dist_quant.patches],
                              'zero_count': zero_count})
            )
            
            self.model_names = model_names
            self.data_joy_df = data_joy_df
            self.data_bubble_df = data_bubble_df
            self.data_bubble_stats = data_bubble_stats
            
            
    def plot_distribution(self):
        '''
        Plot distribution as joyplot https://sbebo.github.io/posts/2017/08/01/joypy/
        '''
        
        # plot joyplot
        fig, axes = joypy.joyplot(self.data_joy_df, by="layer", column="bin_count", range_style='own', bins=self.n_bins,
                                  grid="both", linewidth=1, legend=False, figsize=(10, len(self.model_names) / 2), hist=True,
                                  labels = [x[0] for x in self.model_names], overlap=0,
                                  title="% change of weights - "+self.add_title)
        for i in range(len(axes)-1):
            axes[i].annotate('0', xy=(0, 0), ha='center', va='bottom')
        _=axes[-1].set_xticklabels(['{:,}'.format(int(x))+'%' for x in axes[-1].get_xticks()])
        
        plt.show()
        fig.savefig(os.path.join('results', self.save_path, self.model_ID + '_Weight_Distr_hist.png'))
        
        
    def plot_bubble(self):
        '''
        Plot distribution as bubbles
        '''
        
        x_size = 15.0
        y_size = len(self.model_names) / 1.5
        max_marker_size = y_size / 30

        fig,ax = plt.subplots(figsize=(x_size, y_size))
        y_ref = 2
        yticks = []
        x_lim = [self.data_bubble_df.bin.min(), self.data_bubble_df.bin.max()]
        for i in range(len(self.model_names)):

            data_plot = self.data_bubble_df[self.data_bubble_df.layer==i]
            data_stats = self.data_bubble_stats[self.data_bubble_stats.layer==i]
            y_ref -= 2 #len(self.model_names) - i
            yticks.append(y_ref)
            total_count = sum(data_plot['count'])

        #     ax.plot(x_lim, [y_ref, y_ref], 'k-.', alpha=0.2)
            ax.annotate('min:'+'{:,}'.format(int(data_stats['min'].values[0]))+'%\n'+
                        str(int(self.quantile_cutoff*100))+'% quant:'+'{:,}'.format(int(data_stats['low_quantile'].values[0]))+'%',
                        xy=(x_lim[0], y_ref), ha='left', va='center')
            ax.annotate('max:'+'{:,}'.format(int(data_stats['max'].values[0]))+'%\n'+
                        str(int((1-self.quantile_cutoff)*100))+'% quant:'+'{:,}'.format(int(data_stats['up_quantile'].values[0]))+'%',
                        xy=(x_lim[1], y_ref), ha='right', va='center')
            ax.annotate('avg:'+'{:,}'.format(int(data_stats['avg'].values[0]))+'%',
                        xy=(data_stats['avg'].values[0], y_ref+0.8), ha='center', va='center')

            for p in range(len(data_plot)):
                # plot buble for all count
                ax.plot(data_plot.bin[p], y_ref+0.2, marker="o", color='b', alpha=0.5,
                        markersize=data_plot['count'][p] / total_count * 100 / max_marker_size, linewidth=0)
                # plot buble for zero count
                ax.plot(data_plot.bin[p], y_ref-0.2, marker="o", color='r', alpha=1,
                        markersize=data_plot['zero_count'][p] / total_count * 100 / max_marker_size, linewidth=0)

        ax.plot([0, 0], [2, y_ref-2], 'k-', alpha=0.7)
        ax.grid(which='major', axis='both',)
        ax.set_ylim([y_ref-2, 2])
        _=ax.set_yticks(yticks)
        _=ax.set_yticklabels([str(i)+' - '+x[0]+'\n'+'{:,}'.format(np.prod(x[1]))+' params' for i, x in enumerate(self.model_names)])
        xticks = np.unique(np.concatenate((np.linspace(x_lim[0],0,5), np.linspace(0,x_lim[1],5))))
        _=ax.set_xticks(xticks)
        _=ax.set_xticklabels(['{:,}'.format(int(x))+'%' for x in xticks])
        ax.tick_params(labelbottom=True, labeltop=True, bottom=True, top=True)
        legend_elements = [Line2D([0], [0], color='b', lw=0, marker='o', markerfacecolor='b',
                                  markersize=20, label='all weights'),
                           Line2D([0], [0], color='r', lw=0, marker='o', markerfacecolor='r',
                                  markersize=20, label='abs(weight) that became\nsmaller than ' + str(self.zero_toll))]
        fig.legend(handles=legend_elements, ncol=2, bbox_to_anchor=(0.5, 0.01), loc='center', fontsize = 14)
        _=ax.set_title("% change of weights - "+self.add_title, fontsize=25)
#         fig.tight_layout(rect=[0, 0.03, 0, 0])
        plt.subplots_adjust(left=3/x_size, right=1-1/x_size, bottom=2/y_size, top=1-1/y_size)

        plt.show()
        fig.savefig(os.path.join('results', self.save_path, self.model_ID + '_Weight_Distr_bubble.png'))
