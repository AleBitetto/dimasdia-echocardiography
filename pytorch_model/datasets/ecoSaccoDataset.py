import os
import cv2
import pandas as pd
import csv
import numpy as np
import copy
from timeit import default_timer as timer
import datetime
import pickle
from torch.utils.data import Dataset

def create_model_input(metadata, output_folder = './sacco/PKL/model_input/', metadata_out = './sacco/Model_MetaData.csv',
                       frame_to_keep = 50, frame_sampling = 'first_available',
                       apply_frame_difference = True, apply_sampling_first = True,
                      apply_mask = True, resize = None):
    '''
    Args:
        - frame_to_keep: total frames to keep
        - frame_sampling: 'first_available' to keep the first frame_to_keep frames,
                          'equally_spaced' to sample equally-spaced frame_to_keep frames
        - apply_frame_difference: if True applies frames difference instead of raw frames
        - apply_sampling_first: is True, applies sampling first and then difference (if any). If False, in reversed order
        - apply_mask: applies mask to frames
        - resize: resizes frames in square shape
    '''

    if frame_sampling not in ['first_available', 'equally_spaced']:
        raise ValueError('Please provide frame_sampling as \'first_available\' or \'equally_spaced\' - current is: ' + frame_sampling)
    _=os.makedirs(output_folder, exist_ok=True)
    total_files = metadata.shape[0]
    removed_count = sum(metadata.keep == 'Remove')

    # filter by number of frames
    working_df = metadata[(metadata['keep'] == 'Keep') & (metadata.NumberOfFrames >= frame_to_keep)]
    print('\nTotal removed:', total_files-working_df.shape[0])
    print('    - because of color:', removed_count)
    print('    - because of number of frames:', total_files-working_df.shape[0] - removed_count)
    print('\nTotal remaining:', working_df.shape[0], 'out of', total_files, '\n')
    display(working_df.groupby(['covid']).size().to_frame().add_prefix('X_').rename(columns={'X_0': 'count'}).\
            join(metadata.groupby(['covid']).size().to_frame().add_prefix('X_').rename(columns={'X_0': 'out of'})))
    display(working_df.groupby(['akinetic']).size().to_frame().add_prefix('X_').rename(columns={'X_0': 'count'}).\
            join(metadata.groupby(['akinetic']).size().to_frame().add_prefix('X_').rename(columns={'X_0': 'out of'})))

    path = os.path.dirname(metadata.index_pickle[0])
    folder_size = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))) / 2**20
    print('\nRaw pickle folder size:', '{:,}'.format(round(folder_size)), 'MB')
    print('\nTotal number of frames will be:', frame_to_keep-1 if apply_frame_difference else frame_to_keep, '\n')


    # select frames sampling for files with more than frame_to_keep frames
    def sample_frames(frame_array, total_frames, frame_sampling):

        if frame_sampling == 'first_available':
            data_out = frame_array[:total_frames]
        elif frame_sampling == 'equally_spaced':
            data_out = frame_array[np.linspace(0, frame_array.shape[0] - 1, total_frames).astype(int)]
            print(np.linspace(0, frame_array.shape[0] - 1, total_frames).astype(int))

        return data_out

    metadata_model = pd.DataFrame(columns = ['index_pickle_final'] + list(working_df.columns), dtype=int).fillna(0)
    start = timer()
    for i, row in working_df.iterrows():

        filename = row.filename
        print(i+1, '/', metadata.shape[0], end = '\r')
        file_path = os.path.join(output_folder, filename.replace('.dcm', '.pickle'))

        with open(row.index_pickle, 'rb') as handle:
            data_gray = pickle.load(handle)

        # frame sampling and frame difference
        if apply_sampling_first:

            data_out = sample_frames(frame_array=data_gray,
                                     total_frames=frame_to_keep,
                                     frame_sampling=frame_sampling)
            if apply_frame_difference:
                data_out = data_out[1:] - data_out[:-1]

        else:

            if apply_frame_difference:
                data_out = data_gray[1:] - data_gray[:-1]
            data_out = sample_frames(frame_array=data_out,
                                     total_frames=frame_to_keep-1 if apply_frame_difference else frame_to_keep,
                                     frame_sampling=frame_sampling)

        # apply mask
        if apply_mask:
            with open(row.index_mask, 'rb') as handle:
                mask = pickle.load(handle)

            for f in range(data_out.shape[0]):
                data_out[f][mask] = 0

        # resize
        if resize is not None:
            data_temp = copy.deepcopy(data_out)
            data_out = np.zeros((data_out.shape[0], resize, resize))
            for f in range(data_temp.shape[0]):
                data_out[f] = cv2.resize(data_temp[f], (resize, resize), interpolation=cv2.INTER_AREA)

        # convert to uint8
        data_out = data_out.astype(np.uint8)
                
        # save pickle        
        with open(file_path, 'wb') as handle:
            pickle.dump(data_out, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        # update metadata_model
        row_to_add = copy.deepcopy(row)
        row_to_add['index_pickle_final'] = file_path
        row_to_add['NumberOfFrames_final'] = data_out.shape[0]
        metadata_model = metadata_model.append(row_to_add)
        
    print('\nTotal elapsed time:', str(datetime.timedelta(seconds=round(timer()-start))))
    path = output_folder
    folder_size = sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))) / 2**20
    print('\nFiles saved in', output_folder)
    print('Output folder size:', '{:,}'.format(round(folder_size)), 'MB')

    # reorder columns and save csv
    first_columns = ['keep', 'folder', 'index_pickle_final', 'NumberOfFrames_final', 'covid', 'akinetic', 'index_pickle', 'index_mask']
    metadata_model = metadata_model[first_columns + list(set(metadata_model.columns) - set(first_columns))]
    
    metadata_model.to_csv(metadata_out, index=False)
    print('\nModel metadata saved in', metadata_out)
    
    
class ecoSaccoDataset(Dataset):
    def __init__(self, input_name, target_class='', size=128, augment=None, show_info=True):
        '''
        - target_class: is the binary variable to be used as target. 'covid' or 'akinetic'
        '''

        super(ecoSaccoDataset, self).__init__()
        if show_info:
            print('{} initialized with size={}, augment={}'.format(self.__class__.__name__, size, augment))
            print('Dataset is located in {}'.format(os.path.join('./input/', input_name)))
        self.size = size
        self.augment = augment
        
        df_metadata = pd.read_csv(os.path.join('./input/', 'MetaData_' + input_name + '.csv'))
        
        # add label, meaning and source
        df_metadata['source'] = 'Sacco'
        df_metadata['image'] = df_metadata.index_pickle_final
        total_frames = int(df_metadata.NumberOfFrames_final.unique())
        self.total_frames = total_frames
                    
        # final dataset and labels
        self.labels = df_metadata[target_class].astype(float).values.reshape(-1, 1)
        self.df = df_metadata[['image', 'source', target_class]].rename(columns={target_class: 'label'})
        self.df['meaning'] = np.where(self.df.label == 1, target_class, 'not_'+target_class)
        self.df = self.df[['image', 'label', 'meaning', 'source']]
        self.df = self.df.reset_index(drop = True)

        if show_info:
            print('\n############    '+input_name+' dataset    ############')
            print('\nTotal files:', df_metadata.shape[0])
            print('Total frames:', total_frames)
            display(self.df)
            display(self.df.groupby(['label', 'meaning', 'source']).size().to_frame().add_prefix('X_').rename(columns={'X_0': 'count'}))

            
                    
    @staticmethod
    def _load_image(image, size):
                           
        with open(image, 'rb') as handle:
            img = pickle.load(handle)
                    
        orig_size = img.shape[1]   # img is (frames, size, size)
        if size != orig_size:
            print('resizing')
            img_temp = copy.deepcopy(img)
            img = np.zeros((img_temp.shape[0], size, size))
            for f in range(img_temp.shape[0]):
                img[f] = cv2.resize(img[f], (size, size), interpolation=cv2.INTER_AREA)
       
        # format is already (chann=frames, size, size)
       
        return img
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = self._load_image(row['image'], self.size)
        label = row['label']        

        if self.augment is not None:
            img = self.augment(img)

        return img, label, index

    def __len__(self):
        return self.df.shape[0]
    
    
# class ecoSaccoDataset(Dataset):
#     def __init__(self, input_name, diagnosis_class=None, diagnosis_to_keep=['akinetic', 'healthy', 'covid'],
#                  size=128, augment=None, show_info=True):
#         '''
#         diagnosis_class: dataframe with columns 'diagnosis', 'label', 'meaning' used to code final classes.
#                          if None, unique values from df_metadata.diagnosis will be used
#         '''

#         super(ecoSaccoDataset, self).__init__()
#         if show_info:
#             print('{} initialized with size={}, augment={}'.format(self.__class__.__name__, size, augment))
#             print('Dataset is located in {}'.format(os.path.join('./input/', input_name)))
#         self.size = size
#         self.augment = augment
        
#         df_metadata = pd.read_csv(os.path.join('./input/', 'MetaData_' + input_name + '.csv'))
        
#         # filter diagnosis to keep
#         df_metadata = df_metadata[df_metadata['diagnosis'].isin(diagnosis_to_keep)]
        
#         # create label code (if not provided)
#         if diagnosis_class is None:
#             avail_class = df_metadata.diagnosis.unique()
#             diagnosis_class = pd.DataFrame({'diagnosis': avail_class, 'label': range(len(avail_class)), 'meaning': avail_class})
            
#         # add label, meaning and source
#         df_metadata = pd.merge(df_metadata, diagnosis_class, left_on='diagnosis', right_on='diagnosis')
#         df_metadata['source'] = 'Sacco'
#         df_metadata['image'] = df_metadata.index_pickle_final
#         total_frames = int(df_metadata.NumberOfFrames_final.unique())
                    
#         # final dataset and labels
#         self.labels = df_metadata.label.astype(float)
#         self.df = df_metadata[['image', 'label', 'meaning', 'source']]
#         self.df = self.df.reset_index(drop = True)

#         if show_info:
#             print('\n############    '+input_name+' dataset    ############')
#             print('\nTotal files:', df_metadata.shape[0])
#             print('Total frames:', total_frames)
#             display(self.df)
#             display(self.df.groupby(['label', 'meaning', 'source']).size().to_frame().add_prefix('X_').rename(columns={'X_0': 'count'}))

            
                    
#     @staticmethod
#     def _load_image(image, size):
                           
#         with open(image, 'rb') as handle:
#             img = pickle.load(handle)
                    
#         orig_size = img.shape[1]   # img is (frames, size, size)
#         if size != orig_size:
#             print('resizing')
#             img_temp = copy.deepcopy(img)
#             img = np.zeros((img_temp.shape[0], size, size))
#             for f in range(img_temp.shape[0]):
#                 img[f] = cv2.resize(img[f], (size, size), interpolation=cv2.INTER_AREA)
       
#         # format is already (chann=frames, size, size)
       
#         return img
    
#     def __getitem__(self, index):
#         row = self.df.iloc[index]
#         img = self._load_image(row['image'], self.size)
#         label = row['label']        

#         if self.augment is not None:
#             img = self.augment(img)

#         return img, label, index

#     def __len__(self):
#         return self.df.shape[0]