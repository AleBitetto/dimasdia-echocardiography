import cv2
import pandas as pd
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset

# include IAI-anonimizzati

class COVIDChestXRayExtendedDataset(Dataset):
    def __init__(self, path, path_IAI, allowed_views = ['PA', 'AP', 'AP Supine'], new_batch=True, only_IAI=False,
                 size=128, augment=None, show_info=True):
        '''
            new_batch: if True takes all images from IAI, if False takes only first batch (181 images, all COVID positive)
        '''

        super(COVIDChestXRayExtendedDataset, self).__init__()
        if show_info:
            print('{} initialized with size={}, augment={}'.format(self.__class__.__name__, size, augment))
            print('Dataset are located in {} and {}'.format(path, path_IAI))
        self.size = size
        self.augment = augment
        
        
        #### ieee8023 dataset
        modality_keep = 'X-ray'
        finding_keep = 'COVID'   # will keep all finding that contain finding_keep
        
        image_dir = path / 'images'
        metadata_path = path / 'metadata.csv'
        
        if show_info and not only_IAI: print('\n############    IEEE8023 dataset    ############')
        df_metadata = pd.read_csv(metadata_path, header=0)
        # check available modality, view and findings and make a report of kept and left observation
        check = df_metadata['modality'].value_counts().to_frame()
        check.columns = 'Available ' + check.columns
        check['keep'] = np.where(check.index == modality_keep, 'yes', '')
        if show_info and not only_IAI: display(check)
        check = df_metadata['view'].value_counts().to_frame()
        check.columns = 'Available ' + check.columns
        check['keep'] = np.where(check.index.isin(allowed_views), 'yes', '')
        if show_info and not only_IAI: display(check)
        check = df_metadata['finding'].value_counts().to_frame()
        check.columns = 'Available ' + check.columns
#         check['keep'] = np.where(check.index.str.contains(finding_keep), 'yes', '')
        if show_info and not only_IAI: display(check)
        
        # Drop CT scans
        df_metadata = df_metadata[df_metadata['modality'] == modality_keep]
        # Keep only allowed_views
        df_metadata = df_metadata[df_metadata['view'].isin(allowed_views)]
        
        # COVID-19 = 1, SARS/ARDS/Pneumocystis/Streptococcus/No finding = 0
        labels = (df_metadata.finding.str.contains(finding_keep)).values.reshape(-1, 1)
        images = df_metadata.filename
        images = images.apply(lambda x: image_dir / x).values.reshape(-1, 1)
        
        
        #### IAI dataset
        IAI_image_path = path_IAI / 'Edited'
        IAI_metadata_path = path_IAI / 'Patient_MetaData.csv'
        
        if show_info: print('\n############    IAI dataset    ############')
        IAI_metadata = pd.read_csv(IAI_metadata_path, header=0)
        check = IAI_metadata['ViewPosition'].value_counts().to_frame()
        check.columns = 'Available ' + check.columns
        check['keep'] = np.where(check.index.isin(allowed_views), 'yes', '')
        if show_info: display(check)
        check = IAI_metadata['Outcome'].value_counts().to_frame()
        check.columns = 'Available ' + check.columns
        if show_info: display(check)
        
        # keep old batch or all images
        if new_batch == False:
            IAI_metadata = IAI_metadata[IAI_metadata.Batch == 'old_batch']
        
        # keep only Remove == 'No'
        IAI_metadata = IAI_metadata[IAI_metadata.Remove == 'No']
        
        # Keep only allowed_views
        IAI_metadata = IAI_metadata[IAI_metadata['ViewPosition'].isin(allowed_views)]
        
        # covid19 = 1, healthy = 0
        images_IAI = IAI_metadata.Image
        images_IAI = images_IAI.apply(lambda x: IAI_image_path / x).values.reshape(-1, 1)
        labels_IAI = np.where(IAI_metadata.Outcome == 'covid19', 1, 0).reshape(-1, 1)
        
        # merge dataset
        images = np.vstack((images, images_IAI))
        self.labels = np.vstack((labels, labels_IAI)).astype(float)
       
        self.df = pd.DataFrame(np.concatenate((images, self.labels), axis=1), columns=['image', 'label'])
        self.df['meaning'] = np.where(self.df['label']==0, 'other', 'COVID')
        self.df['source'] = np.where(self.df['image'].astype(str).str.contains(str(path_IAI)), 'IAI', 'ChestXRayIEEE8023')
        self.df['view'] = np.concatenate((df_metadata.view.values, IAI_metadata.ViewPosition.values))
        self.df['finding'] = np.concatenate((df_metadata.finding.values, IAI_metadata.Outcome.values))
        self.df['Batch'] = np.concatenate((np.repeat('-', df_metadata.shape[0]), IAI_metadata.Batch.values))
        
        # take only IAI dataset
        if only_IAI:
            self.df = self.df[self.df.source == 'IAI']
            self.labels = self.df.label.values.astype(float)
            
        self.df = self.df.reset_index(drop = True)

        if show_info:
            print("\n\n\n############    Final Dataset    ############")
            display(self.df)
            display(self.df.groupby(['label', 'meaning']).size().to_frame().add_prefix('X_').rename(columns={'X_0': 'count'}))
            display(self.df.groupby(['label', 'meaning', 'source']).size().to_frame().add_prefix('X_').rename(columns={'X_0': 'count'}))
#             print("\n{} (1: {}, 0: {}) images from ieee8023 and {} (all 1) from IAI-anonimizzati".format(df_metadata.shape[0],
#                                                                                                    str(sum(labels)),
#                                                                                                  str(len(labels) - sum(labels)),
#                                                                                                   len(images_IAI)))

    @staticmethod
    def _load_image(path, size):
        img = Image.open(path)
        img = cv2.resize(np.array(img), (size, size), interpolation=cv2.INTER_AREA)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
            img = np.dstack([img, img, img])
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # size, size, chan -> chan, size, size
        img = np.transpose(img, axes=[2, 0, 1])
        
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