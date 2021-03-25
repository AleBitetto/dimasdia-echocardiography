import pydicom as dicom
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from shutil import copyfile
import cv2
import pandas as pd
import csv
import numpy as np
import IPython
import math
from zipfile import ZipFile
from IPython.display import display,clear_output
import copy
from timeit import default_timer as timer
import datetime
from PIL import Image
import pickle
from scipy.spatial import ConvexHull


def update_metadata(metadata, input_folder = './sacco/', metadata_path = './sacco/s3sync.csv'):
    
    named_col = ['SamplesPerPixel', 'PhotometricInterpretation', 'PlanarConfiguration', 'BitsAllocated', 'NumberOfFrames',
                 'Rows', 'Columns', 'CineRate', 'EffectiveDuration', 'FrameTime', 'FrameDelay', 'HeartRate',
                 'ActualFrameDuration', 'RWaveTimeVector', 'PatientName','PatientSex', 'PatientBirthDate']
    meta_to_add=pd.DataFrame(columns = ['Import_Status', 'Frames', 'x_size', 'y_size', 'Transform'] + named_col, dtype=int).fillna(0)
    dd = {}
    for cc in meta_to_add.columns:
        dd[cc] = 0
    row_to_add_empty = pd.DataFrame(dd, dtype=int, index=[0])

    start = timer()
    for i, f in metadata.iterrows():   #.iloc[[0,1,52, 53, 55, 116]]

        print(str(i+1)+'/'+str(metadata.shape[0]), f.filename, end = ': ')

        dcm = dicom.dcmread(os.path.join(input_folder, f['index']))
        row_to_add = copy.deepcopy(row_to_add_empty)

        if hasattr(dcm, 'PixelIntensityRelationship'):
            if dcm.PixelIntensityRelationship == 'LIN' and dcm.PixelIntensityRelationshipSign == 1:
                print('has Linear transformation', end=' ')
            else:
                print('has transformation', dcm.PixelIntensityRelationship, dcm.PixelIntensityRelationshipSign, end=' ')
            print('- size: ', dcm.pixel_array.shape, str(dcm.pixel_array.min())+'-'+str(dcm.pixel_array.max()), end=' ')
            cont = True
            row_to_add['Transform']=dcm.PixelIntensityRelationship
        else:
            print('has no transformation', end=' ')
            row_to_add['Transform']='None'
            try:
                output_image = dcm.pixel_array
                print('- size: ', dcm.pixel_array.shape, str(dcm.pixel_array.min())+'-'+str(dcm.pixel_array.max()), end=' ')
                cont = True
            except:
                print('and error in conversion  <<------------------', end='\n')
                cont = False
                output_image = None

        if cont:
            print(dcm.PhotometricInterpretation, end='\n')
            row_to_add['Frames']=dcm.pixel_array.shape[0]
            row_to_add['x_size']=dcm.pixel_array.shape[2]
            row_to_add['y_size']=dcm.pixel_array.shape[1]

        row_to_add['Import_Status']='OK' if cont else 'Failed'

        for c in named_col:
            try:
                row_to_add[c]=dcm[c].value
            except:
                pass

        meta_to_add=meta_to_add.append(row_to_add)
    print('\nTotal elapsed time:', str(datetime.timedelta(seconds=round(timer()-start))))

    metadata = pd.concat([metadata.reset_index(drop=True), meta_to_add.reset_index(drop=True)],axis=1)
#     metadata['diagnosis'] = metadata[['diagnosis_akinetic', 'diagnosis_covid', 'diagnosis_healthy']].idxmax(axis=1)
#     metadata['diagnosis'] = metadata['diagnosis'].str.replace('diagnosis_','')
    metadata.insert(0, 'folder', metadata.apply(lambda x: x['index'].replace('eco-scan-cardio/', '') \
                                                .replace(x['filename'], '').split('/')[0], axis=1))

    return metadata


def convert_DICOM(metadata_df, filename_list, input_folder = './sacco/',
                  output_PNG_folder = './sacco/PNG/Converted', output_PKL_folder = './sacco/PKL'):

    col_to_keep = ['patient', 'covid', 'akinetic', 'NumberOfFrames', 'Rows', 'Columns', 'CineRate', 'EffectiveDuration', 'FrameTime',
                   'FrameDelay', 'HeartRate', 'ActualFrameDuration', 'RWaveTimeVector',
                   'PatientName','PatientSex', 'PatientBirthDate']
    metadata_by_frame=pd.DataFrame(columns = ['folder', 'index', 'index_pickle', 'filename_orig', 'filename', 'index_frame',
                                              'filename_frame', 'Frame_ID', 'Import_Status'] + col_to_keep, dtype=int).fillna(0)
    _=os.makedirs(output_PNG_folder, exist_ok=True)
    _=os.makedirs(output_PKL_folder, exist_ok=True)
    start = timer()
    for i, row in metadata_df.reset_index(drop=True).iterrows():  # .iloc[[262,263,268,270, 306, 310]]
        status = row.Import_Status
        row['filename_orig'] = row['filename']

        print(i+1, '/', metadata_df.shape[0], end = '\r')

    #     if status == 'OK':
        path = row['index']
        photometric = row.PhotometricInterpretation
        dcm = dicom.dcmread(os.path.join(input_folder, path))
        cont=True
        if photometric == 'RGB':
            out_clip = dcm.pixel_array
        elif photometric == 'YBR_FULL_422' or photometric == 'YBR_FULL':
            out_clip = YBR_to_RGB(dcm, show_info=False)
        elif photometric == 'PALETTE COLOR':
            out_clip = get_pixel_data_with_lut_applied(dcm)
        else:
            print(i, row['index'], ': PhotometricInterpretation not found')
            cont=False
    #     else:
    #         cont=False

        if cont:

            # check for duplicates
            if row.filename in filename_list:
                row['filename'] = row.filename.split('.')[0] + '_BIS.dcm'
                print(i, row['index'], ': duplicated filename, changed in', row.filename)
            filename_list.append(row.filename)

            if row.NumberOfFrames != out_clip.shape[0]:
                print(i, row['index'], ': mismatch in number of frames', row.NumberOfFrames, '(expected) vs',
                      out_clip.shape[0], '(extracted)')
                
            # convert to grayscale and save pickle
            out_clip_gray = np.zeros(out_clip.shape[:3])
            for i in range(out_clip.shape[0]):
                out_clip_gray[i] = cv2.cvtColor(out_clip[i], cv2.COLOR_RGB2GRAY)
            out_clip_gray = out_clip_gray.astype(np.uint8)

            pkl_path = os.path.join(output_PKL_folder, row['filename'].replace('.dcm', '.pickle'))
            with open(pkl_path, 'wb') as handle:
                pickle.dump(out_clip_gray, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            # update metadata_by_frame
            for fr in range(row.NumberOfFrames):
                frame = out_clip[fr]
    #             frame_name = row.filename.split('.')[0] + '_' + "{:03d}".format(fr) + '.png'
                frame_name = row.filename.replace('.dcm', '_' + "{:03d}".format(fr) + '.png')
                frame_path = os.path.join(output_PNG_folder, frame_name)
                cv2.imwrite(frame_path, frame)

                row_to_add = row[['folder', 'index', 'filename', 'filename_orig', 'Import_Status'] + col_to_keep]
                row_to_add['index_pickle'] = pkl_path
                row_to_add['index_frame'] = frame_path
                row_to_add['filename_frame'] = frame_name
                row_to_add['Frame_ID'] = fr
                metadata_by_frame=metadata_by_frame.append(row_to_add)

    print('\nTotal elapsed time:', str(datetime.timedelta(seconds=round(timer()-start))))

    return metadata_by_frame, filename_list


def YBR_to_RGB(dcm, show_info=True):
    '''
    Convert from YBR to RGB
    https://dicom.innolitics.com/ciods/enhanced-mr-image/enhanced-mr-image/00280004
    
    Transformation matrix:
    Y = + .2990R + .5870G + .1140B

    CB= - .1687R - .3313G + .5000B + 128

    CR= + .5000R - .4187G - .0813B + 128
    '''

    bits = dcm.BitsAllocated
    channel = dcm.SamplesPerPixel
    photo = dcm.PhotometricInterpretation
    
    if photo not in ['YBR_FULL_422', 'YBR_FULL']:
        print('\n warning, PhotometricInterpretation is', photo)
    if (bits != 8) or (channel != 3):
        
        print('\n wrong conversion parameters. Import Failed  <------------')
        out_clip=None
    else:
        
        Inv_Mat = np.array([[.2990, .5870, .1140], [-.1687, -.3313, .5000], [.5000, -.4187, -.0813]])
        Const = np.array([0,128,128])
        
        out_clip = copy.deepcopy(dcm.pixel_array)
        tot_frames = out_clip.shape[0]
        start = timer()
        for fr in range(tot_frames):

            if show_info:
                print('Frame:', str(fr+1)+'/'+str(tot_frames)+'    ', end='\r' if fr <= (tot_frames-2) else '')
            
            frame = out_clip[fr]
#             x, y, chann = frame.shape
#             YBR = frame.reshape(x*y, chann).transpose()

#             b = YBR - Const.reshape(chann,1)
#             RGB = np.linalg.solve(Inv_Mat, b).transpose().reshape(x,y,chann)
#             out_clip[fr] = RGB

            out_clip[fr] = _convert_YBR_FULL_to_RGB(frame)
    
        if show_info:
            print(str(datetime.timedelta(seconds=round(timer()-start))))
    
    return out_clip


def _convert_YBR_FULL_to_RGB(arr):
    """
    taken from https://github.com/pydicom/pydicom/blob/9561c5d40ec96076c974b5987767dc7b78c500a6/pydicom/pixel_data_handlers/util.py#L430
    
    Return an ndarray converted from YBR_FULL to RGB color space.
    Parameters
    ----------
    arr : numpy.ndarray
        An ndarray of an 8-bit per channel images in YBR_FULL color space.
    Returns
    -------
    numpy.ndarray
        The array in RGB color space.
    References
    ----------
    * DICOM Standard, Part 3,
      :dcm:`Annex C.7.6.3.1.2<part03/sect_C.7.6.3.html#sect_C.7.6.3.1.2>`
    * ISO/IEC 10918-5:2012, Section 7
    """
    orig_dtype = arr.dtype

    ybr_to_rgb = np.asarray(
        [[1.000, 1.000, 1.000],
         [0.000, -0.114 * 1.772 / 0.587, 1.772],
         [1.402, -0.299 * 1.402 / 0.587, 0.000]],
        dtype=np.float
    )

    arr = arr.astype(np.float)
    arr -= [0, 128, 128]
    arr = np.dot(arr, ybr_to_rgb)

    # Round(x) -> floor of (arr + 0.5)
    arr = np.floor(arr + 0.5)
    # Max(0, arr) -> 0 if 0 >= arr, arr otherwise
    # Min(arr, 255) -> arr if arr <= 255, 255 otherwise
    arr = np.clip(arr, 0, 255)

    return arr.astype(orig_dtype)


def get_pixel_data_with_lut_applied(dcm):
    '''
    used for PhotometricInterpretation = 'PALETTE COLOR'
    taken from https://github.com/pydicom/pydicom/issues/205
    '''

    #For Supplemental, numbers below LUT are greyscale, else clipped
    #Don't have a file to test, or know where this flag is stored in pydicom
    SUPPLEMENTAL_LUT = False


    if dcm.PhotometricInterpretation != 'PALETTE COLOR':
        raise Exception

    if (dcm.RedPaletteColorLookupTableDescriptor[0] != dcm.BluePaletteColorLookupTableDescriptor[0] != dcm.GreenPaletteColorLookupTableDescriptor[0]):
        raise Exception

    if (dcm.RedPaletteColorLookupTableDescriptor[1] != dcm.BluePaletteColorLookupTableDescriptor[1] != dcm.GreenPaletteColorLookupTableDescriptor[1]):
        raise Exception

    if (dcm.RedPaletteColorLookupTableDescriptor[2] != dcm.BluePaletteColorLookupTableDescriptor[2] != dcm.GreenPaletteColorLookupTableDescriptor[2]):
        raise Exception

    if (len(dcm.RedPaletteColorLookupTableData) != len(dcm.BluePaletteColorLookupTableData) != len(dcm.GreenPaletteColorLookupTableData)):
        raise Exception


    lut_num_values = dcm.RedPaletteColorLookupTableDescriptor[0]
    lut_first_value = dcm.RedPaletteColorLookupTableDescriptor[1]
    lut_bits_per_pixel = dcm.RedPaletteColorLookupTableDescriptor[2] # warning that they lie though
    lut_data_len = len(dcm.RedPaletteColorLookupTableData)


    if lut_num_values == 0:
        lut_num_values = 2**16

    if not (lut_bits_per_pixel == 8 or lut_bits_per_pixel == 16):
        raise Exception

    if lut_data_len != lut_num_values * lut_bits_per_pixel // 8:
        #perhaps claims 16 bits but only store 8 (apparently even the spec says implementaions lie)
        if lut_bits_per_pixel == 16:
            if lut_data_len == lut_num_values * 8 / 8:
                lut_bits_per_pixel = 8
            else:
                raise Exception
        else:
            raise Exception


    lut_dtype = None

    if lut_bits_per_pixel == 8:
        lut_dtype = np.uint8

    if lut_bits_per_pixel == 16:
        lut_dtype = np.uint16

    red_palette_data = np.frombuffer(dcm.RedPaletteColorLookupTableData, dtype=lut_dtype)
    green_palette_data = np.frombuffer(dcm.GreenPaletteColorLookupTableData, dtype=lut_dtype)
    blue_palette_data = np.frombuffer(dcm.BluePaletteColorLookupTableData, dtype=lut_dtype)

    if lut_first_value != 0:
        if SUPPLEMENTAL_LUT:
            red_palette_start = np.arange(lut_first_value, dtype=lut_dtype)
            green_palette_start = np.arange(lut_first_value, dtype=lut_dtype)
            blue_palette_start = np.arange(lut_first_value, dtype=lut_dtype)
        else:
            red_palette_start = np.ones(lut_first_value, dtype=lut_dtype) * red_palette_data[0]
            green_palette_start = np.ones(lut_first_value, dtype=lut_dtype) * green_palette_data[0]
            blue_palette_start = np.ones(lut_first_value, dtype=lut_dtype) * blue_palette_data[0]

        red_palette = np.concatenate((red_palette_start, red_palette_data))
        green_palette = np.concatenate((green_palette_start, red_palette_data))
        blue_palette = np.concatenate((blue_palette_start, red_palette_data))
    else:
        red_palette = red_palette_data
        green_palette = green_palette_data
        blue_palette = blue_palette_data


    red = red_palette[dcm.pixel_array]
    green = green_palette[dcm.pixel_array]
    blue = blue_palette[dcm.pixel_array]

    out = np.stack((red, green, blue), axis=-1)

    if lut_bits_per_pixel == 16:
        out = (out // 256).astype(np.uint8)

    return out


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


def plot_frame_sample(metadata, metadata_by_frame, save_folder = './sacco/sample_and_check/frame_sample/',
                      fig_x_size = 15, max_sample_per_split = 80):

    # keep max_sample_per_split below 100 otherwise out of memory (is the number of rows in subplot)
    # single_frame_x_size = 400

    print('\n Red labels mean Failed Import\n\n')
    _=os.makedirs(save_folder, exist_ok=True)
    sample_order = pd.DataFrame(columns=['folder', 'index', 'filename'], dtype=str)
    for i, row in metadata.iterrows():
        sample_order = sample_order.append(
            pd.DataFrame({'folder': row['index'].replace('eco-scan-cardio/', '').replace(row.filename, '').split('/')[0],
                          'index': row['index'],
                          'filename': row.filename}, index = [0])
        )
    sample_order = sample_order.sort_values(by=['folder', 'filename'])
    fig_list = []
    for folder in sample_order.folder.unique():

        folder_list = sample_order[sample_order.folder == folder]
        split_index = np.array_split(np.arange(len(folder_list)), math.ceil(len(folder_list) / max_sample_per_split))
        split_path = []
#         fig_y_size_cum = 0
        for sp, split_ind in enumerate(split_index):

            fig_y_size = fig_x_size / 2 * len(split_ind)
#             fig_y_size_cum += fig_y_size
            fig, ax = plt.subplots(len(split_ind), 3, figsize=(fig_x_size, fig_y_size), sharey=False, sharex=False)
            ax = ax.flatten()
            ax_c = 0
            for filename in folder_list.iloc[split_ind].filename.unique():
                frame_df = metadata_by_frame[metadata_by_frame.filename == filename].sort_values(by=['Frame_ID'])
                metadata_loc = metadata[metadata['index'] == frame_df['index'].values[0]].index[0]
                import_status = metadata[metadata['index'] == frame_df['index'].values[0]].Import_Status.values[0]
                photometric = metadata[metadata['index'] == frame_df['index'].values[0]].PhotometricInterpretation.values[0]
                tot_frames = frame_df.NumberOfFrames.values[0]
                
                for frame in [0, round(tot_frames / 2), tot_frames-1]:
                    frame_path = frame_df.index_frame.values[frame]
                    img = cv2.imread(frame_path)
            #         ratio = img.shape[0] / img.shape[1]  # y / x
            #         img = cv2.resize(img,
            #                        dsize=(single_frame_x_size, round(single_frame_x_size * ratio)), # size is (x, y)
            #                        interpolation=cv2.INTER_NEAREST)
                    ax[ax_c].imshow(img)#, cmap='gray')
                    ax[ax_c].set_xticks([])
                    ax[ax_c].set_yticks([])
                    if frame == 0:
                        ax[ax_c].set_title(str(frame)+' - '+photometric)
                    elif frame == tot_frames-1:
                        ax[ax_c].set_title(str(frame)+' - last')
                    else:
                        ax[ax_c].set_title(frame)
                    if ax_c % 3 == 0:
                        ax[ax_c].set_ylabel('metadata.loc: '+str(metadata_loc)+'\n'+filename,
                                            color='black' if import_status == 'OK' else 'red')
                    ax_c += 1
        #     fig.suptitle(folder, size = 20)
            if sp == 0:
                fig.text(0.5, 0.405, folder, horizontalalignment='center', size = 20)
            plt.subplots_adjust(top=0.4, bottom=0.08, hspace=0.1, wspace=0.1)
        #     plt.tight_layout(rect=[0, 0.03, 1, 0.5])
            plt.close()
            fig_split_path = os.path.join(save_folder, 'sample_'+folder+'_'+str(sp)+'.png')
            split_path.append(fig_split_path)
            fig.savefig(fig_split_path, bbox_inches="tight")

        # merge split figures
        fig_path = os.path.join(save_folder, 'sample_'+folder+'.png')
        if len(split_path) == 1:
            os.rename(split_path[0], fig_path)
            fig_list.append(fig_path)
            print('Saved in', fig_path)
        else:
            fig_list = fig_list + split_path
            [print('Saved in', x) for x in split_path]

    for im_path in fig_list:
        display(Image.open(im_path))
    #         get_concat_v([Image.open(x) for x in split_path]).save(fig_path, bbox_inches="tight")


    #         fig, ax = plt.subplots(len(split_path), 1, figsize=(fig_x_size, fig_y_size_cum), sharey=False, sharex=False)
    #         ax = ax.flatten()
    #         for i, im_path in enumerate(split_path):
    #             ax[i].imshow(cv2.imread(im_path))#, cmap='gray')
    #             ax[i].set_xticks([])
    #             ax[i].set_yticks([])
    #         plt.close()
    #         fig.savefig(fig_path, bbox_inches="tight")


    #     get_concat_v([Image.open(x) for x in split_path]).save(fig_path, bbox_inches="tight")
    #     fig.savefig(fig_path, bbox_inches="tight")
    #     [os.remove(x) for x in split_path]
    
    
def create_difference_GIF(filename, metadata, metadata_by_frame, save_folder = './sacco/GIF', fig_x_size = 10):

    _=os.makedirs(save_folder, exist_ok=True)
    frame_df = metadata_by_frame[metadata_by_frame.filename == filename].sort_values(by=['Frame_ID'])
    folder = frame_df['index'].values[0].replace('eco-scan-cardio/', '').replace(frame_df['filename'].values[0], '').split('/')[0]
    metadata_loc = metadata[metadata['index'] == frame_df['index'].values[0]].index[0]
    import_status = metadata[metadata['index'] == frame_df['index'].values[0]].Import_Status.values[0]
    photometric = metadata[metadata['index'] == frame_df['index'].values[0]].PhotometricInterpretation.values[0]

    # Create the GIF frames
    frame_list = frame_df.index_frame.values
    total_frames = len(frame_list)
    gif_frames = []
    for i in range(total_frames):

        current_img = cv2.imread(frame_list[i])
        ratio = current_img.shape[0] / current_img.shape[1] # y / x

        fig, ax = plt.subplots(1, 2, figsize=(fig_x_size * 2, ratio*fig_x_size), sharey=True, sharex=True)
        ax = ax.flatten()

        ax[0].imshow(current_img)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title('Frame '+str(i)+'/'+str(total_frames-1), size = 20)
        if i == 0:
            ax[1].imshow(np.ones(current_img.shape))
        else:
            ax[1].imshow(current_img - previous_img)
            ax[1].set_title('Frame difference '+str(i)+'-'+str(i-1), size = 20)
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        fig.text(0.5, 0.99, filename+' (metadata.loc='+str(metadata_loc)+')'+'\nFolder: '+folder+
                     '\nImport Status: '+import_status+'\nPhotometric: '+photometric,
                 horizontalalignment='center', size = 22)
        plt.subplots_adjust(top=0.9, bottom=0.08, hspace=0.9, wspace=0.1)
        plt.close()

        previous_img = current_img

        save_path = os.path.join(save_folder, 'frame_to_add.png')
        fig.savefig(save_path, bbox_inches="tight")
        gif_frames.append(Image.open(save_path))
        os.remove(save_path)

    # save the GIF
    gif_path = os.path.join(save_folder, filename.replace('.dcm', '')+'.gif')
    gif_frames[0].save(gif_path, format='GIF',
                   append_images=gif_frames[1:],
                   save_all=True,
                   duration=len(gif_frames), loop=0)

    return gif_path



def extract_mask(frame_list, break_thresh = 20, pixel_to_reduce = 0.02,
                 left_bottom_corner_cut = 0.25, max_diff_to_try = 10):
    '''
    extract mask to isolate shapes. It is supposed to be a central triangle with possible blob on the bottom corners
    
    Args:
        - frame_list: numpy array of (n_frames, y, x)
        - break_thresh: pixel threshold to isolate far left cluster of points, if any
                        (it should correspond to hearthrate signal start)
        - pixel_to_reduce: percentage of max(image.shape), pixel to be removed from all around the final mask
        - left_bottom_corner_cut: percentage of both img.shape to be removed (bottom left triangle)
                        (it should correspond to hearthrate signal start if not removed with break_thresh)
        - max_diff_to_try: try different consecutive frames difference until enough points remain
                        (sometimes first frames are equal apart from bottom left corner)
                        
    Output:
        - final_mask: a boolean mask where True means pixels to set to 0. If max_diff_to_try is reached,
                        a matrix of False is returned and a warning message is displayed
    '''

    
    # try different consecutive frame differences until enough points remain
    for diff_to_try in range(max_diff_to_try):

        img = frame_list[diff_to_try+1] - frame_list[diff_to_try]

        # remove bottom left corner
        if  left_bottom_corner_cut > 0:
            img=img.transpose()
            img[np.triu_indices(img.shape[1], k=int(img.shape[1]*(1-left_bottom_corner_cut)))] = 0
            img=img.transpose()

        # binarize image
        i, j = np.where(img > 0)
        points = np.vstack([j, i]).transpose()
        # parse all image from left to right and keep only top and bottom non zero pixel
        min_max_df = pd.DataFrame(points, columns = ['x', 'y']).groupby(['x']).agg({'y': [min, max]})
        # isolate cluster of point on far left, if any (it should correspond to hearthrate signal start)
        if min_max_df.shape[0] > 0:
            min_break_ind = np.where(np.diff(min_max_df.index) > break_thresh)[0]
            min_break_ind = min_max_df.index[min_break_ind+1][0] if len(min_break_ind) > 0 else min_max_df.index[0]
            if min_break_ind > int(img.shape[1]*0.3):
                min_break_ind = min_max_df.index[0]
            min_max_df = min_max_df.loc[min_break_ind:]
            # get list of point defining the shape
            mix_max_points = np.hstack([np.vstack([min_max_df.index.values, min_max_df[('y', 'min')].values]),
                               np.vstack([min_max_df.index.values, min_max_df[('y', 'max')].values])]).transpose()
            if min_max_df.index.max() > int(img.shape[1]*0.3) and mix_max_points.shape[0] > 3:
                # at least 4 points for hull convex and maximum x should exceed 30% of x size
                break
            

#     if diff_to_try > 0:
#         print('\nskipped first', diff_to_try, 'frames')

    if diff_to_try == (max_diff_to_try-1):
        print('\n######### no mask applied, reached maximum diff_to_try')
        final_mask = np.zeros(img.shape).astype(bool)
    else:

        # get the convex hull for all points
        hull = ConvexHull(mix_max_points)
        # evaluate mask for pixel inside hull
        image_point_list = [(x, y) for x in range(img.shape[1]) for y in range(img.shape[0])]
        A = hull.equations[:,0:-1]
        b = np.transpose(np.array([hull.equations[:,-1]]))
        mask = np.all((A @ np.transpose(image_point_list)) <= np.tile(-b,(1,len(image_point_list))),axis=0)  # array of True/False
        mask = mask.reshape((img.shape[1], img.shape[0])).transpose()

        if pixel_to_reduce == 0:
            final_mask = ~mask
        else:

            # reduce pixels from all around mask
            pixel_to_reduce = int(np.max(img.shape) * pixel_to_reduce)
            i, j = np.where(mask > 0)
            mask_points = np.vstack([j, i]).transpose()
            mask_min_max_df = pd.DataFrame(mask_points, columns = ['x', 'y']).groupby(['x']).agg({'y': [min, max]})
            mask_min_max_df['span'] = mask_min_max_df[('y', 'max')] - mask_min_max_df[('y', 'min')]
            mask_min_max_df = mask_min_max_df[mask_min_max_df.span > pixel_to_reduce*2]
            mask_min_max_df[('y', 'max')] = mask_min_max_df[('y', 'max')] - pixel_to_reduce
            mask_min_max_df[('y', 'min')] = mask_min_max_df[('y', 'min')] + pixel_to_reduce
            final_mask = np.ones(img.shape).astype(bool)

            for x, row in mask_min_max_df.iterrows():
                final_mask[np.arange(row[('y', 'min')], row[('y', 'max')]+1), x] = False
    
    masked_percentage = round(sum(final_mask).sum() / np.prod(img.shape) * 100)
    if masked_percentage >= 85:
        print('\n######### warning, mask is removing '+ str(masked_percentage)+ '% of image')

    return final_mask, diff_to_try, masked_percentage


def save_mask(metadata, image_folder = './sacco/PKL/masks', mask_folder = './sacco/PKL/masks'):

    _=os.makedirs(mask_folder, exist_ok=True)
    diff_to_try_log = []
    mask_percentage_log = []
    start = timer()
    for i, row in metadata.iterrows():

        filename = row.filename
        print(i+1, '/', metadata.shape[0], end = '\r')
        mask_path = os.path.join(mask_folder, filename.replace('.dcm', '_mask.pickle'))

        with open(row.index_pickle, 'rb') as handle:
            data_gray = pickle.load(handle)

        mask, diff_to_try, masked_percentage = extract_mask(frame_list = data_gray, break_thresh = 20, pixel_to_reduce = 0.02,
                                                            left_bottom_corner_cut = 0.25, max_diff_to_try = 10)
        diff_to_try_log.append([i, diff_to_try, row['index'], row['index_pickle'], mask_path])
        mask_percentage_log.append([i, masked_percentage, row['index'], row['index_pickle'], mask_path])

        with open(mask_path, 'wb') as handle:
            pickle.dump(mask, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('\nTotal elapsed time:', str(datetime.timedelta(seconds=round(timer()-start))))
    print('Max frame difference tried:', max([x[1] for x in diff_to_try_log]), ' - full list saved in diff_to_try_log')
    print('Total mask with masked pixel >= 85%:', len([x[0] for x in mask_percentage_log if x[1] >= 85]), ' - full list saved in mask_percentage_log')

    metadata['index_mask'] = metadata.index_pickle.str.replace('.pickle', '_mask.pickle').str.replace(image_folder, mask_folder)
    # metadata_by_frame['index_mask'] = metadata_by_frame.index_pickle.str.replace('.pickle', '_mask.pickle').str.replace(image_folder, mask_folder)
    metadata.to_csv('./sacco/Clip_MetaData.csv', index=False)
    # metadata_by_frame.to_csv('./sacco/Frame_MetaData.csv', index=False)
    print('\nmetadata file updated and saved')

    return diff_to_try_log, mask_percentage_log


def plot_mask_sample(metadata, save_folder = './sacco/sample_and_check/mask_sample/',
                      fig_x_size = 15, max_sample_per_split = 80):
    # keep max_sample_per_split below 100 otherwise out of memory (is the number of rows in subplot)
    # single_frame_x_size = 400

    _=os.makedirs(save_folder, exist_ok=True)
    sample_order = metadata[['folder', 'index_pickle', 'index_mask', 'filename']].sort_values(by=['folder', 'filename'])
    fig_list = []
    for folder in sample_order.folder.unique():

        folder_list = sample_order[sample_order.folder == folder]
        split_index = np.array_split(np.arange(len(folder_list)), math.ceil(len(folder_list) / max_sample_per_split))
        split_path = []
        for sp, split_ind in enumerate(split_index):

            fig_y_size = fig_x_size / 2 * len(split_ind)
            fig, ax = plt.subplots(len(split_ind), 4, figsize=(fig_x_size, fig_y_size), sharey=False, sharex=False)
            ax = ax.flatten()
            ax_c = 0
            for i, row in folder_list.iloc[split_ind].iterrows():

                filename = row.filename
                metadata_loc = metadata[metadata['index_pickle'] == row.index_pickle].index[0]

                with open(row.index_pickle, 'rb') as handle:
                    data_gray = pickle.load(handle)

                with open(row.index_mask, 'rb') as handle:
                    mask = pickle.load(handle)

                ax[ax_c].imshow(data_gray[0], cmap='gray')
                ax[ax_c].set_title('Original frame 1')
                ax[ax_c].set_xticks([])
                ax[ax_c].set_yticks([])
                ax[ax_c].set_ylabel('metadata.loc: '+str(metadata_loc)+'\n'+filename)
                ax[ax_c+1].imshow(mask, cmap='gray')
                ax[ax_c+1].set_title('Mask')
                ax[ax_c+1].set_xticks([])
                ax[ax_c+1].set_yticks([])
                masked_image = copy.deepcopy(data_gray[0])
                masked_image[mask] = 0
                ax[ax_c+2].imshow(masked_image, cmap='gray')
                ax[ax_c+2].set_title('Masked frame 1')
                ax[ax_c+2].set_xticks([])
                ax[ax_c+2].set_yticks([])
                masked_image = copy.deepcopy(data_gray[-1])
                masked_image[mask] = 0
                ax[ax_c+3].imshow(masked_image, cmap='gray')
                ax[ax_c+3].set_title('Masked frame last')
                ax[ax_c+3].set_xticks([])
                ax[ax_c+3].set_yticks([])
                ax_c += 4
            if sp == 0:
                fig.text(0.5, 0.405, folder, horizontalalignment='center', size = 20)
            plt.subplots_adjust(top=0.4, bottom=0.08, hspace=0.1, wspace=0.1)
            plt.close()
            fig_split_path = os.path.join(save_folder, 'sample_'+folder+'_'+str(sp)+'.png')
            split_path.append(fig_split_path)
            fig.savefig(fig_split_path, bbox_inches="tight")

        # merge split figures
        fig_path = os.path.join(save_folder, 'sample_'+folder+'.png')
        if len(split_path) == 1:
            os.rename(split_path[0], fig_path)
            fig_list.append(fig_path)
            print('Saved in', fig_path)
        else:
            fig_list = fig_list + split_path
            [print('Saved in', x) for x in split_path]

    for im_path in fig_list:
        display(Image.open(im_path))
        

def filter_duplicated_frames(metadata):
    
    new_metadata = pd.DataFrame(columns = list(metadata.columns) + ['NumberOfFrames_original'], dtype=int).fillna(0)
    duplicated_df = pd.DataFrame(columns = ['iloc', 'filename', 'duplicated_frames', 'total_frames', 'remove_first'], dtype=int).fillna(0)
    start = timer()
    for i, row in metadata.iterrows():

        filename = row.filename
        print(i+1, '/', metadata.shape[0], end = '\r')
        row['NumberOfFrames_original'] = row.NumberOfFrames

        with open(row.index_pickle, 'rb') as handle:
            data_gray = pickle.load(handle)

        with open(row.index_mask, 'rb') as handle:
            mask = pickle.load(handle)

        masked_img = copy.deepcopy(data_gray)
        for f in range(data_gray.shape[0]):
            masked_img[f][mask] = 0

        duplicated_index = []
        for f in range(data_gray.shape[0]-1):
            frame_diff = masked_img[f+1] - masked_img[f]
            if frame_diff.max() == 0:
                duplicated_index.append(f)

        if len(duplicated_index) > 0:
            row_to_add = pd.DataFrame([[i, filename, duplicated_index, data_gray.shape[0], 'No']], columns = duplicated_df.columns)
            if 0 in duplicated_index:
                row_to_add.remove_first = 'Yes'
                data_gray = data_gray[1:]
                row.NumberOfFrames = data_gray.shape[0]

                with open(row.index_pickle, 'wb') as handle:
                    pickle.dump(data_gray, handle, protocol=pickle.HIGHEST_PROTOCOL)

            duplicated_df = duplicated_df.append(row_to_add)

        new_metadata = new_metadata.append(row)

    duplicated_df.to_csv('./sacco/duplicated_frames.csv', index=False)        
    print('\nTotal elapsed time:', str(datetime.timedelta(seconds=round(timer()-start))))
    print('\nTotal files with removed frame (first):', sum(duplicated_df.remove_first == 'Yes'))
    print('Full list saved in \'./sacco/duplicated_frames.csv\'')

    if sum(new_metadata.NumberOfFrames != metadata.NumberOfFrames) != sum(duplicated_df.remove_first == 'Yes'):
        print('\n\n########### warning: edited NumberOfFrames doesn\'t match with edited files')
    else:
        new_metadata.to_csv('./sacco/Clip_MetaData.csv', index=False)
        print('\nmetadata file updated and saved')
        
        
def create_model_input(metadata, output_folder = './sacco/PKL/model_input/', metadata_out = './sacco/Model_MetaData.csv',
                       frame_to_keep = 50, frame_sampling = 'first_available',
                       apply_frame_difference = True, apply_sampling_first = True,
                      apply_mask = True, resize = None):

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
    display(working_df.groupby(['diagnosis']).size().to_frame().add_prefix('X_').rename(columns={'X_0': 'count'}).\
            join(metadata.groupby(['diagnosis']).size().to_frame().add_prefix('X_').rename(columns={'X_0': 'out of'})))

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
    first_columns = ['keep', 'folder', 'index_pickle_final', 'NumberOfFrames_final', 'diagnosis', 'index_pickle', 'index_mask']
    metadata_model = metadata_model[first_columns + list(set(metadata_model.columns) - set(first_columns))]
    
    metadata_model.to_csv(metadata_out, index=False)
    print('\nModel metadata saved in', metadata_out)