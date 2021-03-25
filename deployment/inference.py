import pydicom as dicom
import io
import os
import cv2
import json
import pandas as pd
import numpy as np
import copy
import urllib.request
from scipy.spatial import ConvexHull
from struct import pack, unpack
import warnings
import torch
import torch.nn as nn
from torchvision.models import resnet34


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
        out_clip = copy.deepcopy(dcm.pixel_array)
        tot_frames = out_clip.shape[0]
        for fr in range(tot_frames):
            frame = out_clip[fr]
            out_clip[fr] = _convert_YBR_FULL_to_RGB(frame)
    
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

def create_model_input(data_gray, mask, frame_to_keep = 50, frame_sampling = 'first_available', apply_frame_difference = True,
                       apply_sampling_first = True, apply_mask = True, resize = None):
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
    
    # select frames sampling for files with more than frame_to_keep frames
    def sample_frames(frame_array, total_frames, frame_sampling):

        if frame_sampling == 'first_available':
            data_out = frame_array[:total_frames]
        elif frame_sampling == 'equally_spaced':
            data_out = frame_array[np.linspace(0, frame_array.shape[0] - 1, total_frames).astype(int)]
            print(np.linspace(0, frame_array.shape[0] - 1, total_frames).astype(int))

        return data_out

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
                
    return data_out


'''
_rle_decode_frame_mod try to fix an error in DICOM files when using dcm.pixel_array:

  "The amount of decoded RLE segment data doesn't match the expected amount (276025 vs. 276024 bytes)"
 
 
https://github.com/pydicom/pydicom/blob/9561c5d40ec96076c974b5987767dc7b78c500a6/pydicom/pixel_data_handlers/rle_handler.py#L337

has been used to edit the function _rle_decode_frame. _parse_rle_header and _rle_decode_segment are needed to run the edited function

'''

def _rle_decode_frame_mod(data, rows, columns, nr_samples, nr_bits):
    """Decodes a single frame of RLE encoded data.
    Each frame may contain up to 15 segments of encoded data.
    Parameters
    ----------
    data : bytes
        The RLE frame data
    rows : int
        The number of output rows
    columns : int
        The number of output columns
    nr_samples : int
        Number of samples per pixel (e.g. 3 for RGB data).
    nr_bits : int
        Number of bits per sample - must be a multiple of 8
    Returns
    -------
    bytearray
        The frame's decoded data in big endian and planar configuration 1
        byte ordering (i.e. for RGB data this is all red pixels then all
        green then all blue, with the bytes for each pixel ordered from
        MSB to LSB when reading left to right).
    """
    if nr_bits % 8:
        raise NotImplementedError(
            "Unable to decode RLE encoded pixel data with a (0028,0100) "
            "'Bits Allocated' value of {}".format(nr_bits)
        )

    # Parse the RLE Header
    offsets = _parse_rle_header(data[:64])
    nr_segments = len(offsets)

    # Check that the actual number of segments is as expected
    bytes_per_sample = nr_bits // 8
    if nr_segments != nr_samples * bytes_per_sample:
        raise ValueError(
            "The number of RLE segments in the pixel data doesn't match the "
            "expected amount ({} vs. {} segments)"
            .format(nr_segments, nr_samples * bytes_per_sample)
        )

    # Ensure the last segment gets decoded
    offsets.append(len(data))

    # Preallocate with null bytes
    decoded = bytearray(rows * columns * nr_samples * bytes_per_sample)

    # Example:
    # RLE encoded data is ordered like this (for 16-bit, 3 sample):
    #  Segment: 1     | 2     | 3     | 4     | 5     | 6
    #           R MSB | R LSB | G MSB | G LSB | B MSB | B LSB
    #  A segment contains only the MSB or LSB parts of all the sample pixels

    # To minimise the amount of array manipulation later, and to make things
    # faster we interleave each segment in a manner consistent with a planar
    # configuration of 1 (and maintain big endian byte ordering):
    #    All red samples             | All green samples           | All blue
    #    Pxl 1   Pxl 2   ... Pxl N   | Pxl 1   Pxl 2   ... Pxl N   | ...
    #    MSB LSB MSB LSB ... MSB LSB | MSB LSB MSB LSB ... MSB LSB | ...

    # `stride` is the total number of bytes of each sample plane
    stride = bytes_per_sample * rows * columns
    for sample_number in range(nr_samples):
        for byte_offset in range(bytes_per_sample):
            # Decode the segment
            # ii is 0, 1, 2, 3, ..., (nr_segments - 1)
            ii = sample_number * bytes_per_sample + byte_offset
            segment = _rle_decode_segment(data[offsets[ii]:offsets[ii + 1]])
            # Check that the number of decoded pixels is correct
            if len(segment) != rows * columns:
#                 raise ValueError(
#                     "The amount of decoded RLE segment data doesn't match the "
#                     "expected amount ({} vs. {} bytes)"
#                     .format(len(segment), rows * columns)
#                 )
                segment = segment[:rows * columns]
            

            # For 100 pixel/plane, 32-bit, 3 sample data `start` will be
            #   0, 1, 2, 3, 400, 401, 402, 403, 800, 801, 802, 803
            start = byte_offset + sample_number * stride
            decoded[start:start + stride:bytes_per_sample] = segment

    return decoded

def _parse_rle_header(header):
    """Return a list of byte offsets for the segments in RLE data.
    **RLE Header Format**
    The RLE Header contains the number of segments for the image and the
    starting offset of each segment. Each of these numbers is represented as
    an unsigned long stored in little-endian. The RLE Header is 16 long words
    in length (i.e. 64 bytes) which allows it to describe a compressed image
    with up to 15 segments. All unused segment offsets shall be set to zero.
    As an example, the table below describes an RLE Header with 3 segments as
    would typically be used with 8-bit RGB or YCbCr data (with 1 segment per
    channel).
    +--------------+---------------------------------+------------+
    | Byte  offset | Description                     | Value      |
    +==============+=================================+============+
    | 0            | Number of segments              | 3          |
    +--------------+---------------------------------+------------+
    | 4            | Offset of segment 1, N bytes    | 64         |
    +--------------+---------------------------------+------------+
    | 8            | Offset of segment 2, M bytes    | 64 + N     |
    +--------------+---------------------------------+------------+
    | 12           | Offset of segment 3             | 64 + N + M |
    +--------------+---------------------------------+------------+
    | 16           | Offset of segment 4 (not used)  | 0          |
    +--------------+---------------------------------+------------+
    | ...          | ...                             | 0          |
    +--------------+---------------------------------+------------+
    | 60           | Offset of segment 15 (not used) | 0          |
    +--------------+---------------------------------+------------+
    Parameters
    ----------
    header : bytes
        The RLE header data (i.e. the first 64 bytes of an RLE frame).
    Returns
    -------
    list of int
        The byte offsets for each segment in the RLE data.
    Raises
    ------
    ValueError
        If there are more than 15 segments or if the header is not 64 bytes
        long.
    References
    ----------
    DICOM Standard, Part 5, :dcm:`Annex G<part05/chapter_G.html>`
    """
    if len(header) != 64:
        raise ValueError('The RLE header can only be 64 bytes long')

    nr_segments = unpack('<L', header[:4])[0]
    if nr_segments > 15:
        raise ValueError(
            "The RLE header specifies an invalid number of segments ({})"
            .format(nr_segments)
        )

    offsets = unpack('<{}L'.format(nr_segments),
                     header[4:4 * (nr_segments + 1)])

    return list(offsets)

def _rle_decode_segment(data):
    """Return a single segment of decoded RLE data as bytearray.
    Parameters
    ----------
    data : bytes
        The segment data to be decoded.
    Returns
    -------
    bytearray
        The decoded segment.
    """

    data = bytearray(data)
    result = bytearray()
    pos = 0
    result_extend = result.extend

    try:
        while True:
            # header_byte is N + 1
            header_byte = data[pos] + 1
            pos += 1
            if header_byte > 129:
                # Extend by copying the next byte (-N + 1) times
                # however since using uint8 instead of int8 this will be
                # (256 - N + 1) times
                result_extend(data[pos:pos + 1] * (258 - header_byte))
                pos += 1
            elif header_byte < 129:
                # Extend by literally copying the next (N + 1) bytes
                result_extend(data[pos:pos + header_byte])
                pos += header_byte

    except IndexError:
        pass

    return result

def input_fn(request_body, content_type='application/dicom'):
    
    MIN_FRAMES = 50
    FRAME_SAMPLING = 'first_available'
    APPLY_FRAME_DIFFERENCE = False
    APPLY_SAMPLING_FIRST = True
    APPLY_MASK = True
    RESIZE = 256

    if content_type == 'application/dicom':
        dcm = dicom.dcmread(io.BytesIO(request_body))
    elif content_type == 'application/json':
        json_request = json.loads(request_body)
        dicom_url = json_request['source']
        req = urllib.request.urlopen(dicom_url)

        if(req.status != 200):
            raise('Unable to download DICOM from {}'.format(url))
        raw_req = req.read()
        dcm = dicom.dcmread(io.BytesIO(raw_req))
    else:
        raise Exception(f'Requested unsupported ContentType in content_type: {content_type}')

    # check if file is correctly loaded
    try:
        array_image = dcm.pixel_array
    except:
        # try edited pydicom function to fix the issue
        try:
            # edit function and save original for later
            original_function = copy.deepcopy(dicom.pixel_data_handlers.rle_handler._rle_decode_frame)
            dicom.pixel_data_handlers.rle_handler._rle_decode_frame = _rle_decode_frame_mod

            array_image = dcm.pixel_array

            # set back original function
            dicom.pixel_data_handlers.rle_handler._rle_decode_frame = original_function
        except:
            raise ValueError('file corrupted - loading failed')

    # check PhotometricInterpretation
    photometric = dcm.PhotometricInterpretation
    if photometric not in ['RGB', 'YBR_FULL_422', 'YBR_FULL', 'PALETTE COLOR']:
        raise ValueError('file corrupted - loading failed')

    # check minimum number of frames
    if (len(array_image.shape) != 4): print('\n--- loaded file doesn\'t have 4 dimensions')
    if array_image.shape[0] < MIN_FRAMES: raise ValueError('file has less than ' + str(MIN_FRAMES) + ' frames')
    if (array_image.shape[0] != int(dcm.NumberOfFrames)): raise ValueError('extracted frames doesn\'t match expected ones')
    del array_image

    # extract file and convert according to photometric
    if photometric == 'RGB':
        out_clip = dcm.pixel_array
    elif photometric == 'YBR_FULL_422' or photometric == 'YBR_FULL':
        out_clip = YBR_to_RGB(dcm, show_info=False)
    elif photometric == 'PALETTE COLOR':
        out_clip = get_pixel_data_with_lut_applied(dcm)

    # convert to grayscale and save pickle
    out_clip_gray = np.zeros(out_clip.shape[:3])
    for i in range(out_clip.shape[0]):
        out_clip_gray[i] = cv2.cvtColor(out_clip[i], cv2.COLOR_RGB2GRAY)
    out_clip_gray = out_clip_gray.astype(np.uint8)
    del out_clip

    # extract mask to isolate central "triangle"
    mask, _, _ = extract_mask(frame_list = out_clip_gray, break_thresh = 20, pixel_to_reduce = 0.02,
                        left_bottom_corner_cut = 0.25, max_diff_to_try = 10)
    if mask.sum().sum() == 0: warnings.warn('no mask detected')

    # remove duplicated frames (if any)
    masked_img = copy.deepcopy(out_clip_gray)
    for f in range(out_clip_gray.shape[0]):
        masked_img[f][mask] = 0

    duplicated_index = []
    for f in range(out_clip_gray.shape[0]-1):
        frame_diff = masked_img[f+1] - masked_img[f]
        if frame_diff.max() == 0:
            duplicated_index.append(f)

    if len(duplicated_index) > 0:
        if 0 in duplicated_index:
            warnings.warn('first frame removed')
            out_clip_gray = out_clip_gray[1:]
    del masked_img
    if out_clip_gray.shape[0] < MIN_FRAMES: raise ValueError('file has less than ' + str(MIN_FRAMES) + ' frames')

    # create final input for models
    model_input = create_model_input(data_gray = out_clip_gray, mask = mask, frame_to_keep = MIN_FRAMES,
                                    frame_sampling = FRAME_SAMPLING, apply_frame_difference = APPLY_FRAME_DIFFERENCE,
                                    apply_sampling_first = APPLY_SAMPLING_FIRST, apply_mask = APPLY_MASK, resize = RESIZE)
    del out_clip_gray

    return torch.from_numpy(model_input).float() / 255.


def model_fn(model_dir):
    
    NUM_CLASSES=1
    NUM_CHANNELS=50
    FC_LAYERS=1
    DROPOUT=0.5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # define model
    model = resnet34(pretrained=False)
    first_layer = model.conv1
    first_layer = nn.Conv2d(in_channels=NUM_CHANNELS,
                            out_channels=first_layer.out_channels,
                            kernel_size=first_layer.kernel_size,
                            stride=first_layer.stride,
                            padding=first_layer.padding,
                            padding_mode=first_layer.padding_mode,
                            dilation=first_layer.dilation,
                            groups=first_layer.groups,
                            bias=first_layer.bias)
    model.conv1 = first_layer
    bottleneck_features = model.fc.in_features
    last_layer = nn.Sigmoid() if NUM_CLASSES == 1 else nn.Softmax(dim=1)
    fc_module = []
    fc_module.append(nn.BatchNorm1d(bottleneck_features))
    fc_module.append(nn.Dropout(DROPOUT))
    ll=0
    for ll in range(1, FC_LAYERS):
        fc_module.append(nn.Linear(int(bottleneck_features / 2**(ll-1)), int(bottleneck_features / 2**ll)))
        fc_module.append(nn.BatchNorm1d(int(bottleneck_features / 2**ll)))
        fc_module.append(nn.Dropout(DROPOUT))
    fc_module.append(nn.Linear(int(bottleneck_features / 2**ll), NUM_CLASSES))
    fc_module.append(last_layer)
    model.fc = nn.Sequential(*fc_module)
    
    # load state_dict
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device)['state_dict'])

    return model

def predict_fn(input_data, model):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_data = input_data.to(device)
    model.eval()
    with torch.jit.optimized_execution(True):#{"target_device": "eia:0"}):
        output = model(input_data.unsqueeze(0))

    return output.detach().cpu()

def output_fn(prediction_output, accept='application/json'):
    
    result = {'Akinetic': 'Yes' if prediction_output > 0.5 else 'No',
             'Probability': round((float(prediction_output) if prediction_output > 0.5 else 1 - float(prediction_output)), 3)}

    if accept == 'application/json':
        return json.dumps(result), accept
    raise Exception(f'Requested unsupported ContentType in Accept: {accept}')