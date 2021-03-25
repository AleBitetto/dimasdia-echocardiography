from struct import pack, unpack


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