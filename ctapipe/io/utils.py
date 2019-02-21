import gzip


def is_fits_in_header(file_path):
    '''quick check if file is a FITS file

    by looking into the first 1024 bytes and searching for the string "FITS"
    typically used in is_compatible
    '''
    # read the first 1kB
    with open(file_path, 'rb') as f:
        marker_bytes = f.read(1024)

    # if file is gzip, read the first 4 bytes with gzip again
    if marker_bytes[0] == 0x1f and marker_bytes[1] == 0x8b:
        with gzip.open(file_path, 'rb') as f:
            marker_bytes = f.read(1024)

    return b'FITS' in marker_bytes
