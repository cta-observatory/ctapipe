from time import sleep
import threading
import subprocess
import os

def build_command(executable,  input_file, output_dir, out_extension,source_dir=None, options=None):
    '''
    Create a string containing excutable shell command with following format:
    executable -i source_dir/input_file -o output_dir/outfile_file
    output_file if build from input_file without its extension plus out_extension

    Parameters
    ----------
    executable: str
            executable file Name
    input_file: str
        input file name
    output_dir: str
        output directory name
    out_extension: str
        output extension to be added to input_file
        (once orrigin extention have been removed)
    source_dir: str
        source directory name containing input_file
    options : str
        option to be add to the command

    Returns:
    --------
    a string containing the command its parameters and options
    '''
    # remove full path before file name
    output_file = input_file.split('/')[-1]
    # replace extension if need
    if out_extension != None:
        output_file = output_file.rsplit('.', 2)[0]  # remove old_extension
        output_file += "." + out_extension     # add new extension
    # build command
    if source_dir != None:
        input_file = source_dir + '/' + input_file
    cmd = [executable, '-i', input_file]
    if output_dir != None:
        cmd.append("-o")
        output_file = output_dir + "/" + output_file
        cmd.append(output_file)
    if options != None:
        cmd.append(options)
    return cmd, output_file
