#!/usr/bin/env python3
import sys
from glob import glob
import shlex
import subprocess as sp

return_codes = []
for path in glob('examples/notebooks/**/*.ipynb', recursive=True):
    py_path = path.replace('.ipynb', '.py')
    print('testing:', path)
    nbconvert_command = "jupyter-nbconvert --to=script '{}'".format(path)
    print('    ', nbconvert_command, ' >/dev/null 2>&1')
    sp.check_call(
        shlex.split(nbconvert_command),
        stdout=sp.DEVNULL,
        stderr=sp.DEVNULL,
    )

    ipython_command = "ipython {}".format(py_path)

    print('    ', ipython_command, ' >/dev/null 2>&1')
    return_code = sp.call(
        shlex.split(ipython_command),
        stdout=sp.DEVNULL,
        stderr=sp.DEVNULL,
    )
    return_codes.append(return_code)
    print('--> return_code:', return_code)
    print()

sys.exit(max(return_codes))
