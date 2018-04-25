#!/usr/bin/env python3
import sys
from glob import glob
import shlex
import subprocess as sp

return_codes = []
for path in sorted(glob('examples/notebooks/**/*.ipynb', recursive=True)):
    print('=' * 70)
    py_path = path.replace('.ipynb', '')
    print('testing:', path)

    # --ExecutePreprocessor.timeout=60 is the timeout in seconds per cell
    nbconvert_command = """
    jupyter nbconvert
    --to notebook
    --execute
    --ExecutePreprocessor.timeout=60
    '{}'
    """.format(path)
    print('    ', nbconvert_command)
    return_code = sp.call(
        shlex.split(nbconvert_command),
        stdout=sp.DEVNULL,
    )

    return_codes.append(return_code)
    print('--> return_code:', return_code)
    print('=' * 70)

sys.exit(max(return_codes))
