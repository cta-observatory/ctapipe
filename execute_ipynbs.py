#!/usr/bin/env python3
import sys
from glob import glob
import shlex
import subprocess as sp
from collections import namedtuple

xfail = [
    'table_writer_reader.ipynb',
    'check_calib.ipynb',
    'vpython_display.ipynb',
]
TIMEOUT_PER_CELL = 240  # seconds


def main():
    results = {}
    print('testing notebooks: ', end='', flush=True)
    for path in detect_notbooks():

        if is_xfail(path):
            print('X', end='', flush=True)
            results[path] = fake_xfail_result()
        else:
            result = sp.run(
                shlex.split(command(path, timeout=TIMEOUT_PER_CELL)),
                stdout=sp.PIPE,
                stderr=sp.PIPE
            )
            results[path] = result

            if result.returncode == 0:
                print('.', end='', flush=True)
            else:
                print('F', end='', flush=True)
    print()
    return results


def detect_notbooks():
    return sorted(glob('**/*.ipynb', recursive=True))


def command(path, timeout=120):
    # --ExecutePreprocessor.timeout is the timeout in seconds per cell
    return '''
    jupyter nbconvert
    --execute
    --ExecutePreprocessor.timeout={timeout}
    '{path}'
    '''.format(
        path=path, timeout=timeout
    )


def is_xfail(path):
    for s in xfail:
        if s in path:
            return True


FakeCompletedProcess = namedtuple(
    'FakeCompletedProcess',
    ['args', 'returncode', 'stdout', 'stderr']
)


def fake_xfail_result():
    ''' returns a namedtuple with the same fields
    as a real subprocess.CompletedProcess
    just saying that this process did not run at all.
    '''
    return FakeCompletedProcess(
        args='None',
        returncode=-1,
        stderr=b'not executed: expected to fail',
        stdout=b''
    )


if __name__ == '__main__':
    results = main()

    if any([r.returncode != 0 for r in results.values()]):
        print('Captured stderr')
        print('=' * 70)
        for path, result in results.items():
            if result.returncode != 0:
                print(path)
                print(result.stderr.decode('utf8'))
                print('=' * 70)

    sys.exit(max([r.returncode for r in results.values()]))
