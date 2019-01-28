#!/usr/bin/env python3
import sys
import time
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
    for nb_path in detect_notbooks():

        if is_xfail(nb_path):
            print('X', end='', flush=True)
            results[nb_path] = fake_xfail_result()
        else:
            start_time = time.time()
            result = sp.run(
                shlex.split(command(nb_path, timeout=TIMEOUT_PER_CELL)),
                stdout=sp.PIPE,
                stderr=sp.PIPE
            )
            duration = time.time() - start_time
            results[nb_path] = result, duration

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
    ), 0.  # duration of xfails is set to zero


if __name__ == '__main__':
    results = main()

    if any([r.returncode != 0 for r, d in results.values()]):
        print('Captured stderr')
        print('=' * 70)
        for path, (result, duration) in results.items():
            if result.returncode != 0:
                print(path)
                print(result.stderr.decode('utf8'))
                print('=' * 70)

    # print durations longest first

    for nb_path, (r, d) in sorted(
        results.items(),
        key=lambda x: x[1][1],
        reverse=True
    ):
        print(f'{nb_path:.<70} duration {d:.1f}sec')

    sys.exit(max([r.returncode for r, d in results.values()]))
