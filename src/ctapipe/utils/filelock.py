import time
from pathlib import Path

from tqdm import tqdm


class FileLock:
    """
    Use a file as an interprocess lock.

    If the file exists, the lock will block the code until it does not
    exist anymore and then creates it itself.
    """

    def __init__(self, path, description="Another process is already running, waiting"):
        self.path = Path(path)
        self.description = description

    def __enter__(self):
        bar = None
        while True:
            try:
                self.path.open("x").close()
                break
            except FileExistsError:
                if bar is None:
                    bar = tqdm(
                        desc=self.description,
                        leave=False,
                        bar_format="{desc}: {elapsed_s:.1f}s",
                    )
                bar.update(1)
                time.sleep(0.1)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.path.exists():
            self.path.unlink()
