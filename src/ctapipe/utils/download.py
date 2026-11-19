"""Download remote files and reuse local copies in the dataset cache.

Downloads use partial files until they complete. A lock coordinates downloads
to the same cache entry across processes.
"""

import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm.auto import tqdm

from ..version import __version__
from .filelock import FileLock

__all__ = ["download_file", "download_cached", "download_file_cached"]

log = logging.getLogger(__name__)

USER_AGENT = f"ctapipe/{__version__}"


def download_file(url, path, auth=None, chunk_size=10240, progress=False):
    """
    Download a file.

    Will write to ``path + '.part'`` while downloading and rename after successful
    download to the final name.

    Parameters
    ----------
    url : str or url
        The URL to download
    path : pathlib.Path or str
        Where to store the downloaded data.
    auth : tuple or requests.auth.AuthBase or None, optional
        Username and password pair or authentication handler for the HTTP request.
    chunk_size : int
        Chunk size for writing the data file, 10 kB by default.
    progress : bool, optional
        Show a progress bar while downloading.
    """
    log.info(f"Downloading {url} to {path}")
    name = urlparse(url).path.split("/")[-1]
    path = Path(path)
    part_file = None

    headers = {"User-Agent": USER_AGENT}
    with requests.get(url, stream=True, auth=auth, timeout=5, headers=headers) as r:
        # make sure the request is successful
        r.raise_for_status()

        total = float(r.headers.get("Content-Length", float("inf")))
        pbar = tqdm(
            total=total,
            disable=not progress,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {name}",
        )

        try:
            # open a .part file to avoid creating
            # a broken file at the intended location
            part_file = path.with_suffix(path.suffix + ".part")

            part_file.parent.mkdir(parents=True, exist_ok=True)
            with part_file.open("wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
        except BaseException:  # we really want to catch everything here
            # cleanup part file if something goes wrong
            if part_file is not None and part_file.is_file():
                part_file.unlink()
            raise

    # when successful, move to intended location
    part_file.rename(path)


def get_cache_path(url, cache_name="ctapipe", env_override="CTAPIPE_CACHE"):
    if os.getenv(env_override):
        base = Path(os.environ[env_override])
    else:
        base = Path.home() / ".cache" / cache_name

    url = urlparse(url)

    path = os.path.join(url.netloc.rstrip("/"), url.path.lstrip("/"))
    path = base / path
    return path


def download_cached(
    url, cache_name="ctapipe", auth=None, env_prefix="CTAPIPE_DATA_", progress=False
) -> Path:
    """Get the local path to a URL, downloading it on a cache miss.

    Use an existing cached file without contacting the server. A file lock
    prevents concurrent downloads to the same cache entry.

    Parameters
    ----------
    url : str
        URL of the file to retrieve. Its host and path identify the cache entry.
    cache_name : str, optional
        Directory name under the user's ``.cache`` directory. If
        ``CTAPIPE_CACHE`` is set, it specifies the cache root instead.
    auth : bool or tuple of str or requests.auth.AuthBase or None, optional
        Authentication passed to the HTTP request. Use a (username, password)
        pair or set this to True to read the pair from the environment.
    env_prefix : str, optional
        Prefix for the ``USER`` and ``PASSWORD`` environment variables read
        when ``auth=True`` and the file needs to be downloaded.
    progress : bool, optional
        Show a progress bar while downloading the file.

    Returns
    -------
    pathlib.Path
        Path to the cached file.

    Raises
    ------
    KeyError
        If ``auth=True``, the file is not cached, and a required credential
        variable is missing.
    """
    path = get_cache_path(url, cache_name=cache_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = path.with_suffix(path.suffix + ".lock")

    with FileLock(lock_file):
        # if we already downloaded the file, just use it
        if path.is_file():
            log.debug(f"{url} is available in cache.")
            return path

        if auth is True:
            try:
                auth = (
                    os.environ[env_prefix + "USER"],
                    os.environ[env_prefix + "PASSWORD"],
                )
            except KeyError:
                raise KeyError(
                    f'You need to set the env variables "{env_prefix}USER"'
                    f' and "{env_prefix}PASSWORD" to download test files.'
                ) from None

        download_file(url=url, path=path, auth=auth, progress=progress)
        return path


def download_file_cached(
    name,
    cache_name="ctapipe",
    auth=None,
    env_prefix="CTAPIPE_DATA_",
    default_url="https://minio-cta.zeuthen.desy.de/dpps-testdata-public/data/",
    progress=False,
):
    """
    Download a file and cache it locally.

    Reuse files in ``$HOME/.cache/<cache_name>`` without downloading them again.

    Parameters
    ----------
    name : str or pathlib.Path
        the name of the file, relative to the data server url
    cache_name : str
        What name to use for the cache directory
    auth : True, None or tuple of (username, password)
        Authentication data for the request. Will be passed to ``requests.get``.
        If ``True``, read username and password for the request from
        the env variables ``env_prefix + 'USER'`` and ``env_prefix + PASSWORD``
    env_prefix : str
        Prefix for the environt variables used for overriding the URL,
        and providing username and password in case authentication is required.
    default_url : str
        The default url from which to download ``name``, can be overridden
        by setting the env variable ``env_prefix + URL``
    progress : bool
        Whether or not to show a progress bar for downloads

    Returns
    -------
    path: pathlib.Path
        the full path to the downloaded data.
    """
    log.debug("Fetching '%s' from cache or remote server.", name)

    base_url = os.environ.get(env_prefix + "URL", default_url).rstrip("/")
    url = base_url + "/" + str(name).lstrip("/")
    return download_cached(
        url, cache_name=cache_name, auth=auth, env_prefix=env_prefix, progress=progress
    )
