import requests
import os
from pathlib import Path
import logging
from tqdm import tqdm
from urllib.parse import urlparse


log = logging.getLogger(__name__)


def download_file(url, path, auth=None, chunk_size=10240, progress=False):
    """
    Download a file. Will write to ``path + '.part'`` while downloading
    and rename after successful download to the final name.

    Parameters
    ----------
    url: str or url
        The URL to download
    path: pathlib.Path or str
        Where to store the downloaded data.
    auth: None or tuple of (username, password) or a request.AuthBase instance.
    chunk_size: int
        Chunk size for writing the data file, 10 kB by default.
    """
    log.info(f"Downloading {url} to {path}")

    r = requests.get(url, stream=True, auth=auth)
    name = urlparse(url).path.split("/")[-1]

    # make sure the request is successful
    r.raise_for_status()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # open a .part file to avoid a creating a broken file at the intended location
    part_file = path.with_suffix(path.suffix + ".part")

    total = float(r.headers.get("Content-Length", float("inf")))
    pbar = tqdm(
        total=total,
        disable=not progress,
        unit="B",
        unit_scale=True,
        desc=f"Downloading {name}",
    )

    try:
        with part_file.open("wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))
    except Exception:
        # cleanup part file if something goes wrong
        if part_file.is_file():
            part_file.unlink()
        raise

    # when successful, move to intended location
    part_file.rename(path)


def get_cache_path(path, cache_name="lstchain"):
    # need to make it relative
    path = str(path).lstrip("/")
    path = Path(os.environ["HOME"]) / ".cache" / cache_name / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def download_file_cached(
    name,
    cache_name="ctapipe",
    auth=None,
    env_prefix="CTAPIPE_DATA_",
    default_url="http://cccta-dataserver.in2p3.fr/data/",
    progress=False,
):
    """
    Downloads a file from a dataserver and caches the result locally
    in ``$HOME/.cache/<cache_name>``.
    If the file is found in the cache, no new download is performed.

    Parameters
    ----------
    name: str or pathlib.Path
        the name of the file, relative to the data server url
    cache_name: str
        What name to use for the cache directory
    env_prefix: str
        Prefix for the environt variables used for overriding the URL,
        and providing username and password in case authentication is required.
    auth: True, None or tuple of (username, password)
        Authentication data for the request. Will be passed to ``requests.get``.
        If ``True``, read username and password for the request from
        the env variables ``env_prefix + 'USER'`` and ``env_prefix + PASSWORD``
    default_url: str
        The default url from which to download ``name``, can be overriden
        by setting the env variable ``env_prefix + URL``

    Returns
    -------
    path: pathlib.Path
        the full path to the downloaded data.
    """
    path = get_cache_path(name, cache_name=cache_name)

    # if we already dowloaded the file, just use it
    if path.is_file():
        log.debug(f"File {name} is available in cache.")
        return path

    log.debug(f"File {name} is not available in cache, downloading.")

    base_url = os.environ.get(env_prefix + "URL", default_url).rstrip("/")
    url = base_url + "/" + name.lstrip("/")

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
