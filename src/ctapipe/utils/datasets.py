# Licensed under a 3-clause BSD style license - see LICENSE.rst
import json
import logging
import os
import re
from functools import partial
from pathlib import Path

import yaml
from astropy.table import Table
from requests.exceptions import HTTPError

from .download import download_file_cached, get_cache_path

try:
    import ctapipe_resources

    has_resources = True
except ImportError:
    has_resources = False

from importlib.resources import files

from ..core import Provenance

logger = logging.getLogger(__name__)

__all__ = [
    "get_dataset_path",
    "find_in_path",
    "find_all_matching_datasets",
    "get_default_url",
    "DEFAULT_URL",
]


#: default base URL for downloading datasets
DEFAULT_URL = "https://minio-cta.zeuthen.desy.de/dpps-testdata-public/data/ctapipe-test-data/v1.1.0/"


def get_default_url():
    """Get the default download url for datasets

    First tries to look-up CTAPIPE_DATASET_URL and then falls
    back to ``DEFAULT_URL``
    """
    return os.getenv("CTAPIPE_DATASET_URL", DEFAULT_URL)


def get_searchpath_dirs(searchpath=None, url=None):
    """returns a list of dirs in specified searchpath"""
    if url is None:
        url = get_default_url()

    if searchpath is None:
        searchpath = os.getenv("CTAPIPE_SVC_PATH")

    if searchpath == "" or searchpath is None:
        searchpaths = []
    else:
        searchpaths = [Path(p) for p in os.path.expandvars(searchpath).split(":")]
    searchpaths.append(get_cache_path(url))
    return searchpaths


def find_all_matching_datasets(
    pattern,
    searchpath=None,
    regexp_group=None,
    url=None,
):
    """
    Returns a list of resource names (or substrings) matching the given
    pattern, searching first in searchpath (a colon-separated list of
    directories) and then in the ctapipe_resources module)

    Parameters
    ----------
    pattern: str
       regular expression to use for matching
    searchpath: str
       colon-seprated list of directories in which to search, defaulting to
       CTAPIPE_SVC_PATH environment variable
    regexp_group: int
       if not None, return the regular expression group indicated (assuming
       pattern has a group specifier in it)

    Returns
    -------
    list(str):
       resources names, use get_dataset_path() to retrieve the full filename
    """
    if url is None:
        url = get_default_url()

    results = set()

    if searchpath is None:
        searchpath = os.getenv("CTAPIPE_SVC_PATH")
    search_path_dirs = get_searchpath_dirs(searchpath, url=url)

    # first check search path
    for path in search_path_dirs:
        if path.is_dir():
            for entry in path.iterdir():
                match = re.match(pattern, entry.name)
                if match:
                    if regexp_group is not None:
                        results.add(match.group(regexp_group))
                    else:
                        results.add(entry)

    # then check resources module
    if has_resources:
        for resource in files("ctapipe_resources").iterdir():
            match = re.match(pattern, resource.name)
            if match:
                if regexp_group is not None:
                    results.add(match.group(regexp_group))
                else:
                    results.add(resource)

    return list(results)


def find_in_path(filename, searchpath, url=None):
    """
    Search in searchpath for filename, returning full path.

    Parameters
    ----------
    searchpath: str
        colon-separated list of directories (like PATH)
    filename:
        filename to find

    Returns
    -------
    full path to file if found, None otherwise

    """
    if url is None:
        url = get_default_url()

    for directory in get_searchpath_dirs(searchpath, url=url):
        path = directory / filename
        if path.exists():
            return path

    return None


def get_dataset_path(filename, url=None):
    """
    Returns the full file path to an auxiliary dataset needed by
    ctapipe, given the dataset's full name (filename with no directory).

    This will first search for the file in directories listed in
    the environment variable CTAPIPE_SVC_PATH (if set), and if not found,
    will look in the ctapipe_resources module
    (if installed with the ctapipe-extra package), which contains the defaults.

    Parameters
    ----------
    filename: str
        name of dataset to fetch

    Returns
    -------
    string with full path to the given dataset
    """
    searchpath = os.getenv("CTAPIPE_SVC_PATH")
    if url is None:
        url = get_default_url()

    if searchpath:
        filepath = find_in_path(filename=filename, searchpath=searchpath, url=url)

        if filepath:
            return filepath

    if has_resources:
        logger.debug(
            "Resource '{}' not found in CTAPIPE_SVC_PATH, looking in "
            "ctapipe_resources...".format(filename)
        )

        return Path(ctapipe_resources.get(filename))

    # last, try downloading the data
    try:
        return download_file_cached(filename, default_url=url, progress=True)
    except HTTPError as e:
        # let 404 raise the FileNotFoundError instead of HTTPError
        if e.response.status_code != 404:
            raise

    raise FileNotFoundError(
        f"Couldn't find resource: '{filename}',"
        " You might want to install ctapipe_resources"
    )


def try_filetypes(basename, role, file_types, url=None, **kwargs):
    """
    Get the contents of dataset as an `astropy.table.Table` object from
    different file types if available.

    Parameters
    ----------
    basename: str
        base-name (without extension) of the dataset
    role: str
        Provenance role. It should be set to the CTA data hierarchy name when
        possible (e.g. dl1.sub.svc.arraylayout). This will be recorded in the
        provenance system.
    file_types: dict
        Mapping of file extensions to readers.
    url : str
        URL where the dataset is stored.
    kwargs:
        extra arguments to pass to Table.read()

    Returns
    -------
    table : astropy.table.Table
        Table containing the data in a dataser from an available file
        type entry.

    """
    if url is None:
        url = get_default_url()
    path = None

    # look first in search paths (includes cache)
    # so we respect user provided paths and don't have to try non-existing downloads
    search_paths = get_searchpath_dirs(url=url)
    for search_path in search_paths:
        for ext, reader in file_types.items():
            filename = basename + ext

            if (search_path / filename).exists():
                path = search_path / filename
                break

        if path is not None:
            break

    # no cache hit
    if path is None:
        for ext, reader in file_types.items():
            filename = basename + ext
            try:
                path = get_dataset_path(filename, url=url)
                break
            except (FileNotFoundError, HTTPError):
                pass

    if path is not None:
        table = reader(path, **kwargs)
        Provenance().add_input_file(path, role)
        return table

    raise FileNotFoundError(
        "Couldn't find any file: {}[{}]".format(basename, ", ".join(file_types))
    )


def get_table_dataset(table_name, role="resource", **kwargs):
    """
    get a tabular dataset as an `astropy.table.Table` object

    Parameters
    ----------
    table_name: str
        base name of table, without file extension
    role: str
        should be set to the CTA data hierarchy name when possible (e.g.
        dl1.sub.svc.arraylayout). This will be recorded in the provenance
        system.
    kwargs:
        extra arguments to pass to Table.read()

    Returns
    -------
    Table
    """

    # a mapping of types (keys) to any extra keyword args needed for
    # table.read()
    table_types = {
        ".fits.gz": Table.read,
        ".fits": Table.read,
        ".ecsv": partial(Table.read, format="ascii.ecsv"),
        ".ecsv.txt": partial(Table.read, format="ascii.ecsv"),
    }

    return try_filetypes(table_name, role, table_types, **kwargs)


def get_structured_dataset(basename, role="resource", **kwargs):
    """
    find and return a YAML or JSON dataset as a dictionary

    Parameters
    ----------
    basename: str
        base-name (without extension) of the dataset
    role: str
        provenence role
    kwargs: dict
        any other arguments to pass on to yaml or json loader.

    Returns
    -------
    dict:
       dictionary of data in the file
    """

    def load_yaml(path, **kwargs):
        with open(path, "rb") as f:
            return yaml.safe_load(f, **kwargs)

    def load_json(path, **kwargs):
        with open(path, "rb") as f:
            return json.load(f, **kwargs)

    # a mapping of types (keys) to any extra keyword args needed for
    # table.read()
    structured_types = {".yaml": load_yaml, ".yml": load_yaml, ".json": load_json}
    return try_filetypes(basename, role, structured_types, **kwargs)


def resource_file(filename):
    """Get the absolute path of ctapipe resource files."""
    return files("ctapipe").joinpath("resources", filename)
