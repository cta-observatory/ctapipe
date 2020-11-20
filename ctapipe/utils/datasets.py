# Licensed under a 3-clause BSD style license - see LICENSE.rst
import json
import logging
import os
import re
from functools import partial
from pathlib import Path

import yaml
from astropy.table import Table
from pkg_resources import resource_listdir
from requests.exceptions import HTTPError

from .download import download_file_cached, get_cache_path

try:
    import ctapipe_resources

    has_resources = True
except ImportError:
    has_resources = False


from ..core import Provenance

logger = logging.getLogger(__name__)

__all__ = ["get_dataset_path", "find_in_path", "find_all_matching_datasets"]


DEFAULT_URL = "http://cccta-dataserver.in2p3.fr/data/ctapipe-extra/v0.3.1/"


def get_searchpath_dirs(searchpath=os.getenv("CTAPIPE_SVC_PATH")):
    """ returns a list of dirs in specified searchpath"""
    if searchpath == "" or searchpath is None:
        searchpaths = []
    else:
        searchpaths = [Path(p) for p in os.path.expandvars(searchpath).split(":")]

    searchpaths.append(get_cache_path(""))

    return searchpaths


def find_all_matching_datasets(pattern, searchpath=None, regexp_group=None):
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
    results = set()

    if searchpath is None:
        searchpath = os.getenv("CTAPIPE_SVC_PATH")
    search_path_dirs = get_searchpath_dirs(searchpath)

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
        for resource in resource_listdir("ctapipe_resources", ""):
            match = re.match(pattern, resource)
            if match:
                if regexp_group is not None:
                    results.add(match.group(regexp_group))
                else:
                    results.add(Path(resource))

    return list(results)


def find_in_path(filename, searchpath):
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

    for directory in get_searchpath_dirs(searchpath):
        path = directory / filename
        if path.exists():
            return path

    return None


def get_dataset_path(filename):
    """
    Returns the full file path to an auxiliary dataset needed by
    ctapipe, given the dataset's full name (filename with no directory).

    This will first search for the file in directories listed in
    tne environment variable CTAPIPE_SVC_PATH (if set), and if not found,
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

    if searchpath:
        filepath = find_in_path(filename=filename, searchpath=searchpath)

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
        return download_file_cached(filename, default_url=DEFAULT_URL, progress=True)
    except HTTPError as e:
        # let 404 raise the FileNotFoundError instead of HTTPError
        if e.response.status_code != 404:
            raise

    raise FileNotFoundError(
        f"Couldn't find resource: '{filename}',"
        " You might want to install ctapipe_resources"
    )


def try_filetypes(basename, role, file_types, **kwargs):
    path = None

    # look first in cache so we don't have to try non-existing downloads
    for ext, reader in file_types.items():
        filename = basename + ext
        cache_path = get_cache_path(filename)
        if cache_path.exists():
            path = cache_path
            break

    # no cache hit
    if path is None:
        for ext, reader in file_types.items():
            filename = basename + ext
            try:
                path = get_dataset_path(filename)
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
