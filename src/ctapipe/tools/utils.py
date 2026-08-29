# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utils to create scripts and command-line tools"""

import argparse
import ast
import logging
from collections import OrderedDict
from importlib.metadata import distribution
from importlib.util import find_spec
from pathlib import Path

import numpy as np
from astropy.table import vstack

from ..core.traits import Int
from ..exceptions import TooFewEvents
from ..instrument import TelescopeDescription
from ..io import TableLoader
from ..reco.preprocessing import check_valid_rows
from ..reco.sklearn import SKLearnReconstructor

LOG = logging.getLogger(__name__)


__all__ = [
    "ArgparseFormatter",
    "get_parser",
    "get_installed_tools",
    "get_all_descriptions",
]


class ArgparseFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter
):
    """ArgumentParser formatter_class argument."""

    pass


def get_parser(function=None, description="N/A"):
    """Make an ArgumentParser how we like it."""
    if function:
        description = function.__doc__
    parser = argparse.ArgumentParser(
        description=description, formatter_class=ArgparseFormatter
    )
    return parser


def get_installed_tools():
    """Get list of installed scripts via ``pkg-resources``.

    See https://setuptools.pypa.io/en/latest/pkg_resources.html#convenience-api
    """
    console_tools = {
        ep.name: ep.value
        for ep in distribution("ctapipe").entry_points
        if ep.group == "console_scripts"
    }
    return console_tools


def get_all_descriptions():
    tools = get_installed_tools()

    descriptions = OrderedDict()
    for name, value in tools.items():
        module_name, _ = value.split(":")
        descrip = ast.get_docstring(
            ast.parse(Path(find_spec(module_name).origin).read_text())
        )
        if descrip is not None:
            descriptions[name] = descrip.replace("\n", " ")
        else:
            descriptions[name] = "[no documentation. Please add a docstring]"

    return descriptions


def _add_optional_columns(table, columns, optional_columns):
    for column in optional_columns:
        if column in table.colnames and column not in columns:
            columns.append(column)


def read_training_events(
    loader: TableLoader,
    chunk_size: Int,
    telescope_type: TelescopeDescription,
    reconstructor: type[SKLearnReconstructor],
    feature_names: list[str],
    rng: np.random.Generator,
    log=LOG,
    n_events=None,
    optional_columns: list[str] | None = None,
):
    """Chunked loading of events for training ML models"""
    chunk_iterator = loader.read_telescope_events_chunked(
        chunk_size,
        telescopes=[telescope_type],
        true_parameters=False,
        instrument=True,
        observation_info=True,
        pointing=True,
    )
    table = []
    n_events_in_file = 0
    n_valid_events_in_file = 0
    n_non_predictable = 0
    columns = feature_names.copy()

    for chunk, (_, _, table_chunk) in enumerate(chunk_iterator):
        log.debug("Events read from chunk %d: %d", chunk, len(table_chunk))
        n_events_in_file += len(table_chunk)

        if len(table) == 0 and optional_columns is not None:
            _add_optional_columns(table_chunk, columns, optional_columns)

        mask = reconstructor.quality_query.get_table_mask(table_chunk)
        table_chunk = table_chunk[mask]
        log.debug(
            "Events in chunk %d after applying quality_query: %d",
            chunk,
            len(table_chunk),
        )
        n_valid_events_in_file += len(table_chunk)

        table_chunk = reconstructor.feature_generator(
            table_chunk, subarray=loader.subarray
        )
        table_chunk = table_chunk[columns]

        valid = check_valid_rows(table_chunk)
        if not np.all(valid):
            n_non_predictable += np.sum(~valid)
            table_chunk = table_chunk[valid]

        table.append(table_chunk)

    table = vstack(table)
    log.info("Events read from input: %d", n_events_in_file)
    log.info("Events after applying quality query: %d", n_valid_events_in_file)

    if len(table) == 0:
        raise TooFewEvents(
            f"No events after quality query for telescope type {telescope_type}"
        )

    if n_non_predictable > 0:
        log.warning("Dropping %d non-predictable events.", n_non_predictable)

    if n_events is not None:
        if n_events > len(table):
            log.warning(
                "Number of events in table (%d) is less"
                " than requested number of events %d",
                len(table),
                n_events,
            )
        else:
            log.info("Sampling %d events", n_events)
            idx = rng.choice(len(table), n_events, replace=False)
            idx.sort()
            table = table[idx]

    return table
