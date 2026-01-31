"""
Handles reading of monitoring data from FITS files
"""

import logging

from astropy.io import fits
from astropy.table import Table

from ..core import Provenance
from ..core.traits import List
from ..core.traits import Path as PathTrait
from ..exceptions import InputMissing
from .fitsdataformat import DESCRIPTION, MONITORING_SOURCES, NAME, SOURCE
from .monitoringsource import MonitoringSource

__all__ = ["FITSMonitoringSource"]

logger = logging.getLogger(__name__)


class FITSMonitoringSource(MonitoringSource):
    """
    Class for reading FITS monitoring data as a `~ctapipe.io.MonitoringSource`.

    This class provides access to FITS monitoring data from CTAO DL0 files,
    including telescope pointing, FRAM, and LiDAR data.

    Note: At the DL0 level, FITS monitoring files are organized with one file
    per subsystem per monitoring type. For example:
    - One file per telescope for drive pointing data
    - Separate files per telescope for bending model corrections
    - Separate files per telescope for starguider corrections
    - One file per FRAM device

    Implementation note: This implementation is based on the CTAO DL0 Data Model
    specification. As of the current specification version, specific FITS header
    keywords and table column names are not yet finalized. This implementation
    uses placeholder values that will need to be updated once the detailed FITS
    format specification is available.
    """

    input_files = List(
        PathTrait(exists=True, directory_ok=False),
        default_value=[],
        help="List of paths to the FITS input files containing monitoring data. "
        "Each file should contain monitoring data for a single subsystem/device.",
    ).tag(config=True)

    def __init__(self, subarray=None, config=None, parent=None, **kwargs):
        super().__init__(
            subarray=subarray,
            config=config,
            parent=parent,
            **kwargs,
        )

        # Check if input_files list is empty
        if not self.input_files:
            raise InputMissing(
                "No input files provided. Please specify a list of input file(s) "
                "via configuration by `--FITSMonitoringSource.input_files` "
                "or using as an argument <input_files> in the constructor."
            )

        # Initialize attributes
        self._monitoring_sources = {}  # Maps source_name -> table data

        # Process all monitoring files
        for file in self.input_files:
            self._process_single_file(file)

    def _process_single_file(self, file):
        """Process a single monitoring file (one subsystem/monitoring type per file)."""
        # Add the file to the provenance
        Provenance().add_input_file(
            str(file),
            role="Monitoring",
        )

        with fits.open(file) as hdul:
            # At DL0 level, each file contains one monitoring type
            # Look for the first binary table HDU with data
            for hdu in hdul:
                if isinstance(hdu, fits.BinTableHDU):
                    # Automatically detect monitoring source from header
                    source_name = self._detect_monitoring_source(hdu)

                    if source_name is None:
                        logger.warning(
                            f"Could not determine monitoring source for HDU {hdu.name} "
                            f"in file {file}. Skipping."
                        )
                        continue

                    # Validate that the HDU has required columns
                    if not self._validate_table_structure(hdu, source_name):
                        logger.warning(
                            f"HDU {hdu.name} in file {file} missing required columns "
                            f"for {source_name}. Skipping."
                        )
                        continue

                    # Process the table
                    self._process_table(hdu, source_name)

                    # Since each file has one monitoring type, we can break
                    break

    def _detect_monitoring_source(self, hdu):
        """Detect monitoring source from FITS header, returns source name or None."""
        if SOURCE in hdu.header:
            source_name = hdu.header[SOURCE]

            if source_name not in MONITORING_SOURCES:
                logger.warning(
                    f"Unknown monitoring source '{source_name}' in header. "
                    f"Registered sources: {list(MONITORING_SOURCES.keys())}"
                )
                return None

            return source_name

        logger.debug(
            f"No {SOURCE} keyword found in header. "
            f"Cannot auto-detect monitoring source."
        )
        return None

    def _validate_table_structure(self, hdu, source_name):
        """Check if HDU has all required columns for the monitoring source."""
        if source_name not in MONITORING_SOURCES:
            logger.error(
                f"Unknown monitoring source '{source_name}'. "
                f"No validation rules defined."
            )
            return False

        monitoring_source = MONITORING_SOURCES[source_name]
        required_cols = set(monitoring_source.data_struct.keys())
        table_cols = set(hdu.columns.names)

        missing_cols = required_cols - table_cols

        if missing_cols:
            logger.error(
                f"Table missing required columns for {source_name}: {missing_cols}. "
                f"Found columns: {table_cols}"
            )
            return False

        logger.debug(
            f"Table structure validated for {source_name}. "
            f"All required columns present."
        )
        return True

    def _process_table(self, hdu, source_name):
        """Read and store monitoring table from FITS HDU."""
        # Read the table
        table = Table.read(hdu)

        # Extract FITS coordinate system metadata if present
        metadata = {}
        for keyword in ["RADESYS", SOURCE, NAME, DESCRIPTION]:
            if keyword in hdu.header:
                metadata[keyword] = hdu.header[keyword]

        # Store metadata in table meta
        if metadata:
            table.meta.update(metadata)

        # Store by source name
        self._monitoring_sources[source_name] = table

    @property
    def is_simulation(self):
        """True for files from simulation (always False for FITS files)."""
        return False

    @property
    def monitoring_sources(self):
        """Return dict of all loaded monitoring sources."""
        return self._monitoring_sources

    def get_table(self, source_name: str):
        """
        Get monitoring table for a source.

        Parameters
        ----------
        source_name : str
            The monitoring source name (e.g., "LST1Drive", "STPN-01")

        Returns
        -------
        astropy.table.Table
            The monitoring data table

        Raises
        ------
        KeyError
            If the source is not loaded
        """
        if source_name not in self._monitoring_sources:
            raise KeyError(
                f"Monitoring source '{source_name}' not available. "
                f"Available sources: {list(self._monitoring_sources.keys())}"
            )
        return self._monitoring_sources[source_name]

    def fill_monitoring_container(self, event):
        raise NotImplementedError(
            "fill_monitoring_container not yet implemented for FITSMonitoringSource"
        )
