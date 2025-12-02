"""Script to download astropy data needed for coordinate transformations."""

from argparse import ArgumentParser
from contextlib import ExitStack
from pathlib import Path

from astropy.config.paths import set_temp_cache
from astropy.coordinates import EarthLocation
from astropy.utils.data import clear_download_cache, download_file
from astropy_iers_data import (
    IERS_A_URL,
    IERS_A_URL_MIRROR,
    IERS_B_URL,
    IERS_LEAP_SECOND_URL,
    IERS_LEAP_SECOND_URL_MIRROR,
)

__all__ = [
    "main",
]


parser = ArgumentParser()
parser.add_argument(
    "-d",
    "--directory",
    help=(
        "Directory to store astropy cache data."
        "If not given, astropy's default will be used."
    ),
    type=Path,
)
parser.add_argument(
    "-f", "--force", help="Clear cache before re-creating it.", action="store_true"
)


def main(args=None):
    """
    Download data needed for astropy coordinate transformations into a cache directory.

    See the `astropy docs <https://docs.astropy.org/en/stable/utils/data.html>`_
    for why and when this might be useful.
    """
    args = parser.parse_args(args)

    ctx = ExitStack()
    with ctx:
        if args.directory is not None:
            args.directory.mkdir(exist_ok=True, parents=True)
            ctx.enter_context(set_temp_cache(args.directory))

        if args.force:
            # force re-creation
            clear_download_cache()

        # IERS data
        download_file(
            IERS_A_URL,
            sources=[IERS_A_URL, IERS_A_URL_MIRROR],
            cache="update",
        )
        # no secondary url for IERS_B ?!
        download_file(IERS_B_URL, cache="update")

        download_file(
            IERS_LEAP_SECOND_URL,
            sources=[IERS_LEAP_SECOND_URL, IERS_LEAP_SECOND_URL_MIRROR],
            cache="update",
        )

        # EarthLocation.of_site names
        EarthLocation.of_site("Roque de los Muchachos")


if __name__ == "__main__":
    main()
