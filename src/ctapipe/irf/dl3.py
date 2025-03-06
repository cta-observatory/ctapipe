from astropy.table import QTable

from ctapipe.compat import COPY_IF_NEEDED


def get_hdu_header_events():
    return {"HDUCLASS": "GADF", "HDUCLAS1": "EVENTS"}


def transform_events_columns_for_gadf_format(events):
    rename_from = ["event_id", "time", "reco_ra", "reco_dec", "reco_energy"]
    rename_to = ["EVENT_ID", "TIME", "RA", "DEC", "ENERGY"]

    rename_from_optional = [
        "multiplicity",
        "reco_glon",
        "reco_glat",
        "reco_alt",
        "reco_az",
        "reco_fov_lon",
        "reco_fov_lat",
        "reco_source_fov_offset",
        "reco_source_fov_position_angle",
        "gh_score",
        "reco_dir_uncert",
        "reco_energy_uncert",
        "reco_core_x",
        "reco_core_y",
        "reco_core_uncert",
        "reco_x_max",
        "reco_x_max_uncert",
    ]
    rename_to_optional = [
        "MULTIP",
        "GLON",
        "GLAT",
        "ALT",
        "AZ",
        "DETX",
        "DETY",
        "THETA",
        "PHI",
        "GAMANESS",
        "DIR_ERR",
        "ENERGY_ERR",
        "COREX",
        "COREY",
        "CORE_ERR",
        "XMAX",
        "XMAX_ERR",
    ]

    for i, c in enumerate(rename_from_optional):
        if c not in events.colnames:
            pass
            # self.log.warning(f"Optional column {c} is missing from the events_table.")
        else:
            rename_from.append(rename_from_optional[i])
            rename_to.append(rename_to_optional[i])

        renamed_events = QTable(events, copy=COPY_IF_NEEDED)
        renamed_events.rename_columns(rename_from, rename_to)
        renamed_events = renamed_events[rename_to]
        return renamed_events
