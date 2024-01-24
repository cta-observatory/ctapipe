from astropy.table import QTable, Table

from ..containers import ArrayEventContainer

__all__ = ["collect_features"]


def collect_features(
    event: ArrayEventContainer, tel_id: int, subarray_table=None
) -> Table:
    """Loop over all containers with features.

    Parameters
    ----------
    event : ArrayEventContainer
        The event container from which to collect the features
    tel_id : int
        The telscope id for which to collect the features
    subarray_table : Table
        The subarray as "to_table("joined")", to be added to the features.

    Returns
    -------
    Table
    """
    features = {}

    features.update(event.trigger.as_dict(recursive=False, flatten=True))

    features.update(
        event.dl1.tel[tel_id].parameters.as_dict(
            add_prefix=True,
            recursive=True,
            flatten=True,
        )
    )

    features.update(
        event.dl2.tel[tel_id].as_dict(
            add_prefix=True,
            recursive=True,
            flatten=True,
            add_key=False,  # prefix is already the map key for dl2 stuff
        )
    )

    features.update(
        event.dl2.stereo.as_dict(
            add_prefix=True,
            recursive=True,
            flatten=True,
            add_key=False,  # prefix is already the map key for dl2 stuff
        )
    )

    if subarray_table is not None:
        # to include units in features
        if not isinstance(subarray_table, QTable):
            subarray_table = QTable(subarray_table, copy=False)

        features.update(subarray_table.loc[tel_id])

    return Table({k: [v] for k, v in features.items()})
