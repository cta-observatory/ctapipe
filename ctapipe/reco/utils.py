def add_defaults_and_meta(table, container, prefix=None, stereo=True):
    """
    Fill column descriptions and default values into table for container

    Parameters
    ----------
    table : astropy.table.Table
        the table to be filled
    container : ctapipe.core.Container
        the container class to add columns and descriptions to the table
    prefix : str
        prefix for the column names
    stereo : bool
        If False, add a ``tel_`` prefix to the column names to signal it's
        telescope-wise quantity
    """
    if prefix is None:
        prefix = container.default_prefix

    for name, field in container.fields.items():
        if not stereo and name == "telescopes":
            continue

        if stereo:
            colname = f"{prefix}_{name}"
        else:
            colname = f"{prefix}_tel_{name}"

        if colname not in table.colnames and field.default is not None:
            table[colname] = field.default

        if colname in table.colnames:
            table[colname].description = field.description
