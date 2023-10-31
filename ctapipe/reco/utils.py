def add_defaults_and_meta(table, container, prefix=None, add_tel_prefix=False):
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
    add_tel_prefix : bool
        If True, add a ``tel_`` prefix to the column names to signal it's
        telescope-wise quantity
    """
    if prefix is None:
        prefix = container.default_prefix

    for name, field in container.fields.items():
        if add_tel_prefix and name == "telescopes":
            continue

        if add_tel_prefix:
            colname = f"{prefix}_tel_{name}"
        else:
            colname = f"{prefix}_{name}"

        if colname not in table.colnames and field.default is not None:
            table[colname] = field.default

        if colname in table.colnames:
            table[colname].description = field.description
