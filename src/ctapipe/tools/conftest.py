from importlib.util import find_spec

if find_spec("pyirf") is None:
    collect_ignore = ["compute_irf.py", "optimize_event_selection.py"]
