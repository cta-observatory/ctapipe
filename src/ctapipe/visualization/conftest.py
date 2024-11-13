import importlib

if importlib.util.find_spec("bokeh") is None:
    collect_ignore = ["bokeh.py"]
