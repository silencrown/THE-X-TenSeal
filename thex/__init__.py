import os
import glob
import importlib

# TODO: add all files in the_x/ to the __all__ lists
__all__ = []

current_directory = os.path.dirname(os.path.abspath(__file__))
pattern = os.path.join(current_directory, '[!_]*.py')

for file in glob.glob(pattern):
    module_name = os.path.splitext(os.path.basename(file))[0]
    module = importlib.import_module(f'.{module_name}', package=__package__)

    for attr_name in dir(module):
        if not attr_name.startswith('_'):
            attr = getattr(module, attr_name)
            __all__.append(attr_name)
            globals()[attr_name] = attr
