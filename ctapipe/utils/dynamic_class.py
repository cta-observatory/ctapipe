# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import sys
import logging
from importlib import import_module
from ctapipe.core.tool import Tool



__all__ = ['dynamic_class_from_module', ]



#def dynamic_class_from_module(class_name, module,  configuration=None):
def dynamic_class_from_module(class_name, module,  tool=None):
	"""
	Create an instance of a class from a configuration service section name

	Parameters:
	-----------
	module: str
		a python module file name. This module containe class to instantiate
	class_name: str
		python class name contained in module

	Returns:
	--------
	A python object instance of a class_name
	"""
	if  module == None :
		return None
	logger = logging.getLogger(__name__)
	try:
		_class = getattr(import_module(module), class_name)
		if isinstance(tool,Tool):
			instance = _class(tool,config=tool.config)
		else:
			instance = _class()
		return instance
	except AttributeError as e:
		logger.fatal("Could not create an instance of {} in module {}: {}"
			.format(class_name, module, e))
	except ImportError as e:
		logger.fatal("Could not create an instance of {} in module {}: {}"
			.format(class_name, module, e))
	except TypeError as e:
		logger.fatal("Could not create an instance of {} in module {}: {}"
			.format(class_name, module, e))
