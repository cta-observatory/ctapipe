from time import sleep
import threading
import subprocess
import os


"""
Base class for stage (producer, stager and consumer) that execute system command with the following format:
cmd -i inputfile -o outputfile 
Other options can be add with options keyword in configuration file
"""

class BaseProcess():
	def __init__(self,configuration=None):
		self.configuration = configuration
		self.section_name = None
		self.source_dir = None
		self.output_dir = None
		self.executable = None
		self.out_extension = None
		self.options = None

	"""
	Inititalises source_dir, output_dir, executable, out_extension and options form configuration file
	Create output_dir if not already exists
	Verify self.executable is defined
	REturn true on success, false otherwise
	"""
	def init(self):
		self.source_dir = self.configuration.get('source_dir', section=self.section_name)
		self.output_dir = self.configuration.get('output_dir', section=self.section_name)
		if self.output_dir != None :
			# If output directory is configure, create it now if it does not already exit
			if  not os.path.exists(self.output_dir):
				try: os.mkdir(self.output_dir)
				except OSError:
					print(self.section_name, ": could not create output directory", self.output_dir)
					return False
		self.executable =  self.configuration.get('executable', section=self.section_name)
		self.out_extension = self.configuration.get('out_extension', section=self.section_name)
		self.options = self.configuration.get('options', section=self.section_name)
		if self.executable  == None: 
			print(self.section_name," :configuration error: ", 'self.executable:', self.executable)
			return False
		return True
	

	"""
	Parameters:
	-----------
	input_file : str
		input_file on which execute command
	Returns:
	-----------
	a command  in a python list form and an output_file full path name
	"""

	def build_command(self,input_file):
		# remove full path before file name
		output_file = input_file.split('/')[-1]
		# replace extension if need
		if self.out_extension != None:
			output_file = output_file.rsplit('.',2)[0] # remove old_extension
			output_file+= "." + self.out_extension     # add new extension
		# build command
		if self.source_dir != None: input_file = self.source_dir + '/' + input_file
		cmd = [self.executable,'-i', input_file]
		if  self.output_dir != None:
			cmd.append("-o")
			output_file =  self.output_dir+"/"+output_file
			cmd.append(output_file)
		if self.options != None: cmd.append(self.options)
		return cmd,output_file