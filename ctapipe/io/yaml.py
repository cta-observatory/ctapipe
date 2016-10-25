
from astropy.table import Table
from astropy.units import Quantity
from collections import OrderedDict
import ruamel.yaml as yaml
import ruamel.yaml.comments as Comments
from ruamel.yaml.nodes       import ScalarNode
from ruamel.yaml.loader      import Loader
from ruamel.yaml.dumper      import Dumper
from ruamel.yaml.constructor import Constructor
from ruamel.yaml.representer import Representer

# Handler for YAML call inside YAML
class YAMLFile(str):
    #yaml_tag = u'!yaml'
    def __init__(self, filename):
        self.filename = filename
    def read(self):
        with open(self.filename,"r") as fin:
            return(yaml.load(fin))
    def append(self,data):
        with open(self.filename,"a+") as fout:
            yaml.dump(data, fout, Dumper=Dumper)
    def write(self,data):
        with open(self.filename,"r+") as fout:
            yaml.dump(data, fout, Dumper=Dumper)
    def truncate(self):
        with open(self.filename,"w") as fout:
            fout.write("")
    def __new__(cls, a):
        return str.__new__(cls, a)
    def __repr__(self):
        return "YAMLFile(%s)" % self

def representer_yamlfile(dumper, data):
    return dumper.represent_scalar(u'!yaml', data.filename)

def constructor_yamlfile(loader, node):
    if isinstance(node, ScalarNode):
        # Load the contents wherever possible (file exists)
        try:
            return File(node.value).read()
        except FileNotFoundError:
            return File(node.value) # Alternatively, give the object.

yaml.add_representer(YAMLFile, representer_yamlfile)
yaml.add_constructor(u'!yaml', constructor_yamlfile)
