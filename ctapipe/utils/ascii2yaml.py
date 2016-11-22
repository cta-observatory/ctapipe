#!/bin/env python2

from astropy.table import Table
from astropy.io import ascii
from collections import OrderedDict
import yaml
import numpy as np



infile = "/home/mnievas/digicam_pixels_mapping_V2.txt"
content = ascii.read(infile)
#content.write("test.ecsv",ascii.ecsv)


def get_flat_type(item):
    if 'int' in str(item.dtype):
        return(int(item))
    elif 'float' in str(item.dtype):
        return(float(item))
    else:
        print('Type not understood')
        return(None)

# Convert the Astropy table to dict
YamlObject = dict()

### OrderedDict are not defined by default, so we have to register it in yaml.
def represent_ordereddict(dumper, data):
    value = []

    for item_key, item_value in data.items():
        node_key = dumper.represent_data(item_key)
        node_value = dumper.represent_data(item_value)

        value.append((node_key, node_value))

    return yaml.nodes.MappingNode(u'tag:yaml.org,2002:map', value)

yaml.add_representer(OrderedDict, represent_ordereddict)

YamlObject['CameraGeometry'] = OrderedDict()
YamlObject['CameraGeometry']['Header'] = content.columns.keys()
YamlObject['CameraGeometry']['Data'] = []

for pixel in content:
    Pixel = []
    for k,item in enumerate(content.columns.keys()):    
        Pixel.append(get_flat_type(pixel[item]))
    
    YamlObject['CameraGeometry']['Data'].append(Pixel) 

with open('test.yaml', 'w') as fout: 
    yaml.dump(YamlObject, fout)#, default_flow_style=True)
    
#exit(0)

'''

for k,item in enumerate(content.columns.keys()):
    # Convert from numpys -> float and append as an array
    YamlObject['CameraGeometry'][item] = [get_flat_type(val) for val in content.columns[item]]

with open('test.yaml', 'w') as fout: 
    yaml.dump(YamlObject, fout)#, default_flow_style=True)
'''

# Read back the file and plot the array

with open('test.yaml', 'r') as fin: 
    YamlObject_read = yaml.load(fin)#, default_flow_style=True)

# The inline np.array(Data) is needed here, otherwise it raises an error.
CameraGeometry = Table(\
        np.array(YamlObject['CameraGeometry']['Data']), \
        names = YamlObject['CameraGeometry']['Header'])

print(CameraGeometry)

import matplotlib.pyplot as plt
x = CameraGeometry['x[mm]']
y = CameraGeometry['y[mm]']
plt.scatter(x=x, y=y, alpha=0.5)
plt.show()





