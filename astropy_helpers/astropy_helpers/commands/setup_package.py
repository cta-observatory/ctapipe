from os.path import join

def get_package_data():
    return {'astropy_helpers.commands': [join('src', 'compiler.c')]}
