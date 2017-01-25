import pkg_resources


__all__ = ['resource_filename', 'gamma_test_file']


def _resource_filename(filename):
    """Return the full pathname of a particular resource"""
    return pkg_resources.resource_filename(__name__, filename)

def get(resource_name):
    """ get the filename for a resource """
    return _resource_filename(resource_name)

# some helper attributes

gamma_test_file = get('gamma_test.simtel.gz')

# a larger test file, from prod3. original name was
# gamma_20deg_0deg_run7514___cta-prod3_desert-2150m-Paranal-HB9-FA_cone10.simtel.gz
test_events_file = get('gamma_test_large.simtel.gz')


