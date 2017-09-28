import pkg_resources
from .VERSION import __version__


__all__ = ['resource_filename', 'gamma_test_file']


def get(resource_name):
    """ get the filename for a resource """
    if not pkg_resources.resource_exists(__name__, resource_name):
        raise FileNotFoundError("Couldn't find resource: '{}'"
                                .format(resource_name))
    return pkg_resources.resource_filename(__name__, resource_name)

# some helper attributes

gamma_test_file = get('gamma_test.simtel.gz')

# a larger test file, from prod3. original name was
# gamma_20deg_0deg_run7514___cta-prod3_desert-2150m-Paranal-HB9-FA_cone10.simtel.gz
test_events_file = get('gamma_test_large.simtel.gz')


