import pkg_resources


__all__ = ['resource_filename', 'gamma_test_file']


def resource_filename(filename):
    return pkg_resources.resource_filename(__name__, filename)


gamma_test_file = resource_filename('gamma_test.simtel.gz')
