#namedtuple is needed for find_mod_objs so it can have a non-local module

import sys
from collections import namedtuple

import pytest

from ..utils import find_mod_objs

PY3 = sys.version_info[0] >= 3
pytestmark = pytest.mark.skipif("PY3")


def test_find_mod_objs():
    lnms, fqns, objs = find_mod_objs('astropy_helpers')

    # this import  is after the above call intentionally to make sure
    # find_mod_objs properly imports astropy on its own
    import astropy_helpers

    # just check for astropy.test ... other things might be added, so we
    # shouldn't check that it's the only thing
    assert lnms == []

    lnms, fqns, objs = find_mod_objs(
        'astropy_helpers.sphinx.ext.tests.test_utils', onlylocals=False)

    assert namedtuple in objs

    lnms, fqns, objs = find_mod_objs(
        'astropy_helpers.sphinx.ext.tests.test_utils', onlylocals=True)
    assert 'namedtuple' not in lnms
    assert 'collections.namedtuple' not in fqns
    assert namedtuple not in objs
