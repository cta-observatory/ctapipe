Contributing to astropy-helpers
===============================

The guidelines for contributing to ``astropy-helpers`` are generally the same
as the [contributing guidelines for the astropy core
package](http://github.com/astropy/astropy/blob/master/CONTRIBUTING.md).
Basically, report relevant issues in the ``astropy-helpers`` issue tracker, and
we welcome pull requests that broadly follow the [Astropy coding
guidelines](http://docs.astropy.org/en/latest/development/codeguide.html).

The key subtlety lies in understanding the relationship between ``astropy`` and
``astropy-helpers``.  This package contains the build, installation, and
documentation tools used by astropy.  It also includes support for the
``setup.py test`` command, though Astropy is still required for this to
function (it does not currently include the full Astropy test runner).  So
issues or improvements to that functionality should be addressed in this
package. Any other aspect of the [astropy core
package](http://github.com/astropy/astropy) (or any other package that uses
``astropy-helpers``) should be addressed in the github repository for that
package.
