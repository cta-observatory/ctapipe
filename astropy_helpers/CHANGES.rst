astropy-helpers Changelog
=========================

1.0.2 (2015-04-02)
------------------

- Various fixes enabling the astropy-helpers Sphinx build command and
  Sphinx extensions to work with Sphinx 1.3. [#148]

- More improvement to the ability to handle multiple versions of
  astropy-helpers being imported in the same Python interpreter session
  in the (somewhat rare) case of nested installs. [#147]

- To better support high resolution displays, use SVG for the astropy
  logo and linkout image, falling back to PNGs for browsers that
  support it. [#150, #151]

- Improve ``setup_helpers.get_compiler_version`` to work with more compilers,
  and to return more info.  This will help fix builds of Astropy on less
  common compilers, like Sun C. [#153]


1.0.1 (2015-03-04)
------------------

- Released in concert with v0.4.8 to address the same issues.

- Improved the ``ah_bootstrap`` script's ability to override existing
  installations of astropy-helpers with new versions in the context of
  installing multiple packages simultaneously within the same Python
  interpreter (e.g. when one package has in its ``setup_requires`` another
  package that uses a different version of astropy-helpers. [#144]

- Added a workaround to an issue in matplotlib that can, in rare cases, lead
  to a crash when installing packages that import matplotlib at build time.
  [#144]


0.4.8 (2015-03-04)
------------------

- Improved the ``ah_bootstrap`` script's ability to override existing
  installations of astropy-helpers with new versions in the context of
  installing multiple packages simultaneously within the same Python
  interpreter (e.g. when one package has in its ``setup_requires`` another
  package that uses a different version of astropy-helpers. [#144]

- Added a workaround to an issue in matplotlib that can, in rare cases, lead
  to a crash when installing packages that import matplotlib at build time.
  [#144]


1.0 (2015-02-17)
----------------

- Added new pre-/post-command hook points for ``setup.py`` commands.  Now any
  package can define code to run before and/or after any ``setup.py`` command
  without having to manually subclass that command by adding
  ``pre_<command_name>_hook`` and ``post_<command_name>_hook`` callables to
  the package's ``setup_package.py`` module.  See the PR for more details.
  [#112]

- The following objects in the ``astropy_helpers.setup_helpers`` module have
  been relocated:

  - ``get_dummy_distribution``, ``get_distutils_*``, ``get_compiler_option``,
    ``add_command_option``, ``is_distutils_display_option`` ->
    ``astropy_helpers.distutils_helpers``

  - ``should_build_with_cython``, ``generate_build_ext_command`` ->
    ``astropy_helpers.commands.build_ext``

  - ``AstropyBuildPy`` -> ``astropy_helpers.commands.build_py``

  - ``AstropyBuildSphinx`` -> ``astropy_helpers.commands.build_sphinx``

  - ``AstropyInstall`` -> ``astropy_helpers.commands.install``

  - ``AstropyInstallLib`` -> ``astropy_helpers.commands.install_lib``

  - ``AstropyRegister`` -> ``astropy_helpers.commands.register``

  - ``get_pkg_version_module`` -> ``astropy_helpers.version_helpers``

  - ``write_if_different``, ``import_file``, ``get_numpy_include_path`` ->
    ``astropy_helpers.utils``

  All of these are "soft" deprecations in the sense that they are still
  importable from ``astropy_helpers.setup_helpers`` for now, and there is
  no (easy) way to produce deprecation warnings when importing these objects
  from ``setup_helpers`` rather than directly from the modules they are
  defined in.  But please consider updating any imports to these objects.
  [#110]

- Use of the ``astropy.sphinx.ext.astropyautosummary`` extension is deprecated
  for use with Sphinx < 1.2.  Instead it should suffice to remove this
  extension for the ``extensions`` list in your ``conf.py`` and add the stock
  ``sphinx.ext.autosummary`` instead. [#131]


0.4.7 (2015-02-17)
------------------

- Fixed incorrect/missing git hash being added to the generated ``version.py``
  when creating a release. [#141]


0.4.6 (2015-02-16)
------------------

- Fixed problems related to the automatically generated _compiler
  module not being created properly. [#139]


0.4.5 (2015-02-11)
------------------

- Fixed an issue where ah_bootstrap.py could blow up when astropy_helper's
  version number is 1.0.

- Added a workaround for documentation of properties in the rare case
  where the class's metaclass has a property of the same name. [#130]

- Fixed an issue on Python 3 where importing a package using astropy-helper's
  generated version.py module would crash when the current working directory
  is an empty git repository. [#114]

- Fixed an issue where the "revision count" appended to .dev versions by
  the generated version.py did not accurately reflect the revision count for
  the package it belongs to, and could be invalid if the current working
  directory is an unrelated git repository. [#107]

- Likewise, fixed a confusing warning message that could occur in the same
  circumstances as the above issue. [#121]


0.4.4 (2014-12-31)
------------------

- More improvements for building the documentation using Python 3.x. [#100]

- Additional minor fixes to Python 3 support. [#115]

- Updates to support new test features in Astropy [#92, #106]


0.4.3 (2014-10-22)
------------------

- The generated ``version.py`` file now preserves the git hash of installed
  copies of the package as well as when building a source distribution.  That
  is, the git hash of the changeset that was installed/released is preserved.
  [#87]

- In smart resolver add resolution for class links when they exist in the
  intersphinx inventory, but not the mapping of the current package
  (e.g. when an affiliated package uses an astropy core class of which
  "actual" and "documented" location differs) [#88]

- Fixed a bug that could occur when running ``setup.py`` for the first time
  in a repository that uses astropy-helpers as a submodule:
  ``AttributeError: 'NoneType' object has no attribute 'mkdtemp'`` [#89]

- Fixed a bug where optional arguments to the ``doctest-skip`` Sphinx
  directive were sometimes being left in the generated documentation output.
  [#90]

- Improved support for building the documentation using Python 3.x. [#96]

- Avoid error message if .git directory is not present. [#91]


0.4.2 (2014-08-09)
------------------

- Fixed some CSS issues in generated API docs. [#69]

- Fixed the warning message that could be displayed when generating a
  version number with some older versions of git. [#77]

- Fixed automodsumm to work with new versions of Sphinx (>= 1.2.2). [#80]


0.4.1 (2014-08-08)
------------------

- Fixed git revision count on systems with git versions older than v1.7.2.
  [#70]

- Fixed display of warning text when running a git command fails (previously
  the output of stderr was not being decoded properly). [#70]

- The ``--offline`` flag to ``setup.py`` understood by ``ah_bootstrap.py``
  now also prevents git from going online to fetch submodule updates. [#67]

- The Sphinx extension for converting issue numbers to links in the changelog
  now supports working on arbitrary pages via a new ``conf.py`` setting:
  ``changelog_links_docpattern``.  By default it affects the ``changelog``
  and ``whatsnew`` pages in one's Sphinx docs. [#61]

- Fixed crash that could result from users with missing/misconfigured
  locale settings. [#58]

- The font used for code examples in the docs is now the
  system-defined ``monospace`` font, rather than ``Minaco``, which is
  not available on all platforms. [#50]


0.4 (2014-07-15)
----------------

- Initial release of astropy-helpers.  See `APE4
  <https://github.com/astropy/astropy-APEs/blob/master/APE4.rst>`_ for
  details of the motivation and design of this package.

- The ``astropy_helpers`` package replaces the following modules in the
  ``astropy`` package:

  - ``astropy.setup_helpers`` -> ``astropy_helpers.setup_helpers``

  - ``astropy.version_helpers`` -> ``astropy_helpers.version_helpers``

  - ``astropy.sphinx`` - > ``astropy_helpers.sphinx``

  These modules should be considered deprecated in ``astropy``, and any new,
  non-critical changes to those modules will be made in ``astropy_helpers``
  instead.  Affiliated packages wishing to make use those modules (as in the
  Astropy package-template) should use the versions from ``astropy_helpers``
  instead, and include the ``ah_bootstrap.py`` script in their project, for
  bootstrapping the ``astropy_helpers`` package in their setup.py script.
