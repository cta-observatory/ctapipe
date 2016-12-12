# Makefile with some convenient quick ways to do common things

PROJECT=ctapipe
PYTHON=CTAPIPE_EXTRA_DIR=${PWD}/ctapipe-extra python

help:
	@echo ''
	@echo '$(PROJECT) available make targets:'
	@echo ''
	@echo '  help         Print this help message (the default)'
	@echo '  init         Set up shell to use and work on ctapipe'
	@echo '  develop      make symlinks to this package in python install dir'
	@echo '  clean        Remove temp files'
	@echo '  test         Run tests'
	@echo '  doc          Generate Sphinx docs'
	@echo '  doc-show     Generate and display docs in browser'
	@echo '  analyze      Do a static code check and report errors'
	@echo ''
	@echo 'Advanced targets (for experts)'
	@echo '  conda        Build a conda package for distribution'
	@echo '  doc-publish  Upload docs to static GitHub page'
	@echo ''

init:
	git submodule init
	git submodule update

clean:
	$(RM) -rf build docs/_build docs/api htmlcov
	find . -name "*.pyc" -exec rm {} \;
	find . -name "*.so" -exec rm {} \;
	find . -name __pycache__ | xargs rm -fr

test:
	$(PYTHON) setup.py test -V $<

doc:
	$(PYTHON) setup.py build_sphinx -w

doc-show:
	$(PYTHON) setup.py build_sphinx -w --open-docs-in-browser

doc-publish:
	ghp-import -n -p -m 'Update gh-pages docs' docs/_build/html

conda:
	python setup.py bdist_conda

analyze:
	@pylint ctapipe --ignored-classes=astropy.units

pep8:
	@pep8 --statistics


trailing-spaces:
	find $(PROJECT) examples docs -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

# any other command can be passed to setup.py
%:
	$(PYTHON) setup.py $@

