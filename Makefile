# Makefile with some convenient quick ways to do common things

PROJECT=ctapipe
PYTHON=python

help:
	@echo ''
	@echo '$(PROJECT) available make targets:'
	@echo ''
	@echo '  help         Print this help message (the default)'
	@echo '  env          Create a conda environment for ctapipe development'
	@echo '  develop      make symlinks to this package in python install dir'
	@echo '  clean        Remove temp files'
	@echo '  test         Run tests'
	@echo '  doc          Generate Sphinx docs'
	@echo '  analyze      Do a static code check and report errors'
	@echo ''
	@echo 'Advanced targets (for experts)'
	@echo '  conda        Build a conda package for distribution'
	@echo '  doc-publish  Upload docs to static GitHub page'
	@echo ''

init:
	@echo "'make init' is no longer needed"

clean:
	$(RM) -rf build docs/_build docs/api htmlcov
	find . -name "*.pyc" -exec rm {} \;
	find . -name "*.so" -exec rm {} \;
	find . -name __pycache__ | xargs rm -fr

test:
	$(PYTHON) setup.py test

doc:
	cd docs && $(MAKE) html
	@echo "------------------------------------------------"
	@echo "Documentation is in: docs/_build/html/index.html"

doc-publish:
	ghp-import -n -p -m 'Update gh-pages docs' docs/_build/html

conda:
	python setup.py bdist_conda

analyze:
	@pylint ctapipe --ignored-classes=astropy.units

pep8:
	@pep8 --statistics

env:
	conda env create -n cta-dev -f environment.yml
	source activate cta-dev

trailing-spaces:
	find $(PROJECT) examples docs -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

# any other command can be passed to setup.py
%:
	$(PYTHON) setup.py $@

