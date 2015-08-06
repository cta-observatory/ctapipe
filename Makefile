# Makefile with some convenient quick ways to do common things

help:
	@echo ''
	@echo 'ctapipe available make targets:'
	@echo ''
	@echo '  help         Print this help message (the default)'
	@echo '  init         Set up shell to use and work on ctapipe'
	@echo '  clean        Remove temp files'
	@echo '  test         Run tests'
	@echo '  doc          Generate Sphinx docs'
	@echo '  docshow      Generate and display docs in browser'
	@echo '  analyze      Do a static code check and report errors'
	@echo ''
	@echo '  conda        Build a conda package for distribution'
	@echo '  doc-publish  Publish docs online'
	@echo ''

init:
	git submodule init
	git submodule update
	export CTAPIPE_EXTRA_DIR=${PWD}/ctapipe-extra

clean:
	rm -rf build docs/_build docs/api

test:
	CTAPIPE_EXTRA_DIR=${PWD}/ctapipe-extra python setup.py test -V

doc:
	python setup.py build_sphinx

docshow:
	python setup.py build_sphinx --open-docs-in-browser

doc-publish:
	ghp-import -n -p -m 'Update gh-pages docs' docs/_build/html

conda:
	python setup.py bdist_conda


analyze:
	@pyflakes ctapipe examples

# any other command can be passed to setup.py
%:
	python setup.py $@
