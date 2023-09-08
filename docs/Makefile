# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = -W --keep-going -n --color
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = ctapipe
SOURCEDIR     = .
BUILDDIR      = _build


# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf api
	rm -rf user-guide/examples/*.zip
	rm -rf user-guide/examples/*.rst
	rm -rf user-guide/examples/*/*.ipynb
	rm -rf user-guide/examples/*/*.py.md5
	rm -rf user-guide/examples/*/*.pickle
	rm -rf user-guide/examples/*/*.rst
	rm -rf user-guide/examples/*/images
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


.PHONY: all help clean Makefile
