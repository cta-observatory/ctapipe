doc:
	python setup.py build_sphinx

# any other thing typed after make is just passed as an option to setup.py
%:
	python setup.py $@
