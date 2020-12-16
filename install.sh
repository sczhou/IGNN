#!/bin/bash
cd ./models/lib
rm -Rf build dist *.egg-info
# python setup.py clean
python setup.py install --user