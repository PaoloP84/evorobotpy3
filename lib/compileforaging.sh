#!/bin/bash

rm ErCforaging*.so
rm ErCforaging.cpp
rm -r build
python3 setupErForaging.py build_ext --inplace
cp ErCforaging*.so ../bin
