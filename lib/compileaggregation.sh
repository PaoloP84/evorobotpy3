#!/bin/bash

rm ErAggregation*.so
rm ErAggregation.cpp
rm -r build
python3 setupErAggregation.py build_ext --inplace
cp ErAggregation*.so ../bin
