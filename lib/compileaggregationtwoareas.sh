#!/bin/bash

rm ErAggregationTwoAreas*.so
rm ErAggregationTwoAreas.cpp
rm -r build
python3 setupErAggregationTwoAreas.py build_ext --inplace
cp ErAggregationTwoAreas*.so ../bin
