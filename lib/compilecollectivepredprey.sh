#!/bin/bash

rm ErPredprey*.so
rm ErPredprey.cpp
rm -r build
python3 setupErCollectivePredprey.py build_ext --inplace
cp ErPredprey*.so ../bin
