#!/bin/bash
# Assuming the appropriate environment is loaded

cd workflows/bnre
rm -rf .workflows

echo ' > Verifying Gravitational Waves pipeline.'
python pipeline.py --problem gw

echo ' > Verifying Lotka Volterra pipeline.'
python pipeline.py --problem lotka_volterra

echo ' > Verifying MG1 pipeline'
python pipeline.py --problem mg1

echo ' > Verifying SLCP pipeline'
python pipeline.py --problem slcp

echo ' > Verifying Spatial SIR pipeline'
python pipeline.py --problem spatialsir

echo ' > Verifying Weinberg pipeline'
python pipeline.py --problem weinberg

cd ../..
