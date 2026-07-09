#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=================================================="
echo "Compiling the Industrial Inspection Application..."
echo "=================================================="
make clean
make

echo -e "\n=================================================="
echo "Running Batch Industrial Inspection..."
echo "=================================================="
./defect_inspector --input data/input_textures --output data/output_textures

echo -e "\n=================================================="
echo "Execution Verification Complete."
echo "=================================================="