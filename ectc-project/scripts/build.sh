#!/bin/bash

# ECTC Testbed Build Script
# Builds all components of ECTC system

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== ECTC Full Build ===${NC}"

# Build MCU firmware
echo -e "\n${YELLOW}Building MCU firmware...${NC}"
cd firmware
./build.sh
cd ..

# Build Python gateway
echo -e "\n${YELLOW}Building Python gateway...${NC}"
cd gateway
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
cd ..

# Build simulation
echo -e "\n${YELLOW}Building simulation...${NC}"
python3 -m pip install -r simulation/requirements.txt

# Run unit tests
echo -e "\n${YELLOW}Running unit tests...${NC}"
python3 -m pytest tests/ -v

# Build documentation
echo -e "\n${YELLOW}Building documentation...${NC}"
if command -v sphinx-build >/dev/null 2>&1; then
    cd docs
    make html
    cd ..
fi

echo -e "\n${GREEN}Build complete!${NC}"
