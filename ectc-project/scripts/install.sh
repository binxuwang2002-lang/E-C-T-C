#!/bin/bash

# ECTC Installation Script
# Automated setup for ECTC development environment

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== ECTC Installation ===${NC}"

# Check OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo -e "${RED}Error: Unsupported OS${NC}"
    exit 1
fi

echo "Detected OS: ${OS}"

# Install system dependencies
echo -e "\n${YELLOW}Installing system dependencies...${NC}"

if [ "$OS" = "linux" ]; then
    sudo apt-get update
    sudo apt-get install -y build-essential git wget curl \
        python3 python3-pip python3-venv \
        cmake doxygen graphviz
elif [ "$OS" = "macos" ]; then
    brew install git wget python3 cmake doxygen graphviz
fi

# Install ARM GCC
echo -e "\n${YELLOW}Installing ARM GCC toolchain...${NC}"

if ! command -v arm-none-eabi-gcc &> /dev/null; then
    cd /tmp
    wget -q https://developer.arm.com/-/media/Files/downloads/gnu-rm/10.3-2021.10/gcc-arm-none-eabi-10.3-2021.10-x86_64-${OS}.tar.bz2
    sudo tar -xjf gcc-arm-none-eabi-10.3-2021.10-x86_64-${OS}.tar.bz2 -C /opt/
    echo 'export PATH=/opt/gcc-arm-none-eabi-10.3-2021.10/bin:$PATH' >> ~/.bashrc
    echo "ARM GCC installed to /opt/gcc-arm-none-eabi-10.3-2021.10"
    cd -
else
    echo "ARM GCC already installed"
fi

# Install TI SDK (optional)
echo -e "\n${YELLOW}Installing TI CC2650 SDK (optional)...${NC}"
read -p "Download TI SDK? Requires manual registration (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd /tmp
    wget -q --no-check-certificate https://www.ti.com/tool/download/SIMPLELINK-CC13X2-26X2-SDK
    echo "Download TI SDK from: https://www.ti.com/tool/SIMPLELINK-CC13X2-26X2-SDK"
    cd -
fi

# Create Python virtual environment
echo -e "\n${YELLOW}Creating Python virtual environment...${NC}"
python3 -m venv ectc_env
source ectc_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
echo -e "\n${YELLOW}Installing Python dependencies...${NC}"
pip install -r gateway/requirements.txt

# Install development tools
echo -e "\n${YELLOW}Installing development tools...${NC}"
pip install pytest pytest-cov black flake8 sphinx sphinx-rtd-theme

# Clone/download models (simulated)
echo -e "\n${YELLOW}Preparing model files...${NC}"
mkdir -p gateway/ectc_gateway/models
mkdir -p simulation/rtl_power_models
mkdir -p simulation/spice_netlists

# Create configuration
echo -e "\n${YELLOW}Creating configuration...${NC}"
mkdir -p config
cat > config/ectc_config.yaml << EOF
# ECTC Configuration
project_name: "ECTC Battery-Free Sensor Network"
version: "1.0"

# Hardware Configuration
hardware:
  mcu: "CC2650"
  sensor_tags: 50
  gateway: "raspberry_pi"

# Energy Configuration
energy:
  source: "solar"
  capacitor: 100e-6  # 100 μF
  vdd: 3.3
  mppt_voltage: 2.4

# Network Configuration
network:
  protocol: "IEEE_802_15_4"
  channel: 11
  pan_id: 0x1234
  num_nodes: 50
  num_mobile: 10

# Algorithm Parameters
algorithms:
  lyapunov_v: 50.0
  lyapunov_beta: 0.1
  shapley_epsilon: 0.1
  shapley_delta: 0.05
  lstm_hidden_dim: 32
  lstm_seq_len: 10

# Evaluation
evaluation:
  duration_hours: 100
  num_trials: 5
  energy_source: "solar"
EOF

echo -e "\n${GREEN}Installation complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source ectc_env/bin/activate"
echo "  2. Build project: ./scripts/build.sh"
echo "  3. Run tests: python3 -m pytest tests/"
echo "  4. Deploy: ./scripts/deploy.sh"
echo ""
echo "Documentation: docs/build/html/index.html"
echo ""
