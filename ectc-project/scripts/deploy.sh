#!/bin/bash

# ECTC Deployment Automation Script
# Deploys ECTC system to testbed

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
NUM_NODES=5
GATEWAY_TYPE="raspberry_pi"
SENSOR_TAG_TYPE="CC2650STK"
ENERGY_SOURCE="solar"

echo -e "${GREEN}=== ECTC Testbed Deployment ===${NC}"
echo "Configuration:"
echo "  Nodes: ${NUM_NODES}"
echo "  Gateway: ${GATEWAY_TYPE}"
echo "  MCU: ${SENSOR_TAG_TYPE}"
echo "  Energy: ${ENERGY_SOURCE}"

# Check prerequisites
echo -e "\n${YELLOW}Checking prerequisites...${NC}"

command -v python3 >/dev/null 2>&1 || { echo -e "${RED}Error: Python3 not found${NC}"; exit 1; }
command -v git >/dev/null 2>&1 || { echo -e "${RED}Error: Git not found${NC}"; exit 1; }

if [ ! -f "firmware/build/cc2650/ectc_node.bin" ]; then
    echo -e "${YELLOW}Warning: Firmware not built. Building now...${NC}"
    ./firmware/build.sh
fi

# Deploy to gateway
echo -e "\n${YELLOW}Deploying gateway...${NC}"

if [ "$GATEWAY_TYPE" = "raspberry_pi" ]; then
    deploy_gateway_raspberry_pi
elif [ "$GATEWAY_TYPE" = "xpc240400b" ]; then
    deploy_gateway_xpc()
else
    echo -e "${RED}Error: Unknown gateway type${NC}"
    exit 1
fi

# Deploy to nodes
echo -e "\n${YELLOW}Deploying to ${NUM_NODES} nodes...${NC}"
deploy_nodes

# Configure energy harvesting
echo -e "\n${YELLOW}Configuring energy harvesting...${NC}"
configure_energy_harvesting

# Start system
echo -e "\n${YELLOW}Starting ECTC system...${NC}"
start_ectc_system

echo -e "\n${GREEN}=== Deployment Complete ===${NC}"
echo "Gateway: running on http://localhost:8080"
echo "Monitoring: http://localhost:8080/metrics"
echo "To stop: ./scripts/stop_ectc.sh"

exit 0

deploy_gateway_raspberry_pi() {
    echo "  Creating Python virtual environment..."
    python3 -m venv ectc_env
    source ectc_env/bin/activate

    echo "  Installing dependencies..."
    pip install -r gateway/requirements.txt

    echo "  Configuring gateway services..."
    cat > gateway/config.yaml << EOF
gateway:
  port: 8080
  log_level: INFO

network:
  num_nodes: ${NUM_NODES}
  radio_channel: 11
  pan_id: 0x1234

energy_source:
  type: ${ENERGY_SOURCE}
  mppt_voltage: 2.4

monitoring:
  enable_prometheus: true
  prometheus_port: 9090
EOF

    echo "  Starting gateway service..."
    cd gateway
    nohup python -m ectc_gateway.main --config config.yaml > ../logs/gateway.log 2>&1 &
    cd ..
}

deploy_nodes() {
    echo "  Building firmware for all nodes..."
    for i in $(seq 1 $NUM_NODES); do
        node_id=$((i - 1))
        echo "    Node ${i} (ID: ${node_id})..."

        # Create node-specific build
        NODE_BUILD_DIR="firmware/build/node_${node_id}"
        mkdir -p ${NODE_BUILD_DIR}
        cp -r firmware/ectc_node/* ${NODE_BUILD_DIR}/

        # Add node ID to build
        sed -i "s/define NODE_ID_DEFAULT.*/define NODE_ID_DEFAULT ${node_id}/" \
            ${NODE_BUILD_DIR}/include/ectc_main.h

        # Build
        cd ${NODE_BUILD_DIR}
        ../../../firmware/build.sh
        cd -

        # Flash (if hardware is connected)
        if command -v jlink >/dev/null 2>&1; then
            echo "      Flashing..."
            # JLink -Commander -Device CC2650F128 -if SWD -speed 4000 -commandfile flash_${node_id}.cmd
        else
            echo "      (J-Link not found, skipping flash. Use TI Uniflash.)"
        fi
    done
}

configure_energy_harvesting() {
    echo "  Configuring BQ25570 settings..."

    # Default BQ25570 configuration
    cat > config/bq25570_config.txt << EOF
BQ25570 Configuration
=====================
MPPT Voltage: 2.4V (solar optimized)
VBAT Under-voltage: 2.3V
VBAT OK threshold: 2.5V
VSTOR OK threshold: 3.3V
VSTOR Maximum: 4.5V
Current Limit: 100mA
VOUT Target: 3.3V

Installation Notes:
1. Connect solar panel to VIN_DC+/-
2. Connect 100μF capacitor to VSTOR/GND
3. Connect VSTOR to CC2650 VDD pin
4. Connect capacitor voltage monitor to ADC pin
EOF

    echo "  Creating capacitor installation guide..."
    cat > docs/hardware_capacitor.md << EOF
# Capacitor Installation Guide

## Required Components
- AVX FFB45J108K (100μF, 6.3V, 0805) or
- Murata GRM155R71C104KA88D (100μF, 6.3V, 0402)

## Installation Steps
1. Remove original 1μF capacitor from CC2650STK
2. Solder 100μF capacitor between VSTOR and GND
   - Polarity: Positive to VSTOR, Negative to GND
3. Verify with multimeter (should read ~3.3V when charged)

## Safety Notes
- Do not exceed 6.3V rating
- Ensure proper polarity
- Use ESD-safe handling
EOF
}

start_ectc_system() {
    # Create logs directory
    mkdir -p logs

    # Start monitoring (optional)
    if command -v prometheus >/dev/null 2>&1; then
        echo "  Starting Prometheus..."
        prometheus --config.file=monitoring/prometheus.yml &
    fi

    # Create status file
    cat > ectc_status.json << EOF
{
    "status": "running",
    "num_nodes": ${NUM_NODES},
    "gateway": "http://localhost:8080",
    "start_time": "$(date -Iseconds)",
    "energy_source": "${ENERGY_SOURCE}"
}
EOF

    echo "  ECTC system is running!"
    echo ""
    echo "  Useful commands:"
    echo "    - View gateway logs: tail -f logs/gateway.log"
    echo "    - Check node status: curl http://localhost:8080/status"
    echo "    - Stop system: ./scripts/stop_ectc.sh"
}
