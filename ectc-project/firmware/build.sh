#!/bin/bash

# ECTC Firmware Build Script
# Builds MCU firmware for CC2650

set -e

# Configuration
PROJECT_NAME="ectc_node"
MCU="cc2650"
TOOLCHAIN_PREFIX="arm-none-eabi-"
SDK_PATH="${TI_SDK_PATH:-/opt/ti/simplelink_cc13x2_26x2_sdk_5_30_00_00}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== ECTC Firmware Build ===${NC}"
echo "MCU: ${MCU}"
echo "SDK: ${SDK_PATH}"

# Check dependencies
echo -e "\n${YELLOW}Checking dependencies...${NC}"

if ! command -v ${TOOLCHAIN_PREFIX}gcc &> /dev/null; then
    echo -e "${RED}Error: ARM GCC not found. Install with:${NC}"
    echo "  wget https://developer.arm.com/-/media/Files/downloads/gnu-rm/10.3-2021.10/gcc-arm-none-eabi-10.3-2021.10-x86_64-linux.tar.bz2"
    exit 1
fi

# Create build directory
BUILD_DIR="build/${MCU}"
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

echo -e "\n${YELLOW}Cleaning previous build...${NC}"
rm -rf *.o *.axf *.bin *.map *.lst *.sym *.map

# Compiler flags
CFLAGS="-mcpu=cortex-m3"
CFLAGS="${CFLAGS} -mthumb -mfloat-abi=soft"
CFLAGS="${CFLAGS} -O2 -ffunction-sections -fdata-sections"
CFLAGS="${CFLAGS} -g3 -gdwarf-3"
CFLAGS="${CFLAGS} -Wall -Wextra -Wpedantic"
CFLAGS="${CFLAGS} -std=c99 -D_BSD_SOURCE"
CFLAGS="${CFLAGS} -DTI_DRIVERS_WANT_CONFIG"

# Include paths
INCLUDE="-I../../include"
INCLUDE="${INCLUDE} -I../../trace"
INCLUDE="${INCLUDE} -I${SDK_PATH}/source"
INCLUDE="${INCLUDE} -I${SDK_PATH}/source/ti/drivers"
INCLUDE="${INCLUDE} -I${SDK_PATH}/source/ti/drivers/PWM"
INCLUDE="${INCLUDE} -I${SDK_PATH}/source/ti/drivers/GPIO"
INCLUDE="${INCLUDE} -I${SDK_PATH}/source/ti/drivers/I2C"
INCLUDE="${INCLUDE} -I${SDK_PATH}/source/ti/drivers/UART"
INCLUDE="${INCLUDE} -I${SDK_PATH}/source/ti/drivers/AES"
INCLUDE="${INCLUDE} -I${SDK_PATH}/source/ti/devices/cc26x0"
INCLUDE="${INCLUDE} -I${SDK_PATH}/source/ti/devices/cc26x0/driverlib"

# Source files
SOURCES="../../Core/ectc_main.c"
SOURCES="${SOURCES} ../../ML/tiny_lstm.c"
SOURCES="${SOURCES} ../../Core/shapley_local.c"
SOURCES="${SOURCES} ../../ZKP/pedersen.c"
SOURCES="${SOURCES} ../../trace/trace.c"
SOURCES="${SOURCES} ../../Drivers/bq25570.c"
SOURCES="${SOURCES} ../../Drivers/radio.c"

echo -e "\n${YELLOW}Compiling sources...${NC}"

# Compile each source file
for src in ${SOURCES}; do
    echo "  Compiling: $(basename ${src})"
    ${TOOLCHAIN_PREFIX}gcc ${CFLAGS} ${INCLUDE} -c ${src} -o $(basename ${src%.*}.o)
done

echo -e "\n${YELLOW}Linking...${NC}"

# Linker flags
LDFLAGS="-mcpu=cortex-m3"
LDFLAGS="${LDFLAGS} -mthumb -mfloat-abi=soft"
LDFLAGS="${LDFLAGS} -nostdlib"
LDFLAGS="${LDFLAGS} -T ../../linker.ld"
LDFLAGS="${LDFLAGS} -L${SDK_PATH}/source/ti/devices/cc26x0/driverlib/gcc"
LDFLAGS="${LDFLAGS} -Wl,--gc-sections"
LDFLAGS="${LDFLAGS} -Wl,-Map=${PROJECT_NAME}.map"
LDFLAGS="${LDFLAGS} -Wl,--entry=ResetISR"
LDFLAGS="${LDFLAGS} -Wl,--cref"

# Linker libraries
LIBS="-l:libdriverlib.a"

# Create object file list
OBJECTS=$(ls *.o)
${TOOLCHAIN_PREFIX}gcc ${LDFLAGS} ${OBJECTS} ${LIBS} -o ${PROJECT_NAME}.axf

echo -e "\n${YELLOW}Generating binary files...${NC}"

# Generate bin file
${TOOLCHAIN_PREFIX}objcopy -O binary ${PROJECT_NAME}.axf ${PROJECT_NAME}.bin

# Generate hex file
${TOOLCHAIN_PREFIX}objcopy -O ihex ${PROJECT_NAME}.axf ${PROJECT_NAME}.hex

# Generate listing
${TOOLCHAIN_PREFIX}objdump -h -S ${PROJECT_NAME}.axf > ${PROJECT_NAME}.lst

# Generate symbol table
${TOOLCHAIN_PREFIX}nm -n ${PROJECT_NAME}.axf > ${PROJECT_NAME}.sym

echo -e "\n${YELLOW}Build statistics...${NC}"

# Calculate sizes
SIZE_TEXT=$(${TOOLCHAIN_PREFIX}size ${PROJECT_NAME}.axf | tail -1 | awk '{print $1}')
SIZE_DATA=$(${TOOLCHAIN_PREFIX}size ${PROJECT_NAME}.axf | tail -1 | awk '{print $2}')
SIZE_BSS=$(${TOOLCHAIN_PREFIX}size ${PROJECT_NAME}.axf | tail -1 | awk '{print $3}')
SIZE_TOTAL=$((SIZE_TEXT + SIZE_DATA + SIZE_BSS))

echo "  Text (flash):     ${SIZE_TEXT} bytes"
echo "  Data (ram):       ${SIZE_DATA} bytes"
echo "  BSS (ram):        ${SIZE_BSS} bytes"
echo "  Total:            ${SIZE_TOTAL} bytes"
echo "  Limit:            40KB (40960 bytes)"

# Check against limits
FLASH_LIMIT=131072  # 128KB
RAM_LIMIT=40960     # 40KB

if [ ${SIZE_TEXT} -gt ${FLASH_LIMIT} ]; then
    echo -e "${RED}  ERROR: Flash size exceeds limit!${NC}"
    exit 1
fi

if [ $((SIZE_DATA + SIZE_BSS)) -gt ${RAM_LIMIT} ]; then
    echo -e "${RED}  WARNING: RAM usage exceeds limit!${NC}"
    echo -e "${YELLOW}  Consider optimizing memory usage${NC}"
fi

# Generate size report
echo -e "\n${YELLOW}Generating report...${NC}"
cat > build_report.txt << EOF
ECTC Firmware Build Report
==========================
Build Date: $(date)
Toolchain: ${TOOLCHAIN_PREFIX}gcc (ARM GCC)
MCU: ${MCU}

Memory Usage:
-------------
Flash (Text):    ${SIZE_TEXT} bytes / ${FLASH_LIMIT} bytes
RAM (Data+BSS):  $((SIZE_DATA + SIZE_BSS)) bytes / ${RAM_LIMIT} bytes

Object Files:
-------------
EOF

ls -1 *.o >> build_report.txt

echo -e "\n${GREEN}=== Build Complete ===${NC}"
echo -e "Output files in: ${BUILD_DIR}/"
echo "  - ${PROJECT_NAME}.bin   (Flash image)"
echo "  - ${PROJECT_NAME}.axf   (ELF with debug)"
echo "  - ${PROJECT_NAME}.hex   (Intel HEX)"
echo "  - ${PROJECT_NAME}.map   (Linker map)"
echo "  - build_report.txt      (Build report)"

echo -e "\n${YELLOW}Next steps:${NC}"
echo "  1. Flash to CC2650:"
echo "     ${TOOLCHAIN_PREFIX}gdb ${PROJECT_NAME}.axf"
echo "  2. Or use TI Uniflash GUI tool"
echo "  3. Test on hardware"

exit 0
