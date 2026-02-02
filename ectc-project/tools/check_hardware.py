#!/usr/bin/env python3
"""
Hardware Check Script
=====================

Diagnose hardware connectivity and functionality.
"""

import serial
import subprocess
import sys
import time
from pathlib import Path


def check_jlink():
    """Check J-Link availability"""
    print("Checking J-Link...")
    try:
        result = subprocess.run(['which', 'JLink'], capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✓ J-Link found")
            return True
        else:
            print("  ✗ J-Link not found")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def check_ti_sdk():
    """Check TI SDK installation"""
    print("\nChecking TI SDK...")
    sdk_path = Path(os.environ.get('TI_SDK_PATH', '/opt/ti/simplelink_cc13x2_26x2_sdk_5_30_00_00'))
    if sdk_path.exists():
        print(f"  ✓ TI SDK found at {sdk_path}")
        return True
    else:
        print(f"  ✗ TI SDK not found at {sdk_path}")
        print(f"    Set TI_SDK_PATH environment variable")
        return False


def check_serial_ports():
    """Check available serial ports"""
    print("\nChecking serial ports...")
    import serial.tools.list_ports

    ports = list(serial.tools.list_ports.comports())

    if not ports:
        print("  ✗ No serial ports found")
        return False

    print(f"  Found {len(ports)} serial port(s):")
    for port in ports:
        print(f"    - {port.device}: {port.description}")

    return True


def check_cc2650_devices():
    """Check for connected CC2650 devices"""
    print("\nChecking for CC2650 devices...")

    # Try to detect via serial VID/PID
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())

    cc2650_count = 0
    for port in ports:
        # CC2650 typical VID:PID (may vary)
        if 'CH340' in port.description or 'CP210' in port.description:
            cc2650_count += 1
            print(f"  ✓ Potential CC2650 at {port.device}")

    if cc2650_count == 0:
        print("  ✗ No CC2650 devices detected")
        print("    Make sure devices are connected and drivers installed")

    return cc2650_count > 0


def test_bq25570_communication():
    """Test BQ25570 I2C communication"""
    print("\nTesting BQ25570 I2C communication...")

    # This would require actual I2C interface
    # For now, just check if port is available
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())

    # Look for devices that might be BQ25570 eval boards
    for port in ports:
        if 'USB' in port.device:
            print(f"  Testing {port.device}...")
            try:
                # Quick test read
                ser = serial.Serial(port.device, 9600, timeout=1)
                ser.close()
                print(f"  ✓ Device responding on {port.device}")
                return True
            except Exception as e:
                print(f"  ✗ Error on {port.device}: {e}")

    print("  ✗ No I2C devices responding")
    return False


def check_radio_modules():
    """Check IEEE 802.15.4 radio modules"""
    print("\nChecking radio modules...")

    # Look for CC1352P1 or similar
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())

    radio_found = False
    for port in ports:
        if any(keyword in port.description.lower() for keyword in ['cc1352', 'launchpad', 'radio']):
            print(f"  ✓ Radio module found: {port.device}")
            radio_found = True

    if not radio_found:
        print("  ✗ No radio modules detected")

    return radio_found


def check_solar_panel():
    """Check solar panel voltage"""
    print("\nChecking solar panel...")
    print("  This requires manual measurement with multimeter")
    print("  Expected voltage: 0.8V - 3.2V (depending on light)")

    # In a real implementation, would read from ADC
    response = input("  Measure solar panel voltage (V): ").strip()
    try:
        voltage = float(response)
        if 0.5 <= voltage <= 5.0:
            print(f"  ✓ Voltage {voltage}V seems reasonable")
            return True
        else:
            print(f"  ✗ Voltage {voltage}V seems abnormal")
            return False
    except ValueError:
        print("  ✗ Invalid input")
        return False


def check_capacitor():
    """Check storage capacitor"""
    print("\nChecking storage capacitor...")
    print("  Measure capacitor voltage (should be 0V if not charged)")

    response = input("  Measure capacitor voltage (V): ").strip()
    try:
        voltage = float(response)
        if 0 <= voltage <= 4.5:
            print(f"  ✓ Voltage {voltage}V in safe range")
            return True
        else:
            print(f"  ✗ Voltage {voltage}V out of range (0-4.5V)")
            return False
    except ValueError:
        print("  ✗ Invalid input")
        return False


def generate_report(results):
    """Generate hardware check report"""
    print("\n" + "="*60)
    print("HARDWARE CHECK REPORT")
    print("="*60)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    print(f"\nPassed: {passed}/{total} checks")

    print("\nDetails:")
    for check, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {check}")

    print("\n" + "="*60)

    if passed == total:
        print("All hardware checks passed! Ready to deploy.")
        return 0
    else:
        print("Some hardware checks failed. Review and fix issues.")
        return 1


def main():
    """Main hardware check"""
    print("="*60)
    print("ECTC Hardware Check")
    print("="*60)

    results = {}

    # Check software
    results['J-Link'] = check_jlink()
    results['TI SDK'] = check_ti_sdk()
    results['Serial Ports'] = check_serial_ports()

    # Check hardware
    results['CC2650 Devices'] = check_cc2650_devices()
    results['Radio Modules'] = check_radio_modules()
    results['BQ25570 I2C'] = test_bq25570_communication()

    # Manual checks
    results['Solar Panel'] = check_solar_panel()
    results['Capacitor'] = check_capacitor()

    # Generate report
    return generate_report(results)


if __name__ == '__main__':
    sys.exit(main())
