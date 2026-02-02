#!/usr/bin/env python3
"""
Diagnostic Tool for ECTC System
===============================

Comprehensive diagnostics for debugging and performance analysis.
"""

import requests
import time
import json
import sys
from typing import Dict, Any


class ETCDiagnostics:
    """ECTC diagnostic tool"""

    def __init__(self, gateway_url: str = "http://localhost:8080"):
        self.gateway_url = gateway_url
        self.session = requests.Session()

    def check_gateway(self) -> bool:
        """Check if gateway is responding"""
        try:
            response = self.session.get(f"{self.gateway_url}/api/v1/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Gateway responding (status: {data.get('status')})")
                return True
            else:
                print(f"✗ Gateway returned status {response.status_code}")
                return False
        except requests.RequestException as e:
            print(f"✗ Gateway not responding: {e}")
            return False

    def check_nodes(self) -> Dict[str, Any]:
        """Check node status"""
        try:
            response = self.session.get(f"{self.gateway_url}/api/v1/nodes", timeout=5)
            if response.status_code == 200:
                data = response.json()
                nodes = data.get('nodes', [])
                print(f"✓ {len(nodes)} nodes detected")

                # Analyze node status
                active_nodes = [n for n in nodes if n.get('status') == 'active']
                low_energy = [n for n in nodes if n.get('energy_uj', 0) < 50.0]

                print(f"  Active: {len(active_nodes)}")
                print(f"  Low energy (<50μJ): {len(low_energy)}")

                if low_energy:
                    print(f"  Low energy nodes: {[n['node_id'] for n in low_energy]}")

                return data
            else:
                print(f"✗ Failed to get node data: {response.status_code}")
                return {}
        except requests.RequestException as e:
            print(f"✗ Error checking nodes: {e}")
            return {}

    def check_shapley(self) -> bool:
        """Check Shapley value computation"""
        try:
            response = self.session.get(f"{self.gateway_url}/api/v1/shapley", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('converged'):
                    error = data.get('error', 0)
                    print(f"✓ Shapley values converged (error: {error:.4f})")
                    return True
                else:
                    print("✗ Shapley values not converged")
                    return False
            else:
                print(f"✗ Failed to get Shapley values: {response.status_code}")
                return False
        except requests.RequestException as e:
            print(f"✗ Error checking Shapley: {e}")
            return False

    def check_energy(self) -> Dict[str, Any]:
        """Check energy statistics"""
        try:
            response = self.session.get(f"{self.gateway_url}/api/v1/energy/harvest?duration=1h", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Energy data retrieved")
                print(f"  Total harvested: {data.get('total_harvested_uj', 0):.2f} μJ")
                print(f"  Average rate: {data.get('average_rate_uj_per_slot', 0):.2f} μJ/slot")
                return data
            else:
                print(f"✗ Failed to get energy data: {response.status_code}")
                return {}
        except requests.RequestException as e:
            print(f"✗ Error checking energy: {e}")
            return {}

    def check_memory(self) -> bool:
        """Check gateway memory usage"""
        try:
            response = self.session.get(f"{self.gateway_url}/api/v1/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                memory_mb = data.get('memory_usage_mb', 0)
                print(f"  Memory usage: {memory_mb:.2f} MB")

                if memory_mb > 1024:
                    print(f"  ⚠ High memory usage")
                else:
                    print(f"  ✓ Memory usage OK")

                return True
            else:
                return False
        except:
            return False

    def check_performance(self) -> Dict[str, Any]:
        """Check system performance metrics"""
        metrics = {}

        # Check API latency
        start = time.time()
        try:
            response = self.session.get(f"{self.gateway_url}/api/v1/status", timeout=5)
            latency = (time.time() - start) * 1000
            metrics['api_latency_ms'] = latency

            if latency < 100:
                print(f"✓ API latency: {latency:.2f}ms (good)")
            elif latency < 500:
                print(f"⚠ API latency: {latency:.2f}ms (acceptable)")
            else:
                print(f"✗ API latency: {latency:.2f}ms (slow)")

        except:
            metrics['api_latency_ms'] = -1

        return metrics

    def run_full_diagnostic(self):
        """Run complete diagnostic suite"""
        print("="*60)
        print("ECTC System Diagnostic")
        print("="*60)
        print()

        results = {
            'gateway': self.check_gateway(),
            'nodes': {},
            'shapley': False,
            'energy': {},
            'performance': {}
        }

        if not results['gateway']:
            print("\nGateway not responding. Cannot continue diagnostic.")
            return 1

        print()
        results['nodes'] = self.check_nodes()
        print()
        results['shapley'] = self.check_shapley()
        print()
        results['energy'] = self.check_energy()
        print()
        results['performance'] = self.check_performance()
        print()

        # Generate report
        self.generate_report(results)

        return 0

    def generate_report(self, results: Dict[str, Any]):
        """Generate diagnostic report"""
        print("="*60)
        print("DIAGNOSTIC SUMMARY")
        print("="*60)
        print()

        checks = [
            ('Gateway', results['gateway']),
            ('Nodes', len(results['nodes'].get('nodes', [])) > 0),
            ('Shapley', results['shapley']),
            ('Energy Data', len(results['energy']) > 0),
        ]

        for name, passed in checks:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}: {name}")

        print()
        print("Recommendations:")
        print()

        if not results['gateway']:
            print("- Check if gateway service is running")
            print("- Verify port 8080 is not blocked")
            print("- Check firewall settings")

        if len(results['nodes'].get('nodes', [])) == 0:
            print("- No nodes detected")
            print("- Verify radio modules are connected")
            print("- Check if nodes are powered on")

        if not results['shapley']:
            print("- Shapley values not converging")
            print("- Check if network is stable")
            print("- Verify algorithm parameters")

        performance = results['performance']
        latency = performance.get('api_latency_ms', 0)

        if latency > 500:
            print(f"- High API latency ({latency:.2f}ms)")
            print("- Check server load")
            print("- Consider scaling gateway")

        print()
        print("="*60)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='ECTC Diagnostic Tool')
    parser.add_argument('--gateway', default='http://localhost:8080',
                       help='Gateway URL')
    parser.add_argument('--check', choices=['all', 'gateway', 'nodes', 'shapley', 'energy'],
                       default='all', help='Check to run')

    args = parser.parse_args()

    diag = ETCDiagnostics(args.gateway)

    if args.check == 'all':
        return diag.run_full_diagnostic()
    elif args.check == 'gateway':
        return 0 if diag.check_gateway() else 1
    elif args.check == 'nodes':
        return 0 if len(diag.check_nodes().get('nodes', [])) > 0 else 1
    elif args.check == 'shapley':
        return 0 if diag.check_shapley() else 1
    elif args.check == 'energy':
        return 0 if len(diag.check_energy()) > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
