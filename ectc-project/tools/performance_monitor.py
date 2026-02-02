"""
Performance Monitor for ECTC Gateway
====================================

Real-time performance monitoring and alerting.
"""

import time
import psutil
import logging
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class PerformanceMetrics:
    """Performance metrics collection"""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    disk_usage_percent: float = 0.0
    network_io_mb: float = 0.0
    active_connections: int = 0
    request_rate: float = 0.0
    avg_response_time_ms: float = 0.0
    error_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)


class PerformanceMonitor:
    """Performance monitoring system"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = []
        self.start_time = time.time()
        self.request_times = []
        self.error_count = 0
        self.request_count = 0

    def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        # CPU and memory
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # Network I/O
        network = psutil.net_io_counters()

        # Count active connections (simplified)
        connections = len(psutil.net_connections())

        metrics = PerformanceMetrics(
            cpu_percent=cpu,
            memory_mb=memory.used / (1024 * 1024),
            memory_percent=memory.percent,
            disk_usage_percent=disk.percent,
            network_io_mb=(network.bytes_sent + network.bytes_recv) / (1024 * 1024),
            active_connections=connections,
            request_rate=len(self.request_times) / window_size if self.request_times else 0.0,
            avg_response_time_ms=self._calculate_avg_response_time(),
            error_rate=self.error_count / max(1, self.request_count)
        )

        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.window_size:
            self.metrics_history.pop(0)

        return metrics

    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time"""
        if not self.request_times:
            return 0.0

        return sum(self.request_times) / len(self.request_times)

    def record_request(self, response_time_ms: float):
        """Record a request"""
        self.request_times.append(response_time_ms)
        self.request_count += 1

        # Keep only recent requests
        if len(self.request_times) > self.window_size:
            self.request_times.pop(0)

    def record_error(self):
        """Record an error"""
        self.error_count += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics_history:
            return {}

        latest = self.metrics_history[-1]

        return {
            'uptime_seconds': time.time() - self.start_time,
            'cpu_percent': latest.cpu_percent,
            'memory_mb': latest.memory_mb,
            'memory_percent': latest.memory_percent,
            'active_connections': latest.active_connections,
            'request_rate': latest.request_rate,
            'avg_response_time_ms': latest.avg_response_time_ms,
            'error_rate': latest.error_rate,
            'health_score': self._calculate_health_score()
        }

    def _calculate_health_score(self) -> float:
        """Calculate overall health score (0-100)"""
        if not self.metrics_history:
            return 0.0

        latest = self.metrics_history[-1]

        # CPU score (0-100)
        cpu_score = max(0, 100 - latest.cpu_percent)

        # Memory score (0-100)
        memory_score = max(0, 100 - latest.memory_percent)

        # Response time score (0-100, 100ms is good)
        response_score = max(0, 100 - latest.avg_response_time_ms)

        # Error rate score (0-100, 0% is good)
        error_score = max(0, 100 - (latest.error_rate * 100))

        # Weighted average
        health = (cpu_score * 0.3 +
                 memory_score * 0.3 +
                 response_score * 0.2 +
                 error_score * 0.2)

        return health

    def check_thresholds(self) -> Dict[str, bool]:
        """Check if metrics exceed thresholds"""
        if not self.metrics_history:
            return {}

        latest = self.metrics_history[-1]

        return {
            'cpu_high': latest.cpu_percent > 80.0,
            'memory_high': latest.memory_percent > 85.0,
            'response_slow': latest.avg_response_time_ms > 500.0,
            'error_high': latest.error_rate > 0.05,
            'disk_high': latest.disk_usage_percent > 90.0
        }

    def get_alerts(self) -> list:
        """Get current alerts"""
        alerts = []
        thresholds = self.check_thresholds()

        if thresholds['cpu_high']:
            alerts.append({
                'level': 'warning',
                'message': f'High CPU usage: {self.metrics_history[-1].cpu_percent:.1f}%'
            })

        if thresholds['memory_high']:
            alerts.append({
                'level': 'warning',
                'message': f'High memory usage: {self.metrics_history[-1].memory_percent:.1f}%'
            })

        if thresholds['response_slow']:
            alerts.append({
                'level': 'warning',
                'message': f'Slow response time: {self.metrics_history[-1].avg_response_time_ms:.1f}ms'
            })

        if thresholds['error_high']:
            alerts.append({
                'level': 'critical',
                'message': f'High error rate: {self.metrics_history[-1].error_rate*100:.1f}%'
            })

        if thresholds['disk_high']:
            alerts.append({
                'level': 'warning',
                'message': f'High disk usage: {self.metrics_history[-1].disk_usage_percent:.1f}%'
            })

        return alerts


if __name__ == '__main__':
    # Demo
    monitor = PerformanceMonitor()

    # Simulate requests
    for i in range(50):
        monitor.record_request(50 + i * 2)
        time.sleep(0.1)

    # Collect metrics
    metrics = monitor.collect_metrics()

    # Print summary
    print("Performance Summary:")
    print(f"  CPU: {metrics.cpu_percent:.1f}%")
    print(f"  Memory: {metrics.memory_mb:.1f} MB ({metrics.memory_percent:.1f}%)")
    print(f"  Response time: {metrics.avg_response_time_ms:.1f}ms")
    print(f"  Request rate: {metrics.request_rate:.1f}/s")
    print(f"  Error rate: {metrics.error_rate*100:.1f}%")

    # Get health score
    summary = monitor.get_summary()
    print(f"\nHealth Score: {summary['health_score']:.1f}/100")

    # Check alerts
    alerts = monitor.get_alerts()
    if alerts:
        print("\nAlerts:")
        for alert in alerts:
            print(f"  [{alert['level'].upper()}] {alert['message']}")
    else:
        print("\nNo alerts")
