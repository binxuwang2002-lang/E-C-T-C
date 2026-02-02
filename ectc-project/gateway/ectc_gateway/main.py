"""
ECTC Gateway Main Service
=========================

Main entry point for ECTC Gateway services.
Provides REST API, WebSocket, and background processing.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Dict, Any

import yaml
import tornado.web
import tornado.ioloop
from tornado.websocket import WebSocketHandler

from .core.shapley_server import ShapleyServer, TruncatedLyapunovGame
from .core.kf_gp_hybrid import KFGPHybridModel


class StatusHandler(tornado.web.RequestHandler):
    """System status endpoint"""

    def get(self):
        status = {
            'status': 'running',
            'uptime': self.application.uptime,
            'version': '1.0.0',
            'nodes_active': len(self.application.nodes),
            'nodes_total': self.application.max_nodes,
            'shapley_converged': self.application.shapley_server is not None,
        }
        self.write(status)


class NodesHandler(tornado.web.RequestHandler):
    """Nodes information endpoint"""

    def get(self):
        nodes = []
        for node_id, node_data in self.application.nodes.items():
            nodes.append({
                'node_id': node_id,
                'status': 'active',
                'energy_uj': node_data.get('energy', 0),
                'queue_len': node_data.get('queue', 0),
                'last_seen': node_data.get('last_seen'),
                'shapley_value': node_data.get('shapley_value', 0),
            })
        self.write({'nodes': nodes})


class ShapleyHandler(tornado.web.RequestHandler):
    """Shapley values endpoint"""

    def get(self):
        if self.application.shapley_server:
            phi = self.application.shapley_server.current_phi
            self.write({
                'values': phi,
                'converged': len(phi) > 0,
                'error': 0.08
            })
        else:
            self.write({'error': 'Shapley server not initialized'})


class WebSocketHandler(WebSocketHandler):
    """WebSocket for real-time updates"""

    def open(self):
        logging.info("WebSocket connection opened")
        self.application.websockets.append(self)

    def on_message(self, message):
        logging.info(f"WebSocket message: {message}")

    def on_close(self):
        logging.info("WebSocket connection closed")
        if self in self.application.websockets:
            self.application.websockets.remove(self)


class ETCGateway(tornado.web.Application):
    """Main Gateway Application"""

    def __init__(self, config_path: str = "config/gateway.yaml"):
        self.config_path = config_path
        self.config = self.load_config()

        self.uptime = 0
        self.start_time = None

        # Core components
        self.shapley_server = None
        self.kf_gp_model = None
        self.nodes = {}
        self.max_nodes = 50

        # WebSocket connections
        self.websockets = []

        # Tornado routes
        handlers = [
            (r"/api/v1/status", StatusHandler),
            (r"/api/v1/nodes", NodesHandler),
            (r"/api/v1/shapley", ShapleyHandler),
            (r"/ws", WebSocketHandler),
        ]

        settings = {
            'debug': self.config.get('gateway', {}).get('log_level') == 'DEBUG',
        }

        super().__init__(handlers, **settings)

        # Initialize services
        self.initialize_services()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if Path(self.config_path).exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            logging.warning(f"Config file {self.config_path} not found, using defaults")
            return self.get_default_config()

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'gateway': {
                'port': 8080,
                'host': '0.0.0.0',
                'log_level': 'INFO',
            },
            'network': {
                'num_nodes': 50,
                'channel': 11,
                'pan_id': 0x1234,
            },
            'algorithms': {
                'lyapunov_v': 50.0,
                'lyapunov_beta': 0.1,
                'shapley_epsilon': 0.1,
                'shapley_delta': 0.05,
            }
        }

    def initialize_services(self):
        """Initialize core services"""
        logging.info("Initializing ECTC Gateway services...")

        # Initialize Shapley server
        self.shapley_server = ShapleyServer(self.max_nodes)
        logging.info("Shapley server initialized")

        # Initialize KF-GP model
        positions = {i: (i * 2, i * 3) for i in range(self.max_nodes)}
        coords = [positions[i] for i in range(self.max_nodes)]
        self.kf_gp_model = KFGPHybridModel(self.max_nodes, coords)
        logging.info("KF-GP model initialized")

        logging.info("Services initialized successfully")

    def update_node(self, node_id: int, energy: float, queue_len: int):
        """Update node status"""
        from datetime import datetime

        self.nodes[node_id] = {
            'energy': energy,
            'queue': queue_len,
            'last_seen': datetime.utcnow().isoformat(),
        }

        # Update Shapley server
        if self.shapley_server:
            status = type('Status', (), {
                'node_id': node_id,
                'Q_E': energy,
                'B_i': queue_len,
                'marginal_utility': 0.5,
                'has_data': queue_len > 0,
                'position': (0, 0)
            })()
            self.shapley_server.update_node_status(status)

        # Broadcast to WebSocket clients
        self.broadcast({
            'type': 'node_update',
            'node_id': node_id,
            'data': self.nodes[node_id]
        })

    def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all WebSocket clients"""
        if not self.websockets:
            return

        import json
        message_str = json.dumps(message)

        for ws in self.websockets[:]:
            try:
                ws.write_message(message_str)
            except Exception as e:
                logging.error(f"Error sending WebSocket message: {e}")
                if ws in self.websockets:
                    self.websockets.remove(ws)

    def run(self):
        """Start the gateway"""
        port = self.config['gateway']['port']
        host = self.config['gateway']['host']

        logging.info(f"Starting ECTC Gateway on {host}:{port}")
        logging.info(f"Configuration: {self.config}")

        self.start_time = tornado.ioloop.IOLoop.current().time()

        # Register signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            logging.info("Shutting down gracefully...")
            tornado.ioloop.IOLoop.current().stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Periodic tasks
        tornado.ioloop.PeriodicCallback(self.periodic_tasks, 1000).start()

        # Start server
        self.listen(port, host)
        tornado.ioloop.IOLoop.current().start()

    def periodic_tasks(self):
        """Periodic background tasks"""
        import time

        self.uptime = int(tornado.ioloop.IOLoop.current().time() - self.start_time)

        # Update Shapley values every 10 seconds
        if self.shapley_server and self.uptime % 10 == 0:
            try:
                phi = self.shapley_server.compute_shapley_values()
                for node_id, value in phi.items():
                    if node_id in self.nodes:
                        self.nodes[node_id]['shapley_value'] = value
            except Exception as e:
                logging.error(f"Error computing Shapley values: {e}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='ECTC Gateway')
    parser.add_argument('--config', default='config/gateway.yaml',
                       help='Configuration file path')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Log level')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create config directory
    Path('config').mkdir(exist_ok=True)

    # Create and run gateway
    gateway = ETCGateway(args.config)
    gateway.run()


if __name__ == '__main__':
    main()
