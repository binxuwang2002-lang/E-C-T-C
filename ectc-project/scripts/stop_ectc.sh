#!/bin/bash
# ECTC Stop Script
# Stops all ECTC services

echo "Stopping ECTC services..."

# Kill gateway process
pkill -f "ectc_gateway.main" || echo "Gateway not running"

# Kill evaluation processes
pkill -f "evaluator.py" || echo "Evaluator not running"

# Stop docker containers
docker-compose down 2>/dev/null || echo "Docker containers not running"

echo "ECTC services stopped"
