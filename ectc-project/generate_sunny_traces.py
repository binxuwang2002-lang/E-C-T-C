# Sunny Day Energy Traces - Extended
# 24 hours of solar energy harvesting data (8640 samples)
# Model: Gamma distribution with diurnal cycle
# Unit: μJ per 10-second slot

import numpy as np
import csv

# Generate 24 hours of data
num_samples = 8640  # 24 * 360 * 10

# Day-night cycle
def solar_irradiance(time_step):
    """Generate solar irradiance based on time of day"""
    # 24-hour cycle
    day_fraction = (time_step % 8640) / 8640.0

    # Solar curve (sinusoidal)
    if 0.0 <= day_fraction <= 0.5:  # Day (12 hours)
        solar = np.sin(np.pi * day_fraction * 2)
        # Peak at noon
        solar = max(0, solar)
    else:  # Night (12 hours)
        solar = 0.0

    return solar

# Generate traces
traces = []
for sensor_id in range(5):
    trace = []
    for t in range(num_samples):
        # Base solar
        base = solar_irradiance(t)

        # Random weather variation
        weather = np.random.uniform(0.3, 1.0)

        # Cloud cover (5% chance)
        if np.random.random() < 0.05:
            cloud_factor = 0.3
        else:
            cloud_factor = 1.0

        # Position factor (orientation, shade)
        position = np.random.uniform(0.8, 1.2)

        # Calculate energy (μJ per 10s slot)
        energy = base * weather * cloud_factor * position * 5.0

        # Add noise
        energy += np.random.normal(0, 0.2)
        energy = max(0, energy)

        trace.append(energy)

    traces.append(trace)

# Save to CSV
for sensor_id, trace in enumerate(traces):
    filename = f"sensor_{sensor_id+1:03d}_30days.csv"
    filepath = f"evaluation/traces/sunny/{filename}"

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'energy_uj'])
        for t, energy in enumerate(trace):
            writer.writerow([t, f"{energy:.2f}"])

print(f"Generated {len(traces)} sunny traces")
print(f"Each trace: {num_samples} samples")
print(f"Total: {len(traces) * num_samples} data points")
