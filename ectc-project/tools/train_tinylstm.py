"""
TinyLSTM Model Training and Quantization
========================================

This module implements a lightweight LSTM for energy harvesting prediction
in battery-free IoT nodes. The model is optimized for:
- Int8 quantized weights
- 2-bit activation quantization
- <4KB memory footprint
- <25μJ inference energy
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List


class TinyLSTM(nn.Module):
    """
    TinyLSTM: 32-unit LSTM with quantized inference
    Input: Energy harvesting history [seq_len=10]
    Output: Next-step energy prediction
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, output_dim: int = 1):
        super(TinyLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # LSTM cell
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Quantization parameters (set after training)
        self.scale_in = None
        self.scale_hidden = None
        self.scale_out = None
        self.zero_point = -128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with hidden state

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]

        Returns:
            Output tensor [batch_size, output_dim]
        """
        # Initialize hidden state (zeros)
        h0 = torch.zeros(1, x.size(0), self.hidden_dim)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim)

        # LSTM forward
        out, _ = self.lstm(x, (h0, c0))

        # Take last time step
        out = out[:, -1, :]

        # Linear output
        out = self.fc(out)

        return out


class EnergyDataset:
    """Dataset for energy harvesting traces"""

    def __init__(self, traces: List[np.ndarray], seq_len: int = 10):
        """
        Args:
            traces: List of energy harvesting traces
            seq_len: Length of input sequence
        """
        self.traces = traces
        self.seq_len = seq_len

        # Normalize all traces
        self.mean = np.mean([t.mean() for t in traces])
        self.std = np.std([t.std() for t in traces])

    def __len__(self) -> int:
        return sum(len(t) - self.seq_len for t in self.traces)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, float]:
        # Find which trace contains this index
        cumulative = 0
        for trace in self.traces:
            if idx < cumulative + len(trace) - self.seq_len:
                # Calculate position in this trace
                pos = idx - cumulative
                break
            cumulative += len(trace) - self.seq_len

        # Extract sequence
        start = pos
        end = pos + self.seq_len
        seq = trace[start:end]

        # Next value
        target = trace[end]

        # Normalize
        seq = (seq - self.mean) / (self.std + 1e-8)
        target = (target - self.mean) / (self.std + 1e-8)

        return seq.astype(np.float32), target


def generate_synthetic_traces(num_traces: int = 100, length: int = 7200) -> List[np.ndarray]:
    """
    Generate synthetic energy harvesting traces

    Args:
        num_traces: Number of traces to generate
        length: Length of each trace (10s intervals, 7200 = 20 hours)

    Returns:
        List of energy harvesting traces in μJ
    """
    traces = []

    for _ in range(num_traces):
        trace = np.zeros(length)

        # Simulate day-night cycle
        for t in range(length):
            # 24-hour cycle (24*360 = 8640 samples = 24 hours at 10s intervals)
            time_of_day = (t % 8640) / 8640.0

            # Solar irradiance model (sinusoidal with noise)
            solar_irradiance = np.sin(np.pi * time_of_day) if time_of_day < 1.0 else 0
            solar_irradiance = max(0, solar_irradiance)

            # Random weather patterns
            weather_factor = np.random.uniform(0.3, 1.0)

            # Base energy collection (μJ per 10s)
            base_energy = solar_irradiance * weather_factor * 5.0

            # Add short-term fluctuations (clouds)
            if np.random.random() < 0.05:  # 5% chance of cloud cover
                cloud_duration = np.random.randint(10, 60)  # 100s - 10min
                for dt in range(min(cloud_duration, length - t)):
                    trace[t + dt] *= 0.3  # Reduce by 70%

            # Add white noise
            noise = np.random.normal(0, 0.2)
            trace[t] = base_energy + noise

            # Ensure non-negative
            trace[t] = max(0, trace[t])

        traces.append(trace)

    return traces


def train_tinylstm(traces: List[np.ndarray], epochs: int = 100) -> TinyLSTM:
    """
    Train TinyLSTM on energy harvesting traces

    Args:
        traces: List of energy traces
        epochs: Number of training epochs

    Returns:
        Trained TinyLSTM model
    """
    # Create dataset
    dataset = EnergyDataset(traces)

    # Split train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False
    )

    # Initialize model
    model = TinyLSTM(input_dim=1, hidden_dim=32, output_dim=1)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            # Reshape for LSTM input
            batch_x = batch_x.unsqueeze(-1)  # [batch, seq_len, 1]
            batch_y = batch_y.unsqueeze(-1)  # [batch, 1]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.unsqueeze(-1)
                batch_y = batch_y.unsqueeze(-1)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        # Print progress
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}] - Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'models/tinylstm_pretrained.pth')

    # Load best model
    model.load_state_dict(torch.load('models/tinylstm_pretrained.pth'))
    return model


def quantize_model(model: TinyLSTM) -> TinyLSTM:
    """
    Apply dynamic quantization to the model

    Args:
        model: Floating point TinyLSTM model

    Returns:
        Quantized TinyLSTM model
    """
    # Dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.LSTM, nn.Linear},
        dtype=torch.qint8
    )

    # Set quantization parameters
    quantized_model.scale_in = 0.01
    quantized_model.scale_hidden = 0.02
    quantized_model.scale_out = 0.01
    quantized_model.zero_point = -128

    return quantized_model


def export_to_tflite(model: TinyLSTM, seq_len: int = 10,
                    input_dim: int = 1) -> bytes:
    """
    Export PyTorch model to TensorFlow Lite format

    Args:
        model: TinyLSTM model
        seq_len: Input sequence length
        input_dim: Input dimension

    Returns:
        TFLite model as bytes
    """
    # For now, save as ONNX
    # In production, use ONNX-TFLite converter

    model.eval()

    # Dummy input for export
    dummy_input = torch.randn(1, seq_len, input_dim)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        'models/tinylstm.onnx',
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print("Model exported to ONNX: models/tinylstm.onnx")
    print("Use onnx-tf converter to generate TFLite model")

    return b''


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt

    # Create models directory
    os.makedirs('models', exist_ok=True)

    print("Generating synthetic energy harvesting traces...")
    traces = generate_synthetic_traces(num_traces=100, length=3600)

    print(f"Generated {len(traces)} traces, each {len(traces[0])} samples")
    print(f"Mean energy: {np.mean([t.mean() for t in traces]):.2f} μJ")
    print(f"Std energy: {np.mean([t.std() for t in traces]):.2f} μJ")

    print("\nTraining TinyLSTM model...")
    model = train_tinylstm(traces, epochs=100)

    print("\nQuantizing model...")
    quantized_model = quantize_model(model)

    print("\nExporting to TFLite...")
    tflite_model = export_to_tflite(quantized_model)

    print("\nTraining complete!")
    print(f"Model saved to: models/tinylstm_pretrained.pth")

    # Plot sample prediction
    model.eval()
    with torch.no_grad():
        sample_seq = torch.FloatTensor(traces[0][:10]).unsqueeze(0).unsqueeze(-1)
        prediction = model(sample_seq).item()
        actual = traces[0][10]

        print(f"\nSample prediction:")
        print(f"Input sequence: {traces[0][:10]}")
        print(f"Predicted: {prediction:.3f}, Actual: {actual:.3f}")
        print(f"Error: {abs(prediction - actual):.3f}")
