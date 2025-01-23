# Benchmarking Language Models with PyTorch

This repository provides a Python-based benchmarking tool for evaluating pre-trained language models using Hugging Face's `transformers` library. The tool measures key metrics, including latency, memory usage, throughput, and token efficiency, across multiple devices such as CPU, GPU, and Apple Silicon (MPS).

## Features

* Load and benchmark pre-trained language models from Hugging Face
* Measure the following performance metrics:
  * **Startup Latency**: Time to load the model
  * **Inference Latency**: Time to generate a response
  * **Memory Usage**: Peak memory consumption during inference
  * **Throughput**: Number of requests handled per second
  * **Token Efficiency**: Tokens generated per second
* Supports dynamic device detection (`cpu`, `cuda`, `mps`)
* Logs results for easier analysis

## Requirements

* Python 3.8 or later
* Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/benchmarking-tool.git
   cd benchmarking-tool
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the script with the following command:

```bash
python benchmark.py
```

Edit the `models_to_test` list in `benchmark.py` to include the models you want to benchmark.

Output results are printed to the console and can be exported for further analysis.

### Example Output

```plaintext
2025-01-22 10:00:00 - INFO - Loading model: microsoft/phi-4
2025-01-22 10:00:30 - INFO - Benchmark results for microsoft/phi-4:
{
    "Model": "microsoft/phi-4",
    "Startup Latency (ms)": 30567.45,
    "Latency (ms)": 543.78,
    "Memory Usage (MB)": 512.34,
    "Throughput (req/sec)": 1.45,
    "Response Size (tokens)": 150,
    "Token Efficiency (tokens/sec)": 275.56,
    "Energy Usage (J)": 125.78,
    "CPU Utilization (%)": 45.2
}
```

## Docker Usage

A Dockerfile is provided to containerize the application.

1. Build the Docker image:
   ```bash
   docker build -t model-benchmarking .
   ```

2. Run the Docker container:
   ```bash
   docker run --rm -it model-benchmarking
   ```

## File Structure

* `benchmark.py`: The main script for benchmarking models
* `requirements.txt`: Python dependencies for the project
* `Dockerfile`: Docker configuration for containerizing the application

## Contributions

Contributions are welcome! If you'd like to improve this project, feel free to submit a pull request or open an issue.