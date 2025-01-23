import time
import psutil
import torch
import logging
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_name, device="cpu"):
    """
    Load the specified model and tokenizer.
    Args:
        model_name (str): The name or path of the model to load.
        device (str): The device to load the model onto ('cpu', 'cuda', or 'mps').
    Returns:
        model: The loaded model.
        tokenizer: The corresponding tokenizer.
    """
    logging.info(f"Loading model: {model_name}")
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    model.to(device)
    startup_latency = (time.time() - start_time) * 1000  # Convert to milliseconds
    return model, tokenizer, startup_latency

def run_single_inference(model, tokenizer, prompt):
    """
    Run inference on the given model and prompt.
    Args:
        model: The language model.
        tokenizer: The tokenizer associated with the model.
        prompt (str): The input prompt for the model.
    Returns:
        outputs: The generated outputs from the model.
        inputs: The tokenized inputs used for inference.
        elapsed_time (float): Time taken for inference in milliseconds.
        mem_before: Memory usage before inference.
        mem_after: Memory usage after inference.
        energy_usage: Energy usage during inference.
        cpu_utilization: Average CPU utilization during inference.
    """
    try:
        # Tokenize input and move to model's device
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Prepare for memory and CPU tracking
        process = psutil.Process()
        mem_before = process.memory_info().rss  # Memory before inference
        
        # Start tracking CPU utilization
        start_cpu_percent = psutil.cpu_percent(interval=None)
        
        # Start timer and perform inference
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id)
        elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Measure memory and CPU usage after inference
        mem_after = process.memory_info().rss  # Memory after inference
        end_cpu_percent = psutil.cpu_percent(interval=None)
        avg_cpu_utilization = (start_cpu_percent + end_cpu_percent) / 2  # Approximate CPU utilization
        
        # Estimate energy usage (simplistic model)
        energy_usage = avg_cpu_utilization * (elapsed_time / 1000)  # Energy in Joules (approximation)
        
        return outputs, inputs, elapsed_time, mem_before, mem_after, energy_usage, avg_cpu_utilization
    except Exception as e:
        logging.error(f"Error during single inference: {e}")
        return None, None, None, None, None, None, None

def measure_latency(elapsed_time):
    """Return the latency of the inference process."""
    return elapsed_time

def measure_memory_usage(mem_before, mem_after):
    """Calculate memory usage in megabytes."""
    return (mem_after - mem_before) / (1024 * 1024)

def measure_response_size(outputs, tokenizer):
    """
    Measure the size of the model's response in tokens.
    Args:
        outputs: The generated outputs from the model.
        tokenizer: The tokenizer associated with the model.
    Returns:
        int: The size of the response in tokens.
    """
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_tokens = tokenizer(response, return_tensors="pt")["input_ids"]
    return response_tokens.size(1)

def measure_token_efficiency(outputs, elapsed_time):
    """
    Measure the efficiency of token generation.
    Args:
        outputs: The generated outputs from the model.
        elapsed_time (float): Time taken for inference in milliseconds.
    Returns:
        float: Tokens generated per second.
    """
    num_tokens = outputs[0].size(0)
    return num_tokens / (elapsed_time / 1000)  # Tokens per second

def measure_throughput(model, tokenizer, prompt, duration=10):
    """
    Measure the throughput of the model over a fixed duration.
    Args:
        model: The language model.
        tokenizer: The tokenizer associated with the model.
        prompt (str): The input prompt for the model.
        duration (int): Duration over which to measure throughput (in seconds).
    Returns:
        float: Number of requests processed per second.
    """
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        end_time = time.time() + duration
        iterations = 0

        while time.time() < end_time:
            with torch.no_grad():
                model.generate(**inputs, pad_token_id=tokenizer.eos_token_id)
            iterations += 1

        return iterations / duration  # Requests per second
    except Exception as e:
        logging.error(f"Error measuring throughput: {e}")
        return None

def benchmark_model(model_name, prompt):
    """
    Run all benchmark evaluations on the specified model.
    Args:
        model_name (str): The name or path of the model to benchmark.
        prompt (str): The input prompt for the model.
    Returns:
        dict: Benchmark results including latency, memory usage, throughput, and response size.
    """
    logging.info(f"Benchmarking model: {model_name}")
    model, tokenizer, startup_latency = load_model(model_name, device="cpu")
    if model is None or tokenizer is None:
        logging.error(f"Failed to load model: {model_name}")
        return {"Model": model_name, "Error": "Failed to load model"}

    # Single inference for static metrics
    outputs, inputs, elapsed_time, mem_before, mem_after, energy_usage, cpu_utilization = run_single_inference(model, tokenizer, prompt)
    if outputs is None:
        return {"Model": model_name, "Error": "Inference failed"}

    # Calculate metrics
    latency = measure_latency(elapsed_time)
    memory_usage = measure_memory_usage(mem_before, mem_after)
    response_size = measure_response_size(outputs, tokenizer)
    token_efficiency = measure_token_efficiency(outputs, elapsed_time)
    throughput = measure_throughput(model, tokenizer, prompt)

    # Combine results
    results = {
        "Model": model_name,
        "Startup Latency (ms)": startup_latency,
        "Latency (ms)": latency,
        "Memory Usage (MB)": memory_usage,
        "Throughput (req/sec)": throughput,
        "Response Size (tokens)": response_size,
        "Token Efficiency (tokens/sec)": token_efficiency,
        "Energy Usage (J)": energy_usage,
        "CPU Utilization (%)": cpu_utilization,
    }

    logging.info(f"Benchmark results for {model_name}: {results}")
    return results

def clear_torch_cache():
    """
    Clears the PyTorch CUDA cache.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("PyTorch CUDA cache cleared.")

        
if __name__ == "__main__":
    clear_torch_cache()
    # Models to benchmark
    models_to_test = [
        "microsoft/phi-4",
		"meta-llama/Llama-3.2-3B",
		"meta-llama/Llama-3.2-1B",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    ]

    # Input prompt
    prompt = "As a financial advisor, explain the benefits and risks of diversifying investments into real estate, technology stocks, and government bonds. Provide examples and keep the response under 100 words."

    final_result = []
    # Run benchmarks
    for model_name in models_to_test:
        results = benchmark_model(model_name, prompt)
        final_result.append(results)
    print(final_result)
