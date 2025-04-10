# Federated Learning System Documentation

NOTE: This repo is the for the code for LongFed - a framework for client selection in FL clusters. This is for benchmarking 'SwaFL' - a novel client selection framework for heterogenous, low-resourced FL clusters.

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Components](#components)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Detailed Component Documentation](#detailed-component-documentation)
7. [Metrics and Monitoring](#metrics-and-monitoring)
8. [Usage Examples](#usage-examples)

## Overview
This federated learning system implements a distributed machine learning framework where multiple clients collaborate to train a global model while keeping their data private. The system uses a CNN model for image classification on the CIFAR-10 dataset.

### Key Features
- Selective client participation based on LongFed.
- Resource monitoring and performance tracking
- Adaptive sampling using Dirichlet distribution
- Comprehensive metrics logging
- Fault tolerance and error handling

## System Architecture

### Server
- Central coordinator for federated learning rounds
- Manages client registration and participation
- Aggregates model updates using FedAvg algorithm
- Evaluates global model performance

### Clients
- Train local models on private data
- Make intelligent participation decisions
- Monitor system resources
- Track performance metrics

## Components

### 1. Client Component (`client.py`)
The client implementation contains several key classes and methods:

#### FederatedClient Class
Key responsibilities:
- Initializes connection with server
- Creates and manages local model
- Handles client configuration
- Sets up monitoring

#### Training Methods
Handles:
- Local model training
- Resource monitoring during training
- Performance metric collection

#### Data Sampling
Features:
- Implements Dirichlet distribution sampling
- Ensures balanced class representation
- Handles random data selection

### 2. Monitoring Component (`monitoring.py`)
Features:
- Real-time CPU and memory monitoring
- Metric logging to CSV files
- Thread-safe resource tracking
- Comprehensive metric collection

#### Metrics Tracked:
System Metrics:
   - CPU utilization
   - Memory usage
   - Training time
   - Model accuracy

## Configuration

### config.json

Parameters:
- `dirichlet_alpha`: Controls data distribution skewness
- `participation_threshold`: Percentage of clients that should participate
- `min_clients`: Minimum required participating clients
- `rounds`: Total training rounds

## Detailed Component Documentation

### Data Distribution System
Implements sophisticated data sampling:

1. Dirichlet Distribution:
   - Controls class distribution across clients
   - Configurable via alpha parameter
   - Ensures realistic non-IID scenarios

2. Adaptive Sampling:
   - 50% data sampling per round
   - Class-aware selection
   - Balance maintenance

### Monitoring System
Comprehensive metric tracking:

1. Real-time Monitoring:
   - Thread-based resource tracking
   - Configurable sampling frequency
   - Non-blocking implementation

2. Metric Storage:
   - CSV-based logging
   - Timestamp-based tracking
   - Participation history

## Usage Examples

### Starting a Client

```python
from client import FederatedClient

client = FederatedClient(client_id=1, server_url='http://127.0.0.1:5000')
client.start_client()
```

### Running the Setup
    - Install the required packages
        ```bash
        pip install -r requirements.txt
        ```
    - Configure the config.json file
    - Run the server.py file 
        ```python
        python server.py
        ```
    - Run the test_federated.py file
        ```python
        python test_federated.py
        ```

## Performance Considerations

1. Resource Management:
   - Single CPU core utilization
   - Controlled memory usage
   - Thread-safe operations

2. Error Handling:
   - Graceful failure recovery
   - Metric logging persistence
   - Connection retry logic

3. Scalability:
   - Asynchronous client operations
   - Efficient resource monitoring
   - Minimal memory footprint

This system provides a robust, scalable, and monitored federated learning implementation with intelligent participation decisions and comprehensive metric tracking.
