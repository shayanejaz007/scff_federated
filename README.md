# Split Learning for Self Contrastive Forward Forward Networks

## Research Documentation

### Introduction

This repository documents my research on applying split learning techniques to Self Contrastive Forward Forward (SCFF) networks. The original SCFF algorithm was proposed by Siddiqui et al. in their paper ["Self-Contrastive Forward-Forward Algorithm"](https://arxiv.org/abs/2409.11593). My goal was to investigate whether SCFF could be trained in a distributed federated setting while maintaining accuracy comparable to centralized training.

I chose to work with SCFF because of its unique layer wise training approach. Unlike traditional backpropagation where gradients flow through the entire network, SCFF trains each layer independently using a local contrastive loss. This property made me curious about how it would behave in a distributed setting where data is spread across multiple clients.

### Research Question

Can we train SCFF networks across multiple clients without sharing raw data, and still achieve the same accuracy as training on centralized data?

**Answer:** Yes, and in my experiments the federated approach has a difference of +- 2% gap from centralized (78.26% vs 80.15% on CIFAR10 with 5 clients).

### Background on SCFF

Before diving into my implementation, let me briefly explain how SCFF works since understanding this is crucial for understanding my approach.

SCFF is based on the Forward Forward algorithm but adds a self contrastive component. The key idea is that each layer learns to distinguish between "positive" and "negative" inputs by computing a goodness score. The goodness is simply the mean of squared activations after ReLU:

```
goodness = mean(ReLU(activations)^2)
```

For training, the network needs two types of inputs:

1. Positive pairs: Two different augmented views of the same image concatenated together
2. Negative pairs: Augmented views of different images concatenated together

The layer learns to output high goodness for positive pairs and low goodness for negative pairs. The loss function penalizes low goodness on positives and high goodness on negatives.

What makes this interesting for distributed learning is that each layer trains completely independently. Layer 1 does not need any information from Layer 2 during training. This is fundamentally different from backpropagation where you need the full forward and backward pass through all layers.

### My Approach

I implemented a layer wise federated learning system. The basic idea is straightforward:

1. Train Layer 0 completely across all clients using federated averaging
2. Once Layer 0 is done, freeze it
3. Move to Layer 1 and repeat
4. Continue until all layers are trained

I chose this approach because it aligns naturally with how SCFF trains layers sequentially anyway. In centralized SCFF training, you typically train Layer 0 for some number of epochs, then freeze it and move to Layer 1. I am doing the same thing, just distributing each layer's training across multiple clients.

For the aggregation, I used FedAvg which computes a weighted average of client weights based on their number of samples:

```
global_weights = sum(num_samples_k / total_samples * weights_k)
```

### Implementation Details

The system consists of two main components:

**Server**: Coordinates training, broadcasts model weights, collects updates, and performs aggregation. The server also handles evaluation after each layer is trained.

**Clients**: Each client holds a partition of the training data. When instructed by the server, the client trains the current layer on its local data and sends the updated weights back.

The training flow for one layer looks like this:

```
Round 1:
    Server sends Layer 0 weights to all clients
    Client 0 trains on its 10,000 samples
    Client 1 trains on its 10,000 samples
    Client 2 trains on its 10,000 samples
    (and so on for all clients)
    All clients send updated weights to server
    Server averages the weights

Round 2:
    Server sends averaged weights back to clients
    Training continues...

After 20 rounds:
    Layer 0 is frozen
    Move to Layer 1
```

### Problems I Encountered

I want to document the issues I ran into because they took significant time to debug and might help others.

**The Augmentation Problem**

This was the biggest issue. I was getting around 68% test accuracy when the baseline centralized training achieves 76%. I spent a lot of time checking the federated aggregation logic, thinking there must be something wrong with how I was averaging weights or synchronizing state.

Turns out the problem was much simpler. The data loader was configured with `augment="no"` instead of `augment="dual"`. This means the model was receiving identical images as positive pairs. In contrastive learning, positive pairs need to be different views of the same image. If they are identical, the model cannot learn anything meaningful because it cannot distinguish positive from negative pairs based on content.

After fixing this single line, accuracy jumped to 75%.

**Client Drift**

My first implementation used sequential training where Client 0 trains, passes the model to Client 1, then Client 1 trains, and so on. This caused what is called catastrophic forgetting. By the time Client 4 finished training, the model had essentially "forgotten" what it learned from Client 0.

I measured this using metrics from the continual learning literature:

1. EWC Penalty: Measures how much important parameters changed
2. Backward Transfer: Measures accuracy change on earlier tasks

Both metrics confirmed significant forgetting was happening. The solution was to use proper FedAvg aggregation where all clients train in parallel and their contributions are averaged together, rather than sequential overwriting.

**Optimizer State Synchronization**

Initially I was only synchronizing model weights between server and clients. But the Adam optimizer maintains momentum and variance estimates for each parameter. When clients start each round with fresh optimizer states, training becomes unstable.

I ended up synchronizing the optimizer state as well, and also tracking a global batch counter for learning rate scheduling. This ensures all clients are on the same page regarding where they are in the training process.

### Experimental Setup

**Dataset**: CIFAR10 with 50,000 training images and 10,000 test images

**Model**: 3 layer SCFF network as specified in the original paper

**Clients**: 5 clients, each receiving 10,000 training images

**Training**: 20 rounds per layer, 1 local epoch per round

**Hardware**: Tested on CPU (works fine for CIFAR10 scale)

### Results

I ran the experiment with 5 clients on the full CIFAR10 dataset (50,000 training images split into 10,000 per client). Here are the actual results:

**Training Progress:**

| Epoch | Train Accuracy | Validation Accuracy |
|-------|----------------|---------------------|
| 1 | 55.67% | 60.76% |
| 11 | 72.46% | 73.57% |
| 21 | 75.04% | 74.78% |
| 31 | 77.53% | 77.20% |
| 41 | 78.33% | 78.15% |
| 50 | 78.65% | 78.26% |

**Final Results:**

| Metric | Value |
|--------|-------|
| Train Accuracy | 86.74% |
| Test Accuracy | 78.26% |
| Best Test | 78.26% |

**Comparison with Baseline:**

| Configuration | Test Accuracy |
|---------------|---------------|
| Centralized SCFF (baseline) | 80.15% |
| Federated SCFF (5 clients) | 78.26% |


The train test gap of about 8% (86.74% vs 78.26%) indicates some overfitting but is within acceptable range for CIFAR10. The model generalizes well to unseen data.

### How to Run the Experiments

First set up the environment:

```
cd scff_federated
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

You need 6 terminal windows to run the full experiment with 5 clients.

**Terminal 1 (Server):**
```
cd scff_federated
.\.venv\Scripts\activate
python federated/server.py --num_clients 5 --rounds_per_layer 20 --local_epochs 1 --port 29500 --chart training_history.png
```

**Terminal 2 (Client 0):**
```
cd scff_federated
.\.venv\Scripts\activate
python federated/client.py --client_id 0 --num_clients 5 --port 29500 --dataset_size 50000
```

**Terminal 3 (Client 1):**
```
cd scff_federated
.\.venv\Scripts\activate
python federated/client.py --client_id 1 --num_clients 5 --port 29500 --dataset_size 50000
```

**Terminal 4 (Client 2):**
```
cd scff_federated
.\.venv\Scripts\activate
python federated/client.py --client_id 2 --num_clients 5 --port 29500 --dataset_size 50000
```

**Terminal 5 (Client 3):**
```
cd scff_federated
.\.venv\Scripts\activate
python federated/client.py --client_id 3 --num_clients 5 --port 29500 --dataset_size 50000
```

**Terminal 6 (Client 4):**
```
cd scff_federated
.\.venv\Scripts\activate
python federated/client.py --client_id 4 --num_clients 5 --port 29500 --dataset_size 50000
```

Start the server first, then launch clients in any order. They will connect and wait for training commands.

### Diagnostic Tools

I wrote a diagnostics script to help verify that training is working correctly:

```
python diagnostics.py --test all
```

This runs three tests:

**Layer Learning Test**: Trains each layer for 50 batches and measures weight change. If weights are not changing, something is wrong.

**Augmentation Test**: Compares the goodness separation between `augment="no"` and `augment="dual"`. With proper dual augmentation, there should be clear separation between positive and negative goodness values.

**Client Drift Test**: Simulates sequential client training and measures forgetting using EWC penalty and backward transfer. This helped me identify the catastrophic forgetting problem early on.

### Code Structure

```
scff_federated/
├── config.json              Architecture configuration from original SCFF
├── requirements.txt         Python dependencies
├── diagnostics.py           Tests I wrote for debugging
├── scff/
│   ├── data.py              CIFAR10 with dual augmentation
│   ├── layers.py            Conv2d with weight standardization
│   ├── model.py             Layer construction utilities
│   ├── trainer.py           Local training loop
│   └── eval.py              Linear probe evaluation
└── federated/
    ├── partition.py         Splits data across clients
    ├── protocol.py          Message classes for communication
    ├── server.py            Federated server implementation
    └── client.py            Federated client implementation
```

### Privacy Analysis

In this implementation:

**What stays on the client:**
- All raw training images
- Individual sample gradients
- Local training dynamics

**What is sent to the server:**
- Model weights after local training
- Aggregate statistics (average loss, goodness values)
- Number of samples trained on

The server never sees raw data. However, I should note that model weights can potentially leak information about training data through model inversion attacks. This implementation provides what researchers call "honest but curious" privacy. For stronger guarantees, future work could add differential privacy (noise injection) or secure aggregation (cryptographic protocols).

### Conclusions

My main finding is that SCFF is surprisingly well suited for federated learning. The layer wise training approach means we can train each layer to completion across all clients before moving on, avoiding many of the convergence issues that plague standard federated learning.

The key insight is that SCFF layers are independent. There is no gradient flow between layers, so we do not have to worry about stale gradients or mismatched optimizer states across layers. Each layer can be treated as a separate federated learning problem.

### Future Directions

Things I would like to explore next:

1. Testing with non IID data distributions where clients have different class distributions
2. Adding differential privacy for formal privacy guarantees
3. Scaling to larger models and datasets
4. Comparing with other federated learning algorithms like FedProx and SCAFFOLD
5. Investigating asynchronous training where clients can operate at different speeds

### References

1. Siddiqui, S. A., et al. "Self-Contrastive Forward-Forward Algorithm." arXiv:2409.11593 (2024)

2. McMahan, B., et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS (2017)

3. Kirkpatrick, J., et al. "Overcoming catastrophic forgetting in neural networks." PNAS (2017)

4. Li, T., et al. "Federated Optimization in Heterogeneous Networks." MLSys (2020)

### Acknowledgments

This research builds on the SCFF implementation from the original paper. I adapted their training code for the federated setting and added the server/client architecture for distributed training.
