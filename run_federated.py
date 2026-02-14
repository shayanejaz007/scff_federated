#!/usr/bin/env python3
"""
Run Federated SCFF Training

Convenient script to run the entire federated training setup.
Launches server and multiple clients in sequence for testing.

For actual distributed training, run server and clients separately:
  Terminal 1: python federated/server.py --num_clients 2 ...
  Terminal 2: python federated/client.py --client_id 0 ...
  Terminal 3: python federated/client.py --client_id 1 ...
"""
import argparse
import subprocess
import sys
import time
import threading
import os


def run_server(args, output_list):
    """Run the server process."""
    cmd = [
        sys.executable, 'federated/server.py',
        '--dataset', args.dataset,
        '--NL', str(args.NL),
        '--device', args.device,
        '--seed', str(args.seed),
        '--num_clients', str(args.num_clients),
        '--rounds', str(args.rounds),
        '--local_epochs', str(args.local_epochs),
        '--port', str(args.port),
        '--config', args.config,
        '--batchsize', str(args.batchsize),
        '--eval_every', str(args.eval_every)
    ]
    
    if args.local_steps:
        cmd.extend(['--local_steps', str(args.local_steps)])
    
    env = os.environ.copy()
    env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, env=env
    )
    
    output = []
    for line in process.stdout:
        print(f"[SERVER] {line}", end='')
        output.append(line)
    
    process.wait()
    output_list.append(''.join(output))


def run_client(client_id, args, output_list):
    """Run a client process."""
    # Small delay to ensure server is ready
    time.sleep(2 + client_id * 0.5)
    
    cmd = [
        sys.executable, 'federated/client.py',
        '--client_id', str(client_id),
        '--dataset', args.dataset,
        '--NL', str(args.NL),
        '--device', args.device,
        '--seed', str(args.seed),
        '--num_clients', str(args.num_clients),
        '--server_addr', 'localhost',
        '--port', str(args.port),
        '--config', args.config,
        '--batchsize', str(args.batchsize),
        '--dataset_size', str(args.dataset_size)
    ]
    
    env = os.environ.copy()
    env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, env=env
    )
    
    output = []
    for line in process.stdout:
        print(f"[CLIENT {client_id}] {line}", end='')
        output.append(line)
    
    process.wait()
    output_list.append(''.join(output))


def get_arguments():
    parser = argparse.ArgumentParser(description="Run Federated SCFF Training")
    parser.add_argument("--dataset", type=str, default="CIFAR")
    parser.add_argument("--NL", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--local_steps", type=int, default=None)
    parser.add_argument("--port", type=int, default=29500)
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--batchsize", type=int, default=100)
    parser.add_argument("--dataset_size", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=1)
    return parser


def main():
    parser = get_arguments()
    args = parser.parse_args()
    
    print("=" * 70)
    print("Federated SCFF Training Launcher")
    print("=" * 70)
    print(f"Launching 1 server + {args.num_clients} clients")
    print(f"Dataset: {args.dataset_size} samples split across {args.num_clients} clients")
    print(f"Training: {args.rounds} rounds, {args.local_epochs} local epochs each")
    print("=" * 70)
    
    # Start server thread
    server_output = []
    server_thread = threading.Thread(target=run_server, args=(args, server_output))
    server_thread.start()
    
    # Start client threads
    client_threads = []
    client_outputs = []
    
    for client_id in range(args.num_clients):
        output = []
        client_outputs.append(output)
        t = threading.Thread(target=run_client, args=(client_id, args, output))
        client_threads.append(t)
        t.start()
    
    # Wait for all to complete
    for t in client_threads:
        t.join()
    server_thread.join()
    
    print("\n" + "=" * 70)
    print("All processes completed")
    print("=" * 70)


if __name__ == "__main__":
    main()
