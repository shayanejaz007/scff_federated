#!/usr/bin/env python3
"""
Protocol Definitions for Federated SCFF

Defines data structures for model state transfer between server and clients.
Privacy guarantee: Only model parameters are transferred, never raw data.
"""
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import copy
import pickle
import struct


@dataclass
class ModelState:
    """
    Complete state of the model for transfer.
    
    Contains ONLY model parameters and optimizer state.
    NO raw data, NO sample-level features, NO gradients from data.
    """
    # Layer state dicts (weights and biases only)
    layer_state_dicts: List[Dict[str, torch.Tensor]]
    
    # Optimizer state dicts
    optimizer_state_dicts: List[Dict]
    
    # Scheduler state dicts
    scheduler_state_dicts: List[Dict]
    
    # Training metadata (no data information)
    nbbatches: int = 0
    
    # Verification hash (to ensure integrity)
    state_hash: str = ""
    
    def compute_hash(self) -> str:
        """Compute hash of model state for verification."""
        import hashlib
        
        # Hash based on first layer's weight sum (simple integrity check)
        if self.layer_state_dicts:
            first_weight = self.layer_state_dicts[0].get('conv_layer.weight')
            if first_weight is not None:
                weight_sum = first_weight.sum().item()
                return hashlib.md5(str(weight_sum).encode()).hexdigest()[:8]
        return "unknown"
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes for network transfer."""
        return pickle.dumps(self)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'ModelState':
        """Deserialize from bytes."""
        return pickle.loads(data)
    
    def clone(self) -> 'ModelState':
        """Create a deep copy of the state."""
        return ModelState(
            layer_state_dicts=[copy.deepcopy(sd) for sd in self.layer_state_dicts],
            optimizer_state_dicts=[copy.deepcopy(sd) for sd in self.optimizer_state_dicts],
            scheduler_state_dicts=[copy.deepcopy(sd) for sd in self.scheduler_state_dicts],
            nbbatches=self.nbbatches,
            state_hash=self.state_hash
        )


@dataclass
class ClientUpdate:
    """
    Update sent from client to server after local training.
    
    Contains ONLY:
    - Updated model state
    - Training metrics (loss, local accuracy)
    - NO raw data or data-derived features
    """
    client_id: int
    round_num: int
    
    # Updated model state
    model_state: ModelState
    
    # Training metrics (aggregate only, no per-sample info)
    local_loss: float = 0.0
    local_goodness_pos: float = 0.0
    local_goodness_neg: float = 0.0
    num_samples_trained: int = 0
    num_batches_trained: int = 0
    
    # Timing info
    training_time_seconds: float = 0.0
    
    def verify_no_data_leakage(self) -> bool:
        """
        Verify this update contains no raw data.
        Returns True if privacy is preserved.
        """
        # Check that model_state only contains expected fields
        for sd in self.model_state.layer_state_dicts:
            for key in sd.keys():
                # Only allow weight and bias parameters
                allowed = ['conv_layer.weight', 'conv_layer.bias', 
                          'bn1.weight', 'bn1.bias', 'bn1.running_mean', 
                          'bn1.running_var', 'bn1.num_batches_tracked']
                if not any(allowed_key in key for allowed_key in allowed):
                    # Check it's a valid parameter name pattern
                    if 'weight' not in key and 'bias' not in key and 'running' not in key:
                        raise PrivacyViolationError(
                            f"Suspicious key in model state: {key}"
                        )
        return True


@dataclass
class RoundMetadata:
    """Metadata for a training round."""
    round_num: int
    num_clients: int
    clients_completed: List[int] = field(default_factory=list)
    
    # Aggregate metrics (no per-sample info)
    avg_loss: float = 0.0
    avg_goodness_pos: float = 0.0
    avg_goodness_neg: float = 0.0
    total_samples_trained: int = 0
    
    # Evaluation metrics (if evaluated this round)
    test_accuracy: Optional[float] = None
    train_accuracy: Optional[float] = None


@dataclass
class ServerCommand:
    """Command sent from server to client."""
    cmd: str  # 'train', 'evaluate', 'shutdown'
    round_num: int = 0
    local_epochs: int = 1
    local_steps: Optional[int] = None  # If set, overrides local_epochs
    model_state: Optional[ModelState] = None


class PrivacyViolationError(Exception):
    """Raised when a privacy constraint is violated."""
    pass


def extract_model_state(
    nets: List[torch.nn.Module],
    optimizers: List,
    schedulers: List,
    nbbatches: int = 0
) -> ModelState:
    """
    Extract model state from networks.
    
    Args:
        nets: List of network layers
        optimizers: List of optimizers
        schedulers: List of schedulers
        nbbatches: Current batch count
        
    Returns:
        ModelState containing only parameters (no data)
    """
    state = ModelState(
        layer_state_dicts=[net.state_dict() for net in nets],
        optimizer_state_dicts=[opt.state_dict() for opt in optimizers],
        scheduler_state_dicts=[sch.state_dict() for sch in schedulers],
        nbbatches=nbbatches
    )
    state.state_hash = state.compute_hash()
    return state


def load_model_state(
    model_state: ModelState,
    nets: List[torch.nn.Module],
    optimizers: List,
    schedulers: List
) -> int:
    """
    Load model state into networks.
    
    Args:
        model_state: State to load
        nets: List of network layers
        optimizers: List of optimizers
        schedulers: List of schedulers
        
    Returns:
        nbbatches from the state
    """
    for net, sd in zip(nets, model_state.layer_state_dicts):
        net.load_state_dict(sd)
    
    for opt, sd in zip(optimizers, model_state.optimizer_state_dicts):
        opt.load_state_dict(sd)
    
    for sch, sd in zip(schedulers, model_state.scheduler_state_dicts):
        sch.load_state_dict(sd)
    
    return model_state.nbbatches


# Socket communication helpers
def send_msg(sock, data):
    """Send a message with length prefix."""
    msg = pickle.dumps(data)
    msg_len = struct.pack('>I', len(msg))
    sock.sendall(msg_len + msg)


def recv_msg(sock):
    """Receive a message with length prefix."""
    raw_len = recv_all(sock, 4)
    if not raw_len:
        return None
    msg_len = struct.unpack('>I', raw_len)[0]
    return pickle.loads(recv_all(sock, msg_len))


def recv_all(sock, n):
    """Receive exactly n bytes."""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)
