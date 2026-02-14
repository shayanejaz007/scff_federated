#!/usr/bin/env python3
"""
Data Partitioning Utility for Federated SCFF

Splits dataset deterministically and equally across N clients.
Strict privacy: each client can only access its own partition.
"""
import torch
from torch.utils.data import DataLoader, Subset
from typing import List, Tuple, Dict, Optional
import hashlib


class DataPartitioner:
    """
    Partitions a dataset across N clients with strict isolation.
    
    Privacy guarantees:
    - Each client receives disjoint indices
    - No client can access another client's data
    - Partitioning is deterministic given seed
    """
    
    def __init__(
        self,
        dataset_size: int,
        num_clients: int,
        seed: int = 1234
    ):
        """
        Initialize the partitioner.
        
        Args:
            dataset_size: Total number of samples (e.g., 10000)
            num_clients: Number of clients to partition across
            seed: Random seed for deterministic partitioning
        """
        self.dataset_size = dataset_size
        self.num_clients = num_clients
        self.seed = seed
        
        # Generate deterministic partition
        self._partition_indices = self._create_partitions()
        
        # Verify partitions are disjoint
        self._verify_disjoint()
        
    def _create_partitions(self) -> Dict[int, List[int]]:
        """Create disjoint partitions for each client."""
        torch.manual_seed(self.seed)
        
        # Shuffle all indices
        all_indices = torch.randperm(self.dataset_size).tolist()
        
        # Calculate partition sizes (as equal as possible)
        base_size = self.dataset_size // self.num_clients
        remainder = self.dataset_size % self.num_clients
        
        partitions = {}
        start_idx = 0
        
        for client_id in range(self.num_clients):
            # Some clients get one extra sample if there's remainder
            size = base_size + (1 if client_id < remainder else 0)
            end_idx = start_idx + size
            
            partitions[client_id] = all_indices[start_idx:end_idx]
            start_idx = end_idx
        
        return partitions
    
    def _verify_disjoint(self):
        """Verify that all partitions are disjoint."""
        all_indices = []
        for client_id, indices in self._partition_indices.items():
            all_indices.extend(indices)
        
        # Check no duplicates
        assert len(all_indices) == len(set(all_indices)), \
            "PRIVACY VIOLATION: Partitions are not disjoint!"
        
        # Check all indices covered
        assert len(all_indices) == self.dataset_size, \
            f"Partition error: {len(all_indices)} indices vs {self.dataset_size} dataset size"
        
        print(f"✓ Verified: {self.num_clients} disjoint partitions, {self.dataset_size} total samples")
    
    def get_client_indices(self, client_id: int) -> List[int]:
        """
        Get the indices assigned to a specific client.
        
        Args:
            client_id: Client identifier (0 to num_clients-1)
            
        Returns:
            List of dataset indices for this client
        """
        if client_id not in self._partition_indices:
            raise ValueError(f"Invalid client_id: {client_id}. Must be 0 to {self.num_clients-1}")
        
        return self._partition_indices[client_id].copy()  # Return copy for safety
    
    def get_partition_sizes(self) -> Dict[int, int]:
        """Get the size of each client's partition."""
        return {cid: len(indices) for cid, indices in self._partition_indices.items()}
    
    def get_partition_hash(self, client_id: int) -> str:
        """
        Get a hash of a client's partition for verification.
        Used to confirm partitions haven't been tampered with.
        """
        indices = self._partition_indices[client_id]
        indices_str = ','.join(map(str, sorted(indices)))
        return hashlib.sha256(indices_str.encode()).hexdigest()[:16]
    
    def verify_no_overlap(self, client_id_1: int, client_id_2: int) -> bool:
        """Verify two clients have no overlapping data."""
        indices_1 = set(self._partition_indices[client_id_1])
        indices_2 = set(self._partition_indices[client_id_2])
        overlap = indices_1.intersection(indices_2)
        
        if overlap:
            raise PrivacyViolationError(
                f"PRIVACY VIOLATION: Clients {client_id_1} and {client_id_2} "
                f"share {len(overlap)} samples!"
            )
        return True


class PrivacyViolationError(Exception):
    """Raised when a privacy constraint is violated."""
    pass


class ClientDataLoader:
    """
    Isolated data loader for a single client.
    Cannot access any data outside its partition.
    """
    
    def __init__(
        self,
        client_id: int,
        dataset,
        indices: List[int],
        batch_size: int,
        shuffle: bool = True
    ):
        self.client_id = client_id
        self._allowed_indices = set(indices)
        
        # Create subset - this is the ONLY data this client can access
        self.subset = Subset(dataset, indices)
        self.loader = DataLoader(
            self.subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0  # Avoid multiprocessing issues on Windows
        )
        
        self.num_samples = len(indices)
        self.batch_size = batch_size
        
    def __iter__(self):
        return iter(self.loader)
    
    def __len__(self):
        return len(self.loader)
    
    def verify_access(self, indices: List[int]) -> bool:
        """Verify client is only accessing its own data."""
        requested = set(indices)
        unauthorized = requested - self._allowed_indices
        
        if unauthorized:
            raise PrivacyViolationError(
                f"Client {self.client_id} attempted to access {len(unauthorized)} "
                f"unauthorized samples!"
            )
        return True


def get_client_loader(
    client_id: int,
    dataset,
    partitioner: DataPartitioner,
    batch_size: int,
    shuffle: bool = True
) -> ClientDataLoader:
    """
    Create an isolated data loader for a specific client.
    
    Args:
        client_id: Client identifier
        dataset: Full dataset (client will only see its partition)
        partitioner: DataPartitioner instance
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        
    Returns:
        ClientDataLoader with isolated access
    """
    indices = partitioner.get_client_indices(client_id)
    return ClientDataLoader(
        client_id=client_id,
        dataset=dataset,
        indices=indices,
        batch_size=batch_size,
        shuffle=shuffle
    )


def create_federated_loaders(
    train_dataset,
    test_dataset,
    sup_train_dataset,
    num_clients: int,
    batch_size: int,
    seed: int = 1234
) -> Tuple[DataPartitioner, List[ClientDataLoader], DataLoader, DataLoader]:
    """
    Create all data loaders for federated training.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset (shared for evaluation)
        sup_train_dataset: Supervised training dataset for evaluation
        num_clients: Number of clients
        batch_size: Batch size
        seed: Random seed
        
    Returns:
        Tuple of (partitioner, client_loaders, test_loader, sup_train_loader)
    """
    # Create partitioner
    partitioner = DataPartitioner(
        dataset_size=len(train_dataset),
        num_clients=num_clients,
        seed=seed
    )
    
    # Create client loaders
    client_loaders = []
    for client_id in range(num_clients):
        loader = get_client_loader(
            client_id=client_id,
            dataset=train_dataset,
            partitioner=partitioner,
            batch_size=batch_size,
            shuffle=True
        )
        client_loaders.append(loader)
        print(f"  Client {client_id}: {loader.num_samples} samples")
    
    # Test loader (shared - only used for evaluation, not training)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)
    
    # Supervised train loader for evaluation
    # Use full sup_train_dataset for fair evaluation comparison with baseline
    sup_train_loader = DataLoader(sup_train_dataset, batch_size=64, shuffle=True, num_workers=0)
    
    return partitioner, client_loaders, test_loader, sup_train_loader
