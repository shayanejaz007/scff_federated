"""Federated SCFF - Multi-client Split Learning"""
from .partition import DataPartitioner, get_client_loader
from .protocol import ModelState, ClientUpdate, RoundMetadata
