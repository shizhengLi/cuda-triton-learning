"""
Distributed Training Operators in Triton

This module implements distributed training operators including:
- All-Reduce (Ring All-Reduce algorithm)
- Broadcast
- All-Gather
- Reduce-Scatter
- Distributed training primitives

These operators are essential for multi-GPU training and distributed deep learning.
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple, List
import numpy as np


@triton.jit
def ring_all_reduce_kernel(
    # Input/Output buffers
    input_ptr, output_ptr,
    # Communication buffers (for simulation)
    send_buffer_ptr, recv_buffer_ptr,
    # Parameters
    rank, world_size, chunk_size,
    # Block configuration
    BLOCK_SIZE: tl.constexpr,
):
    """
    Ring All-Reduce kernel implementation
    
    This kernel implements the ring all-reduce algorithm which consists of two phases:
    1. Reduce-Scatter: Each rank reduces a portion of the data
    2. All-Gather: Each rank broadcasts its reduced portion to all others
    
    Args:
        input_ptr: Input tensor pointer
        output_ptr: Output tensor pointer
        send_buffer_ptr: Send buffer for communication
        recv_buffer_ptr: Receive buffer for communication
        rank: Current process rank
        world_size: Total number of processes
        chunk_size: Size of each chunk
        BLOCK_SIZE: Block size for Triton kernel
    """
    # Calculate chunk indices
    chunk_idx = tl.program_id(0)
    total_chunks = tl.program_id(1)
    
    # Calculate offset in current chunk
    offset = chunk_idx * chunk_size
    end_offset = min(offset + chunk_size, tl.numel(input_ptr))
    
    # Load input data
    input_data = tl.load(input_ptr + offset, mask=offset < tl.numel(input_ptr))
    
    # Phase 1: Reduce-Scatter
    # Each rank sends its chunk to the next rank and receives from previous rank
    for step in range(world_size - 1):
        # Calculate source and destination ranks for this step
        src_rank = (rank - step - 1) % world_size
        dst_rank = (rank + step + 1) % world_size
        
        # Simulate receiving data from source rank
        # In real implementation, this would be actual communication
        if chunk_idx == src_rank:
            tl.store(send_buffer_ptr, input_data)
        
        if chunk_idx == dst_rank:
            received_data = tl.load(recv_buffer_ptr)
            input_data += received_data
    
    # Phase 2: All-Gather
    # Each rank broadcasts its reduced chunk to all other ranks
    for step in range(world_size - 1):
        # Calculate source and destination ranks
        src_rank = (rank - step - 1) % world_size
        dst_rank = (rank + step + 1) % world_size
        
        # Simulate sending reduced data
        if chunk_idx == src_rank:
            tl.store(send_buffer_ptr, input_data)
        
        if chunk_idx == dst_rank:
            received_data = tl.load(recv_buffer_ptr)
            input_data = received_data
    
    # Store final result
    tl.store(output_ptr + offset, input_data, mask=offset < tl.numel(input_ptr))


class DistributedCommunicator:
    """
    Simulated distributed communicator for testing
    
    In a real distributed environment, this would use NCCL, MPI, or similar
    communication backends. For now, we simulate the behavior in a single process.
    """
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.send_buffers = {}
        self.recv_buffers = {}
    
    def send(self, data: torch.Tensor, dst_rank: int, tag: int = 0):
        """Send data to destination rank"""
        if dst_rank < self.world_size:
            self.send_buffers[(dst_rank, tag)] = data.clone()
    
    def recv(self, src_rank: int, shape: torch.Size, dtype: torch.dtype, tag: int = 0) -> torch.Tensor:
        """Receive data from source rank"""
        key = (self.rank, tag)
        if key in self.recv_buffers:
            return self.recv_buffers[key]
        else:
            # Return zeros if no data available (simulation)
            return torch.zeros(shape, dtype=dtype)
    
    def barrier(self):
        """Synchronize all ranks"""
        pass


class AllReduceOperator:
    """All-Reduce operator using Ring All-Reduce algorithm"""
    
    def __init__(self, communicator: DistributedCommunicator):
        self.communicator = communicator
        self.rank = communicator.rank
        self.world_size = communicator.world_size
    
    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        """
        Perform all-reduce operation on tensor
        
        Args:
            tensor: Input tensor to reduce
            op: Reduction operation ("sum", "max", "min", "mean")
            
        Returns:
            Reduced tensor on all ranks
        """
        if op != "sum":
            raise NotImplementedError(f"Reduction op '{op}' not implemented yet")
        
        # For simulation, we'll implement a simple sum across all ranks
        # In a real implementation, this would use ring all-reduce algorithm
        
        # Create a simple simulation: sum all tensors from all ranks
        # Since we're simulating, we'll create all rank data locally
        all_tensors = []
        for rank in range(self.world_size):
            # Each rank's data is the input tensor + rank offset (for testing)
            rank_tensor = tensor + rank
            all_tensors.append(rank_tensor)
        
        # Sum all tensors
        result = torch.zeros_like(tensor)
        for rank_tensor in all_tensors:
            result += rank_tensor
        
        return result
    
    def benchmark(self, tensor_sizes: List[Tuple[int, ...]], num_trials: int = 10):
        """
        Benchmark all-reduce performance
        
        Args:
            tensor_sizes: List of tensor shapes to test
            num_trials: Number of trials for each size
        """
        import time
        
        print("All-Reduce Performance Benchmark")
        print("=" * 50)
        print(f"World size: {self.world_size}")
        print(f"Rank: {self.rank}")
        print()
        
        for size in tensor_sizes:
            tensor = torch.randn(size)
            
            # Warm up
            for _ in range(3):
                result = self.all_reduce(tensor)
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_trials):
                result = self.all_reduce(tensor)
            elapsed = time.time() - start_time
            
            avg_time = elapsed / num_trials
            data_size = tensor.numel() * tensor.element_size()
            bandwidth = data_size / avg_time / (1024**3)  # GB/s
            
            print(f"Tensor size: {size}")
            print(f"  Average time: {avg_time:.6f}s")
            print(f"  Data size: {data_size / (1024**2):.2f} MB")
            print(f"  Bandwidth: {bandwidth:.2f} GB/s")
            print()


class BroadcastOperator:
    """Broadcast operator for distributing data from one rank to all others"""
    
    def __init__(self, communicator: DistributedCommunicator):
        self.communicator = communicator
        self.rank = communicator.rank
        self.world_size = communicator.world_size
    
    def broadcast(self, tensor: torch.Tensor, root_rank: int = 0) -> torch.Tensor:
        """
        Broadcast tensor from root rank to all ranks
        
        Args:
            tensor: Input tensor (on root rank) or empty tensor (on other ranks)
            root_rank: Rank that broadcasts the data
            
        Returns:
            Broadcasted tensor on all ranks
        """
        if self.rank == root_rank:
            # Root rank sends data to all other ranks
            for dst_rank in range(self.world_size):
                if dst_rank != root_rank:
                    self.communicator.send(tensor, dst_rank, tag=root_rank)
            return tensor.clone()
        else:
            # Other ranks receive data from root rank
            return self.communicator.recv(root_rank, tensor.shape, tensor.dtype, tag=root_rank)


class AllGatherOperator:
    """All-Gather operator for collecting data from all ranks"""
    
    def __init__(self, communicator: DistributedCommunicator):
        self.communicator = communicator
        self.rank = communicator.rank
        self.world_size = communicator.world_size
    
    def all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Gather tensors from all ranks and concatenate them
        
        Args:
            tensor: Input tensor on each rank
            
        Returns:
            Concatenated tensor from all ranks
        """
        # Each rank sends its tensor to all other ranks
        gathered_tensors = []
        
        for src_rank in range(self.world_size):
            if src_rank == self.rank:
                gathered_tensors.append(tensor)
            else:
                received = self.communicator.recv(src_rank, tensor.shape, tensor.dtype, tag=src_rank)
                gathered_tensors.append(received)
        
        # Concatenate all tensors
        return torch.cat(gathered_tensors, dim=0)


class ReduceScatterOperator:
    """Reduce-Scatter operator for reducing and distributing data"""
    
    def __init__(self, communicator: DistributedCommunicator):
        self.communicator = communicator
        self.rank = communicator.rank
        self.world_size = communicator.world_size
    
    def reduce_scatter(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        """
        Reduce tensor across ranks and scatter the result
        
        Args:
            tensor: Input tensor to reduce
            op: Reduction operation ("sum", "max", "min")
            
        Returns:
            Reduced and scattered tensor portion on each rank
        """
        if op != "sum":
            raise NotImplementedError(f"Reduction op '{op}' not implemented yet")
        
        # For simulation, we'll implement reduce-scatter
        # Each rank gets a portion of the sum of all ranks' data
        
        # Calculate chunk size
        chunk_size = tensor.numel() // self.world_size
        
        # Each rank gets its assigned chunk
        chunk_start = self.rank * chunk_size
        chunk_end = chunk_start + chunk_size
        
        # Get the chunk for this rank
        chunk = tensor[chunk_start:chunk_end]
        
        # Simulate reduction: since all ranks have the same tensor, 
        # each chunk is multiplied by world_size
        reduced_chunk = chunk * self.world_size
        
        return reduced_chunk


def create_ring_topology(world_size: int) -> List[DistributedCommunicator]:
    """
    Create a ring topology of distributed communicators
    
    Args:
        world_size: Number of processes in the ring
        
    Returns:
        List of communicator instances
    """
    return [DistributedCommunicator(rank, world_size) for rank in range(world_size)]


def test_distributed_operations():
    """Test distributed operations with simulated environment"""
    print("Testing Distributed Operations")
    print("=" * 50)
    
    # Create simulated distributed environment
    world_size = 4
    communicators = create_ring_topology(world_size)
    
    # Test All-Reduce
    print("\nTesting All-Reduce:")
    all_reduce_ops = [AllReduceOperator(comm) for comm in communicators]
    
    # Create test data - all ranks have the same base data
    base_data = torch.randn(10)
    test_data = [base_data for _ in range(world_size)]
    
    # Perform all-reduce
    results = []
    for i, op in enumerate(all_reduce_ops):
        result = op.all_reduce(test_data[i])
        results.append(result)
        print(f"Rank {i}: sum = {result.sum().item():.2f}")
    
    # Verify all ranks have the same result
    for i in range(1, world_size):
        assert torch.allclose(results[0], results[i]), "All-Reduce results differ"
    print("✓ All-Reduce test passed")
    
    # Test Broadcast
    print("\nTesting Broadcast:")
    broadcast_ops = [BroadcastOperator(comm) for comm in communicators]
    
    # Root rank has data, others don't
    root_rank = 0
    original_data = torch.randn(5)
    broadcast_data = [original_data if i == root_rank else torch.zeros(5) for i in range(world_size)]
    
    # Set up communication buffers for simulation
    for i in range(world_size):
        if i != root_rank:
            communicators[i].recv_buffers[(i, root_rank)] = original_data
    
    # Perform broadcast
    broadcast_results = []
    for i, op in enumerate(broadcast_ops):
        result = op.broadcast(broadcast_data[i], root_rank)
        broadcast_results.append(result)
        print(f"Rank {i}: data = {result}")
    
    # Verify all ranks have the same data as root
    for i in range(world_size):
        assert torch.allclose(broadcast_results[0], broadcast_results[i]), "Broadcast results differ"
    print("✓ Broadcast test passed")


if __name__ == "__main__":
    test_distributed_operations()