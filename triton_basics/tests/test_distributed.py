"""
Unit tests for distributed training operators
"""

import pytest
import torch
import numpy as np
from typing import List, Tuple
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '05_advanced', '03_distributed'))

from distributed_ops import (
    DistributedCommunicator, AllReduceOperator, BroadcastOperator, 
    AllGatherOperator, ReduceScatterOperator, create_ring_topology
)


class TestDistributedCommunicator:
    """Test suite for DistributedCommunicator"""
    
    def test_communicator_initialization(self):
        """Test communicator initialization"""
        comm = DistributedCommunicator(rank=0, world_size=4)
        assert comm.rank == 0
        assert comm.world_size == 4
        assert len(comm.send_buffers) == 0
        assert len(comm.recv_buffers) == 0
    
    def test_communicator_send_recv(self):
        """Test send and receive operations"""
        comm1 = DistributedCommunicator(rank=0, world_size=2)
        comm2 = DistributedCommunicator(rank=1, world_size=2)
        
        # Send data
        data = torch.tensor([1.0, 2.0, 3.0])
        comm1.send(data, dst_rank=1, tag=0)
        
        # Simulate receiving (in real implementation, this would be async)
        comm2.recv_buffers[(1, 0)] = data
        
        # Receive data
        received = comm2.recv(src_rank=0, shape=data.shape, dtype=data.dtype, tag=0)
        assert torch.allclose(data, received)
    
    def test_communicator_barrier(self):
        """Test barrier operation"""
        comm = DistributedCommunicator(rank=0, world_size=2)
        comm.barrier()  # Should not raise an exception


class TestAllReduceOperator:
    """Test suite for AllReduceOperator"""
    
    def test_all_reduce_operator_initialization(self):
        """Test AllReduceOperator initialization"""
        comm = DistributedCommunicator(rank=0, world_size=4)
        op = AllReduceOperator(comm)
        assert op.communicator == comm
        assert op.rank == 0
        assert op.world_size == 4
    
    def test_all_reduce_sum_operation(self):
        """Test all-reduce sum operation"""
        world_size = 4
        communicators = create_ring_topology(world_size)
        operators = [AllReduceOperator(comm) for comm in communicators]
        
        # Create test data: each rank has the same base tensor
        # The simulation will add rank offset internally
        base_tensor = torch.full((10,), 1.0)  # Base tensor for all ranks
        
        # Perform all-reduce
        results = []
        for i, op in enumerate(operators):
            result = op.all_reduce(base_tensor)
            results.append(result)
        
        # Expected result: base_tensor * 4 ranks + sum(0,1,2,3) = 4 + 6 = 10
        expected = torch.full((10,), 10.0)
        
        # All ranks should have the same result
        for result in results:
            assert torch.allclose(result, expected), f"All-reduce result mismatch: {result} vs {expected}"
    
    def test_all_reduce_different_sizes(self):
        """Test all-reduce with different tensor sizes"""
        world_size = 2
        communicators = create_ring_topology(world_size)
        operators = [AllReduceOperator(comm) for comm in communicators]
        
        # Test with different tensor sizes
        test_sizes = [(5,), (10,), (3, 3), (2, 4, 5)]
        
        for size in test_sizes:
            base_tensor = torch.randn(size)
            
            # Perform all-reduce
            results = []
            for i, op in enumerate(operators):
                result = op.all_reduce(base_tensor)
                results.append(result)
            
            # Expected result: base_tensor * 2 ranks + sum(0,1) = 2*base_tensor + 1
            expected = base_tensor * 2 + 1
            
            # All ranks should have the same result
            for result in results:
                assert torch.allclose(result, expected, atol=1e-6), f"All-reduce result mismatch for size {size}"
    
    def test_all_reduce_unsupported_operation(self):
        """Test all-reduce with unsupported operations"""
        comm = DistributedCommunicator(rank=0, world_size=2)
        op = AllReduceOperator(comm)
        tensor = torch.randn(5)
        
        with pytest.raises(NotImplementedError):
            op.all_reduce(tensor, op="max")
        
        with pytest.raises(NotImplementedError):
            op.all_reduce(tensor, op="min")


class TestBroadcastOperator:
    """Test suite for BroadcastOperator"""
    
    def test_broadcast_operator_initialization(self):
        """Test BroadcastOperator initialization"""
        comm = DistributedCommunicator(rank=0, world_size=4)
        op = BroadcastOperator(comm)
        assert op.communicator == comm
        assert op.rank == 0
        assert op.world_size == 4
    
    def test_broadcast_from_root(self):
        """Test broadcast operation from root rank"""
        world_size = 4
        communicators = create_ring_topology(world_size)
        operators = [BroadcastOperator(comm) for comm in communicators]
        
        # Root rank has data, others have zeros
        root_rank = 0
        original_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        
        broadcast_data = []
        for i in range(world_size):
            if i == root_rank:
                broadcast_data.append(original_data)
            else:
                broadcast_data.append(torch.zeros(5))
        
        # Set up communication buffers for simulation
        for i in range(world_size):
            if i != root_rank:
                communicators[i].recv_buffers[(i, root_rank)] = original_data
        
        # Perform broadcast
        results = []
        for i, op in enumerate(operators):
            result = op.broadcast(broadcast_data[i], root_rank)
            results.append(result)
        
        # All ranks should have the same data as root
        for result in results:
            assert torch.allclose(result, original_data), f"Broadcast result mismatch: {result} vs {original_data}"
    
    def test_broadcast_from_different_root(self):
        """Test broadcast operation from different root ranks"""
        world_size = 3
        communicators = create_ring_topology(world_size)
        operators = [BroadcastOperator(comm) for comm in communicators]
        
        # Test broadcasting from each rank
        for root_rank in range(world_size):
            original_data = torch.tensor([float(root_rank + 1)] * 5)
            
            broadcast_data = []
            for i in range(world_size):
                if i == root_rank:
                    broadcast_data.append(original_data)
                else:
                    broadcast_data.append(torch.zeros(5))
                
                # Set up communication buffers
                if i != root_rank:
                    communicators[i].recv_buffers[(i, root_rank)] = original_data
            
            # Perform broadcast
            results = []
            for i, op in enumerate(operators):
                result = op.broadcast(broadcast_data[i], root_rank)
                results.append(result)
            
            # All ranks should have the same data as root
            for result in results:
                assert torch.allclose(result, original_data), f"Broadcast from rank {root_rank} failed"


class TestAllGatherOperator:
    """Test suite for AllGatherOperator"""
    
    def test_all_gather_operator_initialization(self):
        """Test AllGatherOperator initialization"""
        comm = DistributedCommunicator(rank=0, world_size=4)
        op = AllGatherOperator(comm)
        assert op.communicator == comm
        assert op.rank == 0
        assert op.world_size == 4
    
    def test_all_gather_operation(self):
        """Test all-gather operation"""
        world_size = 3
        communicators = create_ring_topology(world_size)
        operators = [AllGatherOperator(comm) for comm in communicators]
        
        # Each rank has different data
        rank_data = [torch.tensor([i + 1]) for i in range(world_size)]
        
        # Set up communication buffers for simulation
        for i in range(world_size):
            for j in range(world_size):
                if i != j:
                    communicators[j].recv_buffers[(j, i)] = rank_data[i]
        
        # Perform all-gather
        results = []
        for i, op in enumerate(operators):
            result = op.all_gather(rank_data[i])
            results.append(result)
        
        # Expected result: concatenation of all rank data
        expected = torch.cat([torch.tensor([1]), torch.tensor([2]), torch.tensor([3])])
        
        # All ranks should have the same concatenated result
        for result in results:
            assert torch.allclose(result, expected), f"All-gather result mismatch: {result} vs {expected}"
    
    def test_all_gather_different_shapes(self):
        """Test all-gather with different tensor shapes"""
        world_size = 2
        communicators = create_ring_topology(world_size)
        operators = [AllGatherOperator(comm) for comm in communicators]
        
        # Test with different tensor shapes
        test_shapes = [(3,), (2, 2), (4,)]
        
        for shape in test_shapes:
            rank_data = [torch.randn(shape) + i for i in range(world_size)]
            
            # Set up communication buffers
            for i in range(world_size):
                for j in range(world_size):
                    if i != j:
                        communicators[j].recv_buffers[(j, i)] = rank_data[i]
            
            # Perform all-gather
            results = []
            for i, op in enumerate(operators):
                result = op.all_gather(rank_data[i])
                results.append(result)
            
            # Expected result: concatenation of all rank data
            expected = torch.cat(rank_data, dim=0)
            
            # All ranks should have the same result
            for result in results:
                assert torch.allclose(result, expected, atol=1e-6), f"All-gather failed for shape {shape}"


class TestReduceScatterOperator:
    """Test suite for ReduceScatterOperator"""
    
    def test_reduce_scatter_operator_initialization(self):
        """Test ReduceScatterOperator initialization"""
        comm = DistributedCommunicator(rank=0, world_size=4)
        op = ReduceScatterOperator(comm)
        assert op.communicator == comm
        assert op.rank == 0
        assert op.world_size == 4
    
    def test_reduce_scatter_operation(self):
        """Test reduce-scatter operation"""
        world_size = 2
        communicators = create_ring_topology(world_size)
        operators = [ReduceScatterOperator(comm) for comm in communicators]
        
        # Each rank has the same full tensor
        full_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        # Perform reduce-scatter
        results = []
        for i, op in enumerate(operators):
            result = op.reduce_scatter(full_tensor)
            results.append(result)
        
        # Expected results: each rank gets a portion of the sum
        # Since all ranks have the same data, each portion is multiplied by world_size
        expected_rank0 = torch.tensor([2.0, 4.0])  # First half * 2
        expected_rank1 = torch.tensor([6.0, 8.0])  # Second half * 2
        
        assert torch.allclose(results[0], expected_rank0), f"Reduce-scatter rank 0 failed: {results[0]}"
        assert torch.allclose(results[1], expected_rank1), f"Reduce-scatter rank 1 failed: {results[1]}"
    
    def test_reduce_scatter_unsupported_operation(self):
        """Test reduce-scatter with unsupported operations"""
        comm = DistributedCommunicator(rank=0, world_size=2)
        op = ReduceScatterOperator(comm)
        tensor = torch.randn(8)
        
        with pytest.raises(NotImplementedError):
            op.reduce_scatter(tensor, op="max")


class TestCreateRingTopology:
    """Test suite for ring topology creation"""
    
    def test_create_ring_topology(self):
        """Test ring topology creation"""
        world_size = 4
        communicators = create_ring_topology(world_size)
        
        assert len(communicators) == world_size
        
        for i, comm in enumerate(communicators):
            assert comm.rank == i
            assert comm.world_size == world_size


class TestDistributedIntegration:
    """Integration tests for distributed operations"""
    
    def test_distributed_workflow(self):
        """Test complete distributed workflow"""
        world_size = 3
        communicators = create_ring_topology(world_size)
        
        # Create operators
        all_reduce_ops = [AllReduceOperator(comm) for comm in communicators]
        broadcast_ops = [BroadcastOperator(comm) for comm in communicators]
        all_gather_ops = [AllGatherOperator(comm) for comm in communicators]
        
        # Step 1: All-reduce gradients
        base_gradient = torch.randn(5)
        
        # Set up communication for all-reduce
        reduced_grads = []
        for i, op in enumerate(all_reduce_ops):
            result = op.all_reduce(base_gradient)
            reduced_grads.append(result)
        
        # Verify all-reduce worked: base_gradient * 3 + sum(0,1,2) = 3*base_gradient + 3
        expected_sum = base_gradient * 3 + 3
        for result in reduced_grads:
            assert torch.allclose(result, expected_sum, atol=1e-6), "All-reduce failed in workflow"
        
        # Step 2: Broadcast model parameters
        params = torch.tensor([1.0, 2.0, 3.0])
        broadcast_data = [params if i == 0 else torch.zeros(3) for i in range(world_size)]
        
        # Set up communication for broadcast
        for i in range(world_size):
            if i != 0:
                communicators[i].recv_buffers[(i, 0)] = params
        
        # Perform broadcast
        broadcast_results = []
        for i, op in enumerate(broadcast_ops):
            result = op.broadcast(broadcast_data[i], 0)
            broadcast_results.append(result)
        
        # Verify broadcast worked
        for result in broadcast_results:
            assert torch.allclose(result, params), "Broadcast failed in workflow"
        
        print("✓ Distributed workflow test passed")
    
    def test_performance_consistency(self):
        """Test performance consistency across operations"""
        world_size = 2
        communicators = create_ring_topology(world_size)
        
        all_reduce_op = AllReduceOperator(communicators[0])
        
        # Test with multiple tensor sizes
        sizes = [100, 1000, 10000]
        times = []
        
        import time
        for size in sizes:
            tensor = torch.randn(size)
            
            # Warm up
            for _ in range(3):
                result = all_reduce_op.all_reduce(tensor)
            
            # Measure time
            start_time = time.time()
            for _ in range(10):
                result = all_reduce_op.all_reduce(tensor)
            elapsed = time.time() - start_time
            
            times.append(elapsed / 10)
            print(f"Size {size}: {times[-1]:.6f}s")
        
        # Performance should scale reasonably with size
        assert times[1] > times[0], "Performance scaling issue"
        assert times[2] > times[1], "Performance scaling issue"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])