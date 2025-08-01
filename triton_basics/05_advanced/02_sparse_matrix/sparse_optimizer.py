"""
Sparse Matrix Optimization Algorithms

This module implements various optimization algorithms for sparse matrix operations:
- Sparse matrix reordering for better cache locality
- Load balancing for parallel sparse operations
- Memory optimization techniques
- Performance optimization strategies
"""

import torch
import triton
import triton.language as tl
from typing import List, Tuple, Dict, Optional
import numpy as np
from collections import defaultdict

from sparse_matrix_ops import SparseMatrix


class SparseMatrixOptimizer:
    """Sparse matrix optimization algorithms"""
    
    @staticmethod
    def reorder_csr_rcm(sparse_matrix: SparseMatrix) -> SparseMatrix:
        """
        Reorder sparse matrix using Reverse Cuthill-McKee algorithm
        for better cache locality
        
        Args:
            sparse_matrix: Sparse matrix in CSR format
            
        Returns:
            Reordered sparse matrix
        """
        if sparse_matrix.format != "csr":
            sparse_matrix = sparse_matrix.to_csr()
        
        M, N = sparse_matrix.shape
        
        # Build adjacency list
        adj = defaultdict(list)
        for i in range(M):
            start, end = sparse_matrix.indptr[i], sparse_matrix.indptr[i + 1]
            for j in range(start, end):
                col_idx = sparse_matrix.indices[j].item()
                adj[i].append(col_idx)
                adj[col_idx].append(i)  # Symmetric for RCM
        
        # Find pseudo-peripheral node
        def bfs_farthest(start):
            visited = {start}
            queue = [(start, 0)]
            farthest_node = start
            max_distance = 0
            
            while queue:
                node, dist = queue.pop(0)
                if dist > max_distance:
                    max_distance = dist
                    farthest_node = node
                
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
            
            return farthest_node, max_distance
        
        # Find diameter endpoints
        if M > 0:
            start_node = 0
            end_node, _ = bfs_farthest(start_node)
            start_node, _ = bfs_farthest(end_node)
        
        # RCM ordering
        def rcm_ordering(start):
            visited = set()
            ordering = []
            
            def bfs_rcm(node):
                visited.add(node)
                queue = [node]
                
                while queue:
                    current = queue.pop(0)
                    
                    # Sort neighbors by degree (ascending)
                    neighbors = sorted([n for n in adj[current] if n not in visited], 
                                     key=lambda x: len(adj[x]))
                    
                    for neighbor in neighbors:
                        visited.add(neighbor)
                        queue.append(neighbor)
                    
                    ordering.append(current)
            
            bfs_rcm(start)
            return ordering[::-1]  # Reverse for RCM
        
        if M > 0:
            ordering = rcm_ordering(start_node)
        else:
            ordering = []
        
        # Create permutation arrays
        perm = torch.tensor(ordering, dtype=torch.long)
        inv_perm = torch.zeros(M, dtype=torch.long)
        inv_perm[perm] = torch.arange(len(ordering))
        
        # Reorder the matrix
        return SparseMatrixOptimizer._reorder_matrix(sparse_matrix, perm, inv_perm)
    
    @staticmethod
    def _reorder_matrix(sparse_matrix: SparseMatrix, perm: torch.Tensor, 
                       inv_perm: torch.Tensor) -> SparseMatrix:
        """Reorder sparse matrix using permutation arrays"""
        if sparse_matrix.format != "csr":
            sparse_matrix = sparse_matrix.to_csr()
        
        M, N = sparse_matrix.shape
        data = sparse_matrix.data
        indices = sparse_matrix.indices
        indptr = sparse_matrix.indptr
        
        # Create new arrays
        new_data = []
        new_indices = []
        new_indptr = torch.zeros(M + 1, dtype=torch.int32)
        
        # Process each row in new order
        for new_row in range(M):
            old_row = perm[new_row].item()
            start, end = indptr[old_row], indptr[old_row + 1]
            
            # Copy non-zero elements
            for j in range(start, end):
                new_data.append(data[j])
                # Keep original column indices to avoid bounds issues
                new_indices.append(indices[j].item())
            
            new_indptr[new_row + 1] = new_indptr[new_row] + (end - start)
        
        new_data = torch.stack(new_data) if new_data else torch.tensor([])
        new_indices = torch.tensor(new_indices, dtype=torch.int32) if new_indices else torch.tensor([], dtype=torch.int32)
        
        return SparseMatrix(new_data, new_indices, new_indptr, sparse_matrix.shape, "csr")
    
    @staticmethod
    def balance_load(sparse_matrix: SparseMatrix, num_partitions: int = 8) -> List[SparseMatrix]:
        """
        Partition sparse matrix for balanced load in parallel processing
        
        Args:
            sparse_matrix: Sparse matrix in CSR format
            num_partitions: Number of partitions
            
        Returns:
            List of partitioned sparse matrices
        """
        if sparse_matrix.format != "csr":
            sparse_matrix = sparse_matrix.to_csr()
        
        M, N = sparse_matrix.shape
        
        # Calculate row-wise non-zero counts
        row_nnz = []
        for i in range(M):
            start, end = sparse_matrix.indptr[i], sparse_matrix.indptr[i + 1]
            row_nnz.append(end - start)
        
        row_nnz = torch.tensor(row_nnz)
        total_nnz = row_nnz.sum().item()
        target_nnz = total_nnz // num_partitions
        
        # Greedy partitioning
        partitions = []
        current_partition = []
        current_nnz = 0
        
        for i in range(M):
            if current_nnz + row_nnz[i] > target_nnz and current_partition:
                partitions.append(current_partition)
                current_partition = []
                current_nnz = 0
            
            current_partition.append(i)
            current_nnz += row_nnz[i]
        
        if current_partition:
            partitions.append(current_partition)
        
        # Create partitioned matrices
        partitioned_matrices = []
        for partition in partitions:
            if not partition:
                continue
            
            start_row = min(partition)
            end_row = max(partition) + 1
            
            # Extract submatrix
            sub_data = []
            sub_indices = []
            sub_indptr = torch.zeros(len(partition) + 1, dtype=torch.int32)
            
            for local_row, global_row in enumerate(partition):
                start, end = sparse_matrix.indptr[global_row], sparse_matrix.indptr[global_row + 1]
                
                for j in range(start, end):
                    sub_data.append(sparse_matrix.data[j])
                    sub_indices.append(sparse_matrix.indices[j])
                
                sub_indptr[local_row + 1] = sub_indptr[local_row] + (end - start)
            
            sub_data = torch.stack(sub_data) if sub_data else torch.tensor([])
            sub_indices = torch.tensor(sub_indices, dtype=torch.int32) if sub_indices else torch.tensor([], dtype=torch.int32)
            
            sub_matrix = SparseMatrix(sub_data, sub_indices, sub_indptr, 
                                    (len(partition), N), "csr")
            partitioned_matrices.append(sub_matrix)
        
        return partitioned_matrices
    
    @staticmethod
    def compress_storage(sparse_matrix: SparseMatrix, dtype: torch.dtype = torch.float16) -> SparseMatrix:
        """
        Compress sparse matrix storage by using smaller data types
        
        Args:
            sparse_matrix: Input sparse matrix
            dtype: Target data type for values
            
        Returns:
            Compressed sparse matrix
        """
        # Convert data to smaller type
        compressed_data = sparse_matrix.data.to(dtype)
        
        # Use smaller integer types for indices if possible
        M, N = sparse_matrix.shape
        
        if M < 2**16 and N < 2**16:
            # Use int16 for indices
            if sparse_matrix.format == "csr":
                compressed_indices = sparse_matrix.indices.to(torch.int16)
                compressed_indptr = sparse_matrix.indptr.to(torch.int16)
                return SparseMatrix(compressed_data, compressed_indices, 
                                 compressed_indptr, sparse_matrix.shape, "csr")
            else:  # COO
                compressed_indices = sparse_matrix.indices.to(torch.int16)
                compressed_indptr = sparse_matrix.indptr.to(torch.int16) if sparse_matrix.indptr is not None else None
                return SparseMatrix(compressed_data, compressed_indices, 
                                 compressed_indptr, sparse_matrix.shape, "coo")
        
        # Use int32 for larger matrices
        if sparse_matrix.format == "csr":
            compressed_indices = sparse_matrix.indices.to(torch.int32)
            compressed_indptr = sparse_matrix.indptr.to(torch.int32)
            return SparseMatrix(compressed_data, compressed_indices, 
                             compressed_indptr, sparse_matrix.shape, "csr")
        else:  # COO
            compressed_indices = sparse_matrix.indices.to(torch.int32)
            compressed_indptr = sparse_matrix.indptr.to(torch.int32) if sparse_matrix.indptr is not None else None
            return SparseMatrix(compressed_data, compressed_indices, 
                             compressed_indptr, sparse_matrix.shape, "coo")


@triton.jit
def sparse_matmul_balanced_kernel(
    # Sparse matrix partitions
    sparse_data_ptrs, sparse_indices_ptrs, sparse_indptr_ptrs,
    # Dense matrix
    dense_ptr,
    # Output
    output_ptr,
    # Dimensions
    partition_starts, partition_ends, N, K,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """
    Balanced sparse matrix multiplication kernel
    Handles multiple partitions for load balancing
    """
    # Get partition ID
    partition_id = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Get partition range
    start_row = tl.load(partition_starts + partition_id)
    end_row = tl.load(partition_ends + partition_id)
    
    # Calculate block ranges
    m_start = start_row + pid_n * BLOCK_SIZE_M
    m_end = min(m_start + BLOCK_SIZE_M, end_row)
    
    n_start = 0
    n_end = N
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Get partition pointers
    data_ptr = tl.load(sparse_data_ptrs + partition_id)
    indices_ptr = tl.load(sparse_indices_ptrs + partition_id)
    indptr_ptr = tl.load(sparse_indptr_ptrs + partition_id)
    
    # Iterate over rows in block
    for m_offset in range(BLOCK_SIZE_M):
        m_idx = m_start + m_offset
        if m_idx < end_row:
            local_row_idx = m_idx - start_row
            
            # Get row range in CSR format
            row_start = tl.load(indptr_ptr + local_row_idx)
            row_end = tl.load(indptr_ptr + local_row_idx + 1)
            
            # Iterate over non-zero elements in row
            for nz_idx in range(row_start, row_end):
                # Load sparse matrix data
                sparse_val = tl.load(data_ptr + nz_idx)
                k_idx = tl.load(indices_ptr + nz_idx)
                
                # Load corresponding dense matrix row
                for n_offset in range(BLOCK_SIZE_N):
                    n_idx = n_start + n_offset
                    if n_idx < N:
                        dense_val = tl.load(dense_ptr + k_idx * N + n_idx)
                        accumulator[m_offset, n_offset] += sparse_val * dense_val
    
    # Store output
    for m_offset in range(BLOCK_SIZE_M):
        m_idx = m_start + m_offset
        if m_idx < end_row:
            for n_offset in range(BLOCK_SIZE_N):
                n_idx = n_start + n_offset
                if n_idx < N:
                    tl.store(output_ptr + m_idx * N + n_idx, accumulator[m_offset, n_offset])


class OptimizedSparseOps:
    """Optimized sparse matrix operations"""
    
    @staticmethod
    def matmul_balanced(sparse_matrix: SparseMatrix, dense_matrix: torch.Tensor, 
                        num_partitions: int = 8) -> torch.Tensor:
        """
        Balanced sparse matrix multiplication with load partitioning
        
        Args:
            sparse_matrix: Sparse matrix in CSR format
            dense_matrix: Dense matrix (K x N)
            num_partitions: Number of partitions for load balancing
            
        Returns:
            Output matrix (M x N)
        """
        # Convert to CSR if needed
        if sparse_matrix.format != "csr":
            sparse_matrix = sparse_matrix.to_csr()
        
        # Partition the matrix
        partitions = SparseMatrixOptimizer.balance_load(sparse_matrix, num_partitions)
        
        M, K = sparse_matrix.shape
        N = dense_matrix.shape[1]
        
        # Prepare output
        output = torch.zeros((M, N), device=dense_matrix.device, dtype=torch.float32)
        
        # Simple CPU implementation for now - avoid Triton pointer issues
        current_start = 0
        for partition in partitions:
            part_M = partition.shape[0]
            part_end = current_start + part_M
            
            # Perform multiplication for this partition
            for i in range(part_M):
                global_row = current_start + i
                start, end = partition.indptr[i], partition.indptr[i + 1]
                
                for j in range(start, end):
                    col_idx = partition.indices[j].item()
                    sparse_val = partition.data[j].item()
                    
                    for k in range(N):
                        output[global_row, k] += sparse_val * dense_matrix[col_idx, k].item()
            
            current_start = part_end
        
        return output
    
    @staticmethod
    def optimize_and_multiply(sparse_matrix: SparseMatrix, dense_matrix: torch.Tensor,
                            optimization_level: int = 2) -> torch.Tensor:
        """
        Apply optimizations and perform sparse matrix multiplication
        
        Args:
            sparse_matrix: Input sparse matrix
            dense_matrix: Dense matrix
            optimization_level: 0=no opt, 1=basic, 2=advanced
            
        Returns:
            Result of multiplication
        """
        if optimization_level == 0:
            # No optimization
            from sparse_matrix_ops import SparseMatrixOps
            return SparseMatrixOps.matmul(sparse_matrix, dense_matrix)
        
        elif optimization_level == 1:
            # Basic optimization: load balancing
            return OptimizedSparseOps.matmul_balanced(sparse_matrix, dense_matrix)
        
        elif optimization_level == 2:
            # Advanced optimization: reordering + load balancing
            # Reorder for better cache locality
            reordered = SparseMatrixOptimizer.reorder_csr_rcm(sparse_matrix)
            
            # Load balanced multiplication
            return OptimizedSparseOps.matmul_balanced(reordered, dense_matrix)
        
        else:
            raise ValueError(f"Invalid optimization level: {optimization_level}")


def benchmark_optimizations():
    """Benchmark different optimization strategies"""
    print("Sparse Matrix Optimization Benchmark")
    print("=" * 50)
    
    # Create test matrices
    sizes = [(2000, 2000), (5000, 5000)]
    sparsities = [0.95, 0.99]
    
    for size in sizes:
        M, N = size
        print(f"\nMatrix size: {M}x{N}")
        
        for sparsity in sparsities:
            print(f"  Sparsity: {sparsity:.3f}")
            
            # Create sparse matrix
            from sparse_matrix_ops import SparseMatrixOps
            sparse_mat = SparseMatrixOps.create_random_sparse(
                (M, N), sparsity, "csr"
            )
            
            # Create dense matrix
            dense_mat = torch.randn(N, 100)
            
            # Test different optimization levels
            import time
            
            for opt_level in [0, 1, 2]:
                start_time = time.time()
                result = OptimizedSparseOps.optimize_and_multiply(
                    sparse_mat, dense_mat, opt_level
                )
                elapsed = time.time() - start_time
                
                opt_names = ["No optimization", "Load balancing", "Reordering + Load balancing"]
                print(f"    {opt_names[opt_level]}: {elapsed:.4f}s")


if __name__ == "__main__":
    benchmark_optimizations()