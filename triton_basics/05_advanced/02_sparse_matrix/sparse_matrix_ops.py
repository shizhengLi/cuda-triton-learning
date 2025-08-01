"""
Sparse Matrix Operations in Triton

This module implements sparse matrix operations including:
- Sparse matrix multiplication (SpMM)
- Sparse matrix optimization algorithms
- Efficient memory management for sparse operations
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple, List
import numpy as np


@triton.jit
def sparse_dense_matmul_kernel(
    # Sparse matrix (CSR format)
    sparse_data_ptr, sparse_indices_ptr, sparse_indptr_ptr,
    # Dense matrix
    dense_ptr,
    # Output
    output_ptr,
    # Dimensions
    M, N, K, nnz,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """
    Sparse matrix multiplication kernel (CSR format)
    Computes: output = sparse_matrix @ dense_matrix
    """
    # Get program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate block ranges
    m_start = pid_m * BLOCK_SIZE_M
    m_end = min(m_start + BLOCK_SIZE_M, M)
    
    n_start = pid_n * BLOCK_SIZE_N
    n_end = min(n_start + BLOCK_SIZE_N, N)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over rows in block
    for m_offset in range(BLOCK_SIZE_M):
        m_idx = m_start + m_offset
        if m_idx >= M:
            break
            
        # Get row range in CSR format
        row_start = tl.load(sparse_indptr_ptr + m_idx)
        row_end = tl.load(sparse_indptr_ptr + m_idx + 1)
        
        # Iterate over non-zero elements in row
        for nz_idx in range(row_start, row_end):
            if nz_idx >= nnz:
                break
                
            # Load sparse matrix data
            sparse_val = tl.load(sparse_data_ptr + nz_idx)
            k_idx = tl.load(sparse_indices_ptr + nz_idx)
            
            # Load corresponding dense matrix row
            for n_offset in range(BLOCK_SIZE_N):
                n_idx = n_start + n_offset
                if n_idx < N:
                    dense_val = tl.load(dense_ptr + k_idx * N + n_idx)
                    accumulator[m_offset, n_offset] += sparse_val * dense_val
    
    # Store output
    for m_offset in range(BLOCK_SIZE_M):
        m_idx = m_start + m_offset
        if m_idx < M:
            for n_offset in range(BLOCK_SIZE_N):
                n_idx = n_start + n_offset
                if n_idx < N:
                    tl.store(output_ptr + m_idx * N + n_idx, accumulator[m_offset, n_offset])


@triton.jit
def sparse_vector_matmul_kernel(
    # Sparse matrix (COO format)
    sparse_rows_ptr, sparse_cols_ptr, sparse_data_ptr,
    # Dense vector
    dense_ptr,
    # Output
    output_ptr,
    # Dimensions
    M, N, nnz,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Sparse matrix-vector multiplication kernel (COO format)
    Computes: output = sparse_matrix @ dense_vector
    """
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block range
    start_idx = pid * BLOCK_SIZE
    end_idx = min(start_idx + BLOCK_SIZE, nnz)
    
    # Initialize local accumulator
    local_accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Iterate over non-zero elements in block
    for offset in range(BLOCK_SIZE):
        nz_idx = start_idx + offset
        if nz_idx >= nnz:
            break
            
        # Load sparse matrix element
        row_idx = tl.load(sparse_rows_ptr + nz_idx)
        col_idx = tl.load(sparse_cols_ptr + nz_idx)
        sparse_val = tl.load(sparse_data_ptr + nz_idx)
        
        # Load dense vector element
        dense_val = tl.load(dense_ptr + col_idx)
        
        # Accumulate
        local_accumulator[offset] = sparse_val * dense_val
        
        # Atomic add to output
        tl.atomic_add(output_ptr + row_idx, local_accumulator[offset])


class SparseMatrix:
    """Sparse matrix container with multiple format support"""
    
    def __init__(self, data: torch.Tensor, indices: torch.Tensor, 
                 indptr: Optional[torch.Tensor] = None, 
                 shape: Optional[Tuple[int, int]] = None,
                 format: str = "csr"):
        """
        Initialize sparse matrix
        
        Args:
            data: Non-zero values
            indices: Column indices (CSR) or row indices (COO)
            indptr: Row pointers (CSR format only)
            shape: Matrix shape (M, N)
            format: Storage format ("csr", "coo")
        """
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.format = format
        self.shape = shape
        
        if format == "csr" and indptr is None:
            raise ValueError("indptr is required for CSR format")
        
        if shape is None:
            self._infer_shape()
    
    def _infer_shape(self):
        """Infer matrix shape from data"""
        if self.format == "csr":
            M = len(self.indptr) - 1
            N = self.indices.max().item() + 1 if len(self.indices) > 0 else 0
        else:  # COO
            M = self.indices.max().item() + 1 if len(self.indices) > 0 else 0
            N = self.indptr.max().item() + 1 if self.indptr is not None and len(self.indptr) > 0 else 0
        
        self.shape = (M, N)
    
    @property
    def nnz(self) -> int:
        """Number of non-zero elements"""
        return len(self.data)
    
    @property
    def sparsity(self) -> float:
        """Sparsity ratio"""
        total_elements = self.shape[0] * self.shape[1]
        return 1.0 - (self.nnz / total_elements)
    
    def to_csr(self) -> 'SparseMatrix':
        """Convert to CSR format"""
        if self.format == "csr":
            return self
        
        # Convert COO to CSR
        if self.format == "coo":
            rows = self.indices
            cols = self.indptr
            data = self.data
            
            M = self.shape[0]
            indptr = torch.zeros(M + 1, dtype=torch.int32, device=data.device)
            
            # Count non-zeros per row
            for row in rows:
                indptr[row + 1] += 1
            
            # Cumulative sum
            for i in range(1, M + 1):
                indptr[i] += indptr[i - 1]
            
            return SparseMatrix(data, cols, indptr, self.shape, "csr")
        
        raise ValueError(f"Cannot convert {self.format} to CSR")
    
    def to_coo(self) -> 'SparseMatrix':
        """Convert to COO format"""
        if self.format == "coo":
            return self
        
        # Convert CSR to COO
        if self.format == "csr":
            indptr = self.indptr
            indices = self.indices
            data = self.data
            
            rows = torch.zeros(self.nnz, dtype=torch.int32, device=data.device)
            cols = indices
            
            # Fill row indices
            for i in range(self.shape[0]):
                start, end = indptr[i], indptr[i + 1]
                rows[start:end] = i
            
            return SparseMatrix(data, rows, cols, self.shape, "coo")
        
        raise ValueError(f"Cannot convert {self.format} to COO")
    
    def to_dense(self) -> torch.Tensor:
        """Convert to dense matrix"""
        dense = torch.zeros(self.shape, device=self.data.device, dtype=self.data.dtype)
        
        if self.format == "csr":
            for i in range(self.shape[0]):
                start, end = self.indptr[i], self.indptr[i + 1]
                for j in range(start, end):
                    col_idx = self.indices[j].item()
                    dense[i, col_idx] = self.data[j]
        elif self.format == "coo":
            for i in range(self.nnz):
                row_idx = self.indices[i].item()
                col_idx = self.indptr[i].item()
                dense[row_idx, col_idx] = self.data[i]
        
        return dense


class SparseMatrixOps:
    """Sparse matrix operations"""
    
    @staticmethod
    def matmul(sparse_matrix: SparseMatrix, dense_matrix: torch.Tensor) -> torch.Tensor:
        """
        Sparse matrix-dense matrix multiplication
        
        Args:
            sparse_matrix: Sparse matrix in CSR format
            dense_matrix: Dense matrix (K x N)
            
        Returns:
            Output matrix (M x N)
        """
        # Convert to CSR if needed
        if sparse_matrix.format != "csr":
            sparse_matrix = sparse_matrix.to_csr()
        
        M, K = sparse_matrix.shape
        N = dense_matrix.shape[1]
        nnz = sparse_matrix.nnz
        
        # Prepare output
        output = torch.zeros((M, N), device=dense_matrix.device, dtype=torch.float32)
        
        # Configure grid
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 32
        grid_m = triton.cdiv(M, BLOCK_SIZE_M)
        grid_n = triton.cdiv(N, BLOCK_SIZE_N)
        
        # Launch kernel
        sparse_dense_matmul_kernel[
            (grid_m, grid_n)
        ](
            sparse_matrix.data, sparse_matrix.indices, sparse_matrix.indptr,
            dense_matrix,
            output,
            M, N, K, nnz,
            BLOCK_SIZE_M, BLOCK_SIZE_N
        )
        
        return output
    
    @staticmethod
    def matvec(sparse_matrix: SparseMatrix, dense_vector: torch.Tensor) -> torch.Tensor:
        """
        Sparse matrix-vector multiplication
        
        Args:
            sparse_matrix: Sparse matrix in COO format
            dense_vector: Dense vector (N,)
            
        Returns:
            Output vector (M,)
        """
        # Convert to COO if needed
        if sparse_matrix.format != "coo":
            sparse_matrix = sparse_matrix.to_coo()
        
        M, N = sparse_matrix.shape
        nnz = sparse_matrix.nnz
        
        # Prepare output
        output = torch.zeros(M, device=dense_vector.device, dtype=torch.float32)
        
        # Configure grid
        BLOCK_SIZE = 1024
        grid = triton.cdiv(nnz, BLOCK_SIZE)
        
        # Launch kernel
        sparse_vector_matmul_kernel[grid](
            sparse_matrix.indices, sparse_matrix.indptr, sparse_matrix.data,
            dense_vector,
            output,
            M, N, nnz,
            BLOCK_SIZE
        )
        
        return output
    
    @staticmethod
    def create_random_sparse(shape: Tuple[int, int], sparsity: float = 0.9, 
                           format: str = "csr") -> SparseMatrix:
        """
        Create random sparse matrix
        
        Args:
            shape: Matrix shape (M, N)
            sparsity: Sparsity ratio (0.0 = dense, 1.0 = all zeros)
            format: Storage format ("csr", "coo")
            
        Returns:
            Random sparse matrix
        """
        M, N = shape
        total_elements = M * N
        nnz = int(total_elements * (1.0 - sparsity))
        
        # Generate random positions
        positions = torch.randperm(total_elements)[:nnz]
        rows = positions // N
        cols = positions % N
        
        # Generate random values
        data = torch.randn(nnz)
        
        if format == "csr":
            # Convert to CSR
            indptr = torch.zeros(M + 1, dtype=torch.int32)
            for row in rows:
                indptr[row + 1] += 1
            
            # Cumulative sum
            for i in range(1, M + 1):
                indptr[i] += indptr[i - 1]
            
            # Sort by row for CSR format
            sorted_indices = torch.argsort(rows)
            rows = rows[sorted_indices]
            cols = cols[sorted_indices]
            data = data[sorted_indices]
            
            return SparseMatrix(data, cols, indptr, shape, "csr")
        
        else:  # COO
            return SparseMatrix(data, rows, cols, shape, "coo")
    
    @staticmethod
    def sparsity_pattern_analysis(sparse_matrix: SparseMatrix) -> dict:
        """
        Analyze sparsity pattern
        
        Args:
            sparse_matrix: Sparse matrix
            
        Returns:
            Dictionary with sparsity statistics
        """
        if sparse_matrix.format == "csr":
            # Analyze row-wise sparsity
            row_nnz = []
            for i in range(sparse_matrix.shape[0]):
                start, end = sparse_matrix.indptr[i], sparse_matrix.indptr[i + 1]
                row_nnz.append(end - start)
            
            row_nnz = torch.tensor(row_nnz)
            
            return {
                "total_nnz": sparse_matrix.nnz,
                "sparsity": sparse_matrix.sparsity,
                "max_row_nnz": row_nnz.max().item(),
                "min_row_nnz": row_nnz.min().item(),
                "mean_row_nnz": row_nnz.float().mean().item(),
                "std_row_nnz": row_nnz.float().std().item(),
                "empty_rows": (row_nnz == 0).sum().item()
            }
        
        else:  # COO
            return {
                "total_nnz": sparse_matrix.nnz,
                "sparsity": sparse_matrix.sparsity
            }


def benchmark_sparse_ops():
    """Benchmark sparse matrix operations"""
    print("Sparse Matrix Operations Benchmark")
    print("=" * 50)
    
    # Test different matrix sizes
    sizes = [(1000, 1000), (5000, 5000), (10000, 10000)]
    sparsities = [0.95, 0.99, 0.999]
    
    for size in sizes:
        M, N = size
        print(f"\nMatrix size: {M}x{N}")
        
        for sparsity in sparsities:
            print(f"  Sparsity: {sparsity:.3f}")
            
            # Create sparse matrix
            sparse_mat = SparseMatrixOps.create_random_sparse(
                (M, N), sparsity, "csr"
            )
            
            # Create dense matrix/vector
            dense_mat = torch.randn(N, 100)
            dense_vec = torch.randn(N)
            
            # Benchmark matrix multiplication
            import time
            start_time = time.time()
            result_mat = SparseMatrixOps.matmul(sparse_mat, dense_mat)
            matmul_time = time.time() - start_time
            
            # Benchmark matrix-vector multiplication
            sparse_mat_coo = sparse_mat.to_coo()
            start_time = time.time()
            result_vec = SparseMatrixOps.matvec(sparse_mat_coo, dense_vec)
            matvec_time = time.time() - start_time
            
            # Print results
            print(f"    MatMul time: {matmul_time:.4f}s")
            print(f"    MatVec time: {matvec_time:.4f}s")
            print(f"    Memory usage: {sparse_mat.nnz * 4 / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    benchmark_sparse_ops()