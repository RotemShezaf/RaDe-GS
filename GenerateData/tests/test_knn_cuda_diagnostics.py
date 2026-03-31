"""
Diagnostic tests for knn_cuda contiguity issues.

This module tests the specific tensor operations causing contiguity errors
in data_generation_utils.py's ring1_neighbors_gaussians function.
"""

import pytest
import torch
import numpy as np


# Check if knn_cuda is available
try:
    from knn_cuda import KNN
    KNN_CUDA_AVAILABLE = True
except (ImportError, ValueError, RuntimeError, Exception):
    KNN_CUDA_AVAILABLE = False

skip_if_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or not KNN_CUDA_AVAILABLE,
    reason="CUDA or knn_cuda not available"
)


class TestKNNCudaContiguity:
    """Test knn_cuda tensor contiguity requirements."""
    
    @skip_if_no_cuda
    def test_basic_knn_cuda_functionality(self):
        """Test basic knn_cuda functionality with simple contiguous tensors."""
        N = 100
        k = 10
        
        # Create simple contiguous tensors
        vertices = torch.randn(N, 3).cuda()
        ref = vertices.unsqueeze(0).permute(0, 2, 1).contiguous()  # [1, 3, N]
        query = ref.clone()
        
        print(f"ref shape: {ref.shape}")
        print(f"ref.is_contiguous(): {ref.is_contiguous()}")
        print(f"query.is_contiguous(): {query.is_contiguous()}")
        
        knn = KNN(k=k+1, transpose_mode=True)
        dist, indices = knn(ref, query)
        
        assert dist.shape == (1, k+1, N)
        assert indices.shape == (1, k+1, N)
        print(f"✓ Basic knn_cuda test passed")
    
    @skip_if_no_cuda
    def test_permute_contiguous_issue(self):
        """Test the exact permute pattern from ring1_neighbors_gaussians."""
        N = 100
        k = 10
        
        vertices = torch.randn(N, 3).cuda()
        
        # Exactly as in the code
        ref = vertices.unsqueeze(0).permute(0, 2, 1).contiguous()
        query = ref.contiguous()
        
        print(f"\nAfter permute and contiguous:")
        print(f"ref shape: {ref.shape}, stride: {ref.stride()}")
        print(f"ref.is_contiguous(): {ref.is_contiguous()}")
        print(f"query.is_contiguous(): {query.is_contiguous()}")
        
        knn = KNN(k=k+1, transpose_mode=True)
        try:
            dist, indices = knn(ref, query)
            print(f"✓ Permute + contiguous test passed")
        except RuntimeError as e:
            print(f"✗ Error: {e}")
            raise
    
    @skip_if_no_cuda
    def test_alternative_transpose_approach(self):
        """Test using transpose instead of permute."""
        N = 100
        k = 10
        
        vertices = torch.randn(N, 3).cuda()
        
        # Use transpose instead of permute
        ref = vertices.unsqueeze(0).transpose(1, 2).contiguous()  # [1, 3, N]
        query = ref.clone()
        
        print(f"\nUsing transpose instead of permute:")
        print(f"ref shape: {ref.shape}, stride: {ref.stride()}")
        print(f"ref.is_contiguous(): {ref.is_contiguous()}")
        
        knn = KNN(k=k+1, transpose_mode=True)
        dist, indices = knn(ref, query)
        
        assert dist.shape == (1, k+1, N)
        print(f"✓ Transpose approach test passed")
    
    @skip_if_no_cuda
    def test_direct_reshape_approach(self):
        """Test creating the desired shape directly."""
        N = 100
        k = 10
        
        vertices = torch.randn(N, 3).cuda()
        
        # Create [1, 3, N] shape directly
        ref = vertices.T.unsqueeze(0).contiguous()  # [1, 3, N]
        query = ref.clone()
        
        print(f"\nDirect reshape approach:")
        print(f"ref shape: {ref.shape}, stride: {ref.stride()}")
        print(f"ref.is_contiguous(): {ref.is_contiguous()}")
        
        knn = KNN(k=k+1, transpose_mode=True)
        dist, indices = knn(ref, query)
        
        assert dist.shape == (1, k+1, N)
        print(f"✓ Direct reshape approach test passed")
    
    @skip_if_no_cuda
    def test_different_transpose_modes(self):
        """Test knn_cuda with transpose_mode=False."""
        N = 100
        k = 10
        
        vertices = torch.randn(N, 3).cuda()
        
        # Try with transpose_mode=False - expects [batch, N, 3]
        ref = vertices.unsqueeze(0).contiguous()  # [1, N, 3]
        query = ref.clone()
        
        print(f"\nWith transpose_mode=False:")
        print(f"ref shape: {ref.shape}, stride: {ref.stride()}")
        print(f"ref.is_contiguous(): {ref.is_contiguous()}")
        
        knn = KNN(k=k+1, transpose_mode=False)
        dist, indices = knn(ref, query)
        
        assert dist.shape == (1, N, k+1)
        print(f"✓ transpose_mode=False test passed")
    
    @skip_if_no_cuda
    def test_memory_format_check(self):
        """Check memory format of tensors."""
        N = 100
        
        vertices = torch.randn(N, 3).cuda()
        
        print(f"\nMemory format checks:")
        print(f"Original vertices: {vertices.is_contiguous()}")
        
        # Method 1: permute + contiguous
        ref1 = vertices.unsqueeze(0).permute(0, 2, 1).contiguous()
        print(f"permute + contiguous: is_contiguous={ref1.is_contiguous()}, stride={ref1.stride()}")
        
        # Method 2: transpose + contiguous  
        ref2 = vertices.unsqueeze(0).transpose(1, 2).contiguous()
        print(f"transpose + contiguous: is_contiguous={ref2.is_contiguous()}, stride={ref2.stride()}")
        
        # Method 3: T + unsqueeze
        ref3 = vertices.T.unsqueeze(0).contiguous()
        print(f"T + unsqueeze: is_contiguous={ref3.is_contiguous()}, stride={ref3.stride()}")
        
        # Method 4: reshape
        ref4 = vertices.T.reshape(1, 3, N).contiguous()
        print(f"reshape: is_contiguous={ref4.is_contiguous()}, stride={ref4.stride()}")
        
        # Test each with knn_cuda
        knn = KNN(k=11, transpose_mode=True)
        for i, ref in enumerate([ref1, ref2, ref3, ref4], 1):
            try:
                dist, indices = knn(ref, ref.clone())
                print(f"  Method {i}: ✓ Success")
            except RuntimeError as e:
                print(f"  Method {i}: ✗ Failed - {e}")
    
    @skip_if_no_cuda
    def test_clone_vs_contiguous(self):
        """Test difference between clone() and contiguous()."""
        N = 100
        k = 10
        
        vertices = torch.randn(N, 3).cuda()
        ref_base = vertices.unsqueeze(0).permute(0, 2, 1)
        
        print(f"\nClone vs contiguous:")
        print(f"Base (after permute): is_contiguous={ref_base.is_contiguous()}")
        
        # Try with clone()
        ref_clone = ref_base.clone()
        print(f"After clone(): is_contiguous={ref_clone.is_contiguous()}, stride={ref_clone.stride()}")
        
        # Try with contiguous()
        ref_contig = ref_base.contiguous()
        print(f"After contiguous(): is_contiguous={ref_contig.is_contiguous()}, stride={ref_contig.stride()}")
        
        knn = KNN(k=k+1, transpose_mode=True)
        
        # Test clone
        try:
            dist1, indices1 = knn(ref_clone, ref_clone.clone())
            print(f"Clone method: ✓ Success")
        except RuntimeError as e:
            print(f"Clone method: ✗ Failed - {e}")
        
        # Test contiguous
        try:
            dist2, indices2 = knn(ref_contig, ref_contig.clone())
            print(f"Contiguous method: ✓ Success")
        except RuntimeError as e:
            print(f"Contiguous method: ✗ Failed - {e}")


class TestKNNCudaDataTypes:
    """Test knn_cuda with different data types and devices."""
    
    @skip_if_no_cuda
    def test_float32_vs_float64(self):
        """Test different floating point precisions."""
        N = 100
        k = 10
        
        vertices_f32 = torch.randn(N, 3, dtype=torch.float32).cuda()
        vertices_f64 = torch.randn(N, 3, dtype=torch.float64).cuda()
        
        knn = KNN(k=k+1, transpose_mode=True)
        
        # Test float32
        ref32 = vertices_f32.T.unsqueeze(0).contiguous()
        dist32, indices32 = knn(ref32, ref32.clone())
        print(f"Float32: ✓ Success, dist dtype={dist32.dtype}")
        
        # Test float64
        ref64 = vertices_f64.T.unsqueeze(0).contiguous()
        try:
            dist64, indices64 = knn(ref64.float(), ref64.float().clone())
            print(f"Float64 (cast to float32): ✓ Success, dist dtype={dist64.dtype}")
        except Exception as e:
            print(f"Float64: ✗ Failed - {e}")
    
    @skip_if_no_cuda
    def test_batch_sizes(self):
        """Test different batch sizes."""
        N = 100
        k = 10
        knn = KNN(k=k+1, transpose_mode=True)
        
        for batch_size in [1, 2, 4, 8]:
            vertices = torch.randn(batch_size, N, 3).cuda()
            ref = vertices.permute(0, 2, 1).contiguous()
            
            dist, indices = knn(ref, ref.clone())
            assert dist.shape == (batch_size, k+1, N)
            print(f"Batch size {batch_size}: ✓ Success")


def run_all_diagnostics():
    """Run all diagnostic tests and print summary."""
    print("="*60)
    print("KNN_CUDA DIAGNOSTIC TESTS")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("✗ CUDA not available")
        return
    
    if not KNN_CUDA_AVAILABLE:
        print("✗ knn_cuda not available")
        return
    
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"✓ knn_cuda imported successfully")
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA version: {torch.version.cuda}")
    print("="*60)
    
    test_instance = TestKNNCudaContiguity()
    
    tests = [
        ("Basic functionality", test_instance.test_basic_knn_cuda_functionality),
        ("Permute + contiguous", test_instance.test_permute_contiguous_issue),
        ("Transpose approach", test_instance.test_alternative_transpose_approach),
        ("Direct reshape", test_instance.test_direct_reshape_approach),
        ("transpose_mode=False", test_instance.test_different_transpose_modes),
        ("Memory formats", test_instance.test_memory_format_check),
        ("Clone vs contiguous", test_instance.test_clone_vs_contiguous),
    ]
    
    print("\nRunning diagnostic tests...")
    print("-"*60)
    
    for name, test_func in tests:
        print(f"\n{name}:")
        try:
            test_func()
        except Exception as e:
            print(f"✗ Test failed: {e}")
    
    print("\n" + "="*60)
    print("DIAGNOSTIC TESTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    run_all_diagnostics()
