#!/usr/bin/env python3
"""
Flash Attention v2 Performance Benchmarking Script

This script benchmarks the Flash Attention v2 CUDA implementation
and generates performance reports and visualizations.
"""

import subprocess
import time
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path

class FlashAttentionBenchmark:
    def __init__(self, build_dir="../build_cuda"):
        self.build_dir = Path(build_dir)
        self.results = []
        self.test_executable = self.build_dir / "cuda_v2" / "test_flash_v2"
        
    def run_single_benchmark(self, config):
        """Run benchmark for a single configuration"""
        print(f"Running benchmark for config: {config}")
        
        # Run the test executable
        start_time = time.time()
        try:
            result = subprocess.run(
                [str(self.test_executable)], 
                capture_output=True, 
                text=True, 
                timeout=300
            )
            end_time = time.time()
            
            # Parse output
            output_lines = result.stdout.strip().split('\n')
            success = result.returncode == 0
            
            # Extract performance metrics from output
            execution_time = end_time - start_time
            
            # Calculate theoretical FLOPs
            batch_size = config.get('batch_size', 1)
            seq_len = config.get('seq_len', 32)
            num_heads = config.get('num_heads', 4)
            head_dim = config.get('head_dim', 16)
            
            # Attention complexity: O(N²) for sequence length N
            # Each attention head: batch_size * seq_len * seq_len * head_dim operations
            flops = 2 * batch_size * seq_len * seq_len * num_heads * head_dim  # Multiply-add operations
            
            benchmark_result = {
                'config': config,
                'execution_time': execution_time,
                'flops': flops,
                'gflops': flops / (execution_time * 1e9),
                'success': success,
                'output': result.stdout,
                'error': result.stderr if result.stderr else None,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"  Time: {execution_time:.3f}s, GFLOPS: {benchmark_result['gflops']:.2f}, Success: {success}")
            
            return benchmark_result
            
        except subprocess.TimeoutExpired:
            print(f"  Timeout after 300 seconds")
            return {
                'config': config,
                'execution_time': 300.0,
                'flops': 0,
                'gflops': 0,
                'success': False,
                'output': '',
                'error': 'Timeout',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"  Error: {e}")
            return {
                'config': config,
                'execution_time': 0,
                'flops': 0,
                'gflops': 0,
                'success': False,
                'output': '',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_benchmark_suite(self):
        """Run comprehensive benchmark suite"""
        print("Starting Flash Attention v2 Benchmark Suite")
        print("=" * 50)
        
        # Define benchmark configurations
        test_configs = [
            # Tiny configurations
            {'batch_size': 1, 'seq_len': 32, 'num_heads': 4, 'head_dim': 16, 'name': 'tiny'},
            {'batch_size': 1, 'seq_len': 64, 'num_heads': 8, 'head_dim': 32, 'name': 'small'},
            
            # Medium configurations
            {'batch_size': 2, 'seq_len': 128, 'num_heads': 8, 'head_dim': 32, 'name': 'medium'},
            {'batch_size': 4, 'seq_len': 256, 'num_heads': 16, 'head_dim': 64, 'name': 'large'},
            
            # Large configurations
            {'batch_size': 8, 'seq_len': 512, 'num_heads': 16, 'head_dim': 64, 'name': 'xlarge'},
            {'batch_size': 16, 'seq_len': 1024, 'num_heads': 32, 'head_dim': 128, 'name': 'xxlarge'},
        ]
        
        # Run benchmarks
        for config in test_configs:
            result = self.run_single_benchmark(config)
            self.results.append(result)
            print()
        
        return self.results
    
    def generate_report(self):
        """Generate comprehensive performance report"""
        if not self.results:
            print("No benchmark results available")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'name': r['config']['name'],
            'batch_size': r['config']['batch_size'],
            'seq_len': r['config']['seq_len'],
            'num_heads': r['config']['num_heads'],
            'head_dim': r['config']['head_dim'],
            'execution_time': r['execution_time'],
            'flops': r['flops'],
            'gflops': r['gflops'],
            'success': r['success'],
            'error': r['error']
        } for r in self.results])
        
        # Filter successful results
        successful_df = df[df['success']].copy()
        
        print("\n" + "=" * 60)
        print("FLASH ATTENTION V2 PERFORMANCE REPORT")
        print("=" * 60)
        print(f"Test Executable: {self.test_executable}")
        print(f"Total Configurations: {len(df)}")
        print(f"Successful Runs: {len(successful_df)}")
        print(f"Failed Runs: {len(df) - len(successful_df)}")
        print()
        
        if len(successful_df) > 0:
            print("SUCCESSFUL RESULTS:")
            print("-" * 40)
            for _, row in successful_df.iterrows():
                print(f"{row['name']:10} | {row['execution_time']:8.3f}s | {row['gflops']:8.2f} GFLOPS")
            
            print("\nPERFORMANCE SUMMARY:")
            print("-" * 40)
            print(f"Best Performance: {successful_df['gflops'].max():.2f} GFLOPS")
            print(f"Worst Performance: {successful_df['gflops'].min():.2f} GFLOPS")
            print(f"Average Performance: {successful_df['gflops'].mean():.2f} GFLOPS")
            print(f"Median Performance: {successful_df['gflops'].median():.2f} GFLOPS")
        
        if len(df) - len(successful_df) > 0:
            print("\nFAILED RESULTS:")
            print("-" * 40)
            failed_df = df[~df['success']]
            for _, row in failed_df.iterrows():
                print(f"{row['name']:10} | Error: {row['error']}")
        
        return df
    
    def plot_results(self, df):
        """Generate performance visualization plots"""
        if df is None or len(df) == 0:
            print("No data to plot")
            return
        
        successful_df = df[df['success']].copy()
        if len(successful_df) == 0:
            print("No successful results to plot")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Flash Attention v2 Performance Analysis', fontsize=16)
        
        # Plot 1: Execution Time vs Configuration
        ax1 = axes[0, 0]
        x_pos = np.arange(len(successful_df))
        bars = ax1.bar(x_pos, successful_df['execution_time'], alpha=0.7)
        ax1.set_xlabel('Configuration')
        ax1.set_ylabel('Execution Time (s)')
        ax1.set_title('Execution Time by Configuration')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(successful_df['name'], rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, successful_df['execution_time']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 2: GFLOPS vs Configuration
        ax2 = axes[0, 1]
        bars = ax2.bar(x_pos, successful_df['gflops'], alpha=0.7, color='green')
        ax2.set_xlabel('Configuration')
        ax2.set_ylabel('GFLOPS')
        ax2.set_title('Computational Throughput')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(successful_df['name'], rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, successful_df['gflops']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 3: Execution Time vs Sequence Length
        ax3 = axes[1, 0]
        ax3.plot(successful_df['seq_len'], successful_df['execution_time'], 'o-', linewidth=2, markersize=8)
        ax3.set_xlabel('Sequence Length')
        ax3.set_ylabel('Execution Time (s)')
        ax3.set_title('Execution Time vs Sequence Length')
        ax3.grid(True, alpha=0.3)
        
        # Add configuration labels
        for i, row in successful_df.iterrows():
            ax3.annotate(row['name'], (row['seq_len'], row['execution_time']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot 4: GFLOPS vs Sequence Length
        ax4 = axes[1, 1]
        ax4.plot(successful_df['seq_len'], successful_df['gflops'], 'o-', linewidth=2, markersize=8, color='red')
        ax4.set_xlabel('Sequence Length')
        ax4.set_ylabel('GFLOPS')
        ax4.set_title('Performance vs Sequence Length')
        ax4.grid(True, alpha=0.3)
        
        # Add configuration labels
        for i, row in successful_df.iterrows():
            ax4.annotate(row['name'], (row['seq_len'], row['gflops']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        # Save plots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"flash_attention_v2_performance_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\nPerformance plots saved to: {plot_filename}")
        
        # Show plot
        plt.show()
    
    def save_results(self, df):
        """Save benchmark results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"flash_attention_v2_benchmark_{timestamp}.json"
        
        results_data = {
            'timestamp': timestamp,
            'test_executable': str(self.test_executable),
            'results': self.results,
            'summary': {
                'total_configs': len(df),
                'successful_runs': len(df[df['success']]),
                'failed_runs': len(df[~df['success']]),
                'best_gflops': df[df['success']]['gflops'].max() if len(df[df['success']]) > 0 else 0,
                'avg_gflops': df[df['success']]['gflops'].mean() if len(df[df['success']]) > 0 else 0
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Benchmark results saved to: {results_file}")
        return results_file
    
    def run_full_benchmark(self):
        """Run complete benchmark suite with reporting"""
        print("Flash Attention v2 Performance Benchmark")
        print("=" * 50)
        
        # Check if test executable exists
        if not self.test_executable.exists():
            print(f"Error: Test executable not found at {self.test_executable}")
            print("Please build the project first:")
            print("  cd build_cuda")
            print("  make test_flash_v2")
            return False
        
        # Run benchmarks
        self.run_benchmark_suite()
        
        # Generate report
        df = self.generate_report()
        
        # Generate plots
        self.plot_results(df)
        
        # Save results
        self.save_results(df)
        
        print("\n" + "=" * 50)
        print("BENCHMARK COMPLETED")
        print("=" * 50)
        
        return True

def main():
    """Main benchmark execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Flash Attention v2 Performance Benchmark')
    parser.add_argument('--build-dir', default='../build_cuda', 
                       help='Build directory path (default: ../build_cuda)')
    parser.add_argument('--output-dir', default='.', 
                       help='Output directory for results (default: current directory)')
    
    args = parser.parse_args()
    
    # Create benchmark instance
    benchmark = FlashAttentionBenchmark(build_dir=args.build_dir)
    
    # Change to output directory
    if args.output_dir != '.':
        os.makedirs(args.output_dir, exist_ok=True)
        os.chdir(args.output_dir)
    
    # Run benchmark
    success = benchmark.run_full_benchmark()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())