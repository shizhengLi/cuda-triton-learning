#!/usr/bin/env python3
"""
Flash Attention 性能基准测试工具
用于比较不同注意力实现的性能和内存使用
"""
import torch
import sys
import os
import time
import math
import json
import matplotlib.pyplot as plt
from datetime import datetime

# 添加路径到实现
sys.path.append("/data/lishizheng/cpp_projects/cuda-triton-learning/flash_attention/naive")
sys.path.append("/data/lishizheng/cpp_projects/cuda-triton-learning/flash_attention/flash_v1/triton")
sys.path.append("/data/lishizheng/cpp_projects/cuda-triton-learning/flash_attention/flash_v2/triton")

from naive_attention import naive_attention
from flash_attention_v1 import flash_attention as flash_attention_v1
from flash_attention_v2_simple import flash_attention_v2

class FlashAttentionBenchmark:
    """Flash Attention 性能基准测试类"""
    
    def __init__(self):
        self.results = {}
        self.implementations = {
            'naive': naive_attention,
            'flash_v1': flash_attention_v1,
            'flash_v2': flash_attention_v2,
        }
        
    def measure_memory_usage(self, func, *args, **kwargs):
        """测量函数执行的内存使用"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 记录初始内存
        start_memory = torch.cuda.memory_allocated()
        
        # 执行函数
        result = func(*args, **kwargs)
        
        # 记录峰值内存
        peak_memory = torch.cuda.max_memory_allocated()
        
        # 计算内存使用量
        memory_used = peak_memory - start_memory
        
        return result, memory_used / (1024 ** 2)  # 转换为 MB
    
    def benchmark_implementation(self, impl_name, impl_func, q, k, v, num_runs=10):
        """对单个实现进行基准测试"""
        print(f"  测试 {impl_name}...")
        
        try:
            # 测试正确性
            output = impl_func(q, k, v)
            
            # 基本检查
            assert output.shape == q.shape, f"形状不匹配: {output.shape} vs {q.shape}"
            assert not torch.isnan(output).any(), f"输出包含 NaN"
            assert torch.isfinite(output).all(), f"输出包含无限值"
            
            # 测量内存使用
            _, memory_used = self.measure_memory_usage(impl_func, q, k, v)
            
            # 性能测试
            times = []
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start_time = time.time()
                _ = impl_func(q, k, v)
                torch.cuda.synchronize()
                elapsed_time = (time.time() - start_time) * 1000  # 转换为毫秒
                times.append(elapsed_time)
            
            # 计算统计数据
            avg_time = sum(times) / len(times)
            std_time = math.sqrt(sum((t - avg_time) ** 2 for t in times) / len(times))
            min_time = min(times)
            max_time = max(times)
            
            return {
                'success': True,
                'output_shape': tuple(output.shape),
                'avg_time_ms': avg_time,
                'std_time_ms': std_time,
                'min_time_ms': min_time,
                'max_time_ms': max_time,
                'memory_mb': memory_used,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_comprehensive_benchmark(self):
        """运行全面的基准测试"""
        print("Flash Attention 综合性能基准测试")
        print("=" * 60)
        
        # 测试配置
        test_configs = [
            # (batch_size, seq_len, head_dim, description)
            (1, 128, 32, "tiny"),
            (1, 256, 64, "small"),
            (2, 512, 64, "medium"),
            (4, 1024, 64, "large"),
            (8, 2048, 64, "xlarge"),
        ]
        
        results = {}
        
        for batch_size, seq_len, head_dim, config_name in test_configs:
            print(f"\n测试配置: {config_name} ({batch_size}x{seq_len}x{head_dim})")
            print("-" * 40)
            
            # 创建测试数据
            q = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
            k = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
            v = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
            
            config_results = {}
            
            # 测试每个实现
            for impl_name, impl_func in self.implementations.items():
                result = self.benchmark_implementation(impl_name, impl_func, q, k, v)
                config_results[impl_name] = result
                
                if result['success']:
                    print(f"  ✓ {impl_name}: {result['avg_time_ms']:.2f} ± {result['std_time_ms']:.2f} ms, "
                          f"{result['memory_mb']:.2f} MB")
                else:
                    print(f"  ✗ {impl_name}: 失败 - {result['error']}")
            
            # 计算相对性能
            self._calculate_relative_performance(config_results)
            
            results[config_name] = {
                'config': {
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'head_dim': head_dim,
                    'description': config_name
                },
                'results': config_results
            }
        
        self.results = results
        return results
    
    def _calculate_relative_performance(self, config_results):
        """计算相对性能指标"""
        # 找到成功的朴素实现作为基准
        naive_result = config_results.get('naive', {})
        if not naive_result.get('success'):
            return
        
        baseline_time = naive_result['avg_time_ms']
        baseline_memory = naive_result['memory_mb']
        
        # 计算每个实现的相对性能
        for impl_name, result in config_results.items():
            if impl_name == 'naive' or not result.get('success'):
                continue
            
            # 计算加速比
            speedup = baseline_time / result['avg_time_ms']
            result['speedup_vs_naive'] = speedup
            
            # 计算内存节省
            memory_saving_pct = (baseline_memory - result['memory_mb']) / baseline_memory * 100
            result['memory_saving_pct'] = memory_saving_pct
    
    def generate_report(self):
        """生成性能报告"""
        if not self.results:
            print("没有可用的基准测试结果")
            return
        
        print("\n" + "=" * 60)
        print("性能基准测试报告")
        print("=" * 60)
        
        # 总体统计
        total_tests = 0
        successful_tests = 0
        
        for config_name, config_data in self.results.items():
            for impl_name, result in config_data['results'].items():
                total_tests += 1
                if result['success']:
                    successful_tests += 1
        
        print(f"总体成功率: {successful_tests}/{total_tests} ({successful_tests/total_tests*100:.1f}%)")
        
        # 详细结果
        for config_name, config_data in self.results.items():
            print(f"\n配置: {config_name}")
            print("-" * 40)
            
            config_info = config_data['config']
            print(f"  批次大小: {config_info['batch_size']}")
            print(f"  序列长度: {config_info['seq_len']}")
            print(f"  头部维度: {config_info['head_dim']}")
            print(f"  总参数量: {config_info['batch_size'] * config_info['seq_len'] * config_info['head_dim'] * 3:,}")
            
            # 显示结果
            for impl_name, result in config_data['results'].items():
                if result['success']:
                    print(f"  {impl_name}:")
                    print(f"    执行时间: {result['avg_time_ms']:.2f} ± {result['std_time_ms']:.2f} ms")
                    print(f"    内存使用: {result['memory_mb']:.2f} MB")
                    
                    if 'speedup_vs_naive' in result:
                        print(f"    加速比: {result['speedup_vs_naive']:.2f}x")
                        print(f"    内存节省: {result['memory_saving_pct']:.1f}%")
                else:
                    print(f"  {impl_name}: 失败 - {result['error']}")
    
    def plot_results(self):
        """绘制性能图表"""
        if not self.results:
            print("没有可用的基准测试结果")
            return
        
        # 准备数据
        configs = []
        naive_times = []
        flash_v1_times = []
        flash_v2_times = []
        
        for config_name, config_data in self.results.items():
            config_info = config_data['config']
            configs.append(f"{config_info['seq_len']}")
            
            results = config_data['results']
            naive_times.append(results.get('naive', {}).get('avg_time_ms', 0))
            flash_v1_times.append(results.get('flash_v1', {}).get('avg_time_ms', 0))
            flash_v2_times.append(results.get('flash_v2', {}).get('avg_time_ms', 0))
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 执行时间对比
        x = range(len(configs))
        width = 0.25
        
        ax1.bar([i - width for i in x], naive_times, width, label='Naive', alpha=0.8)
        ax1.bar(x, flash_v1_times, width, label='Flash v1', alpha=0.8)
        ax1.bar([i + width for i in x], flash_v2_times, width, label='Flash v2', alpha=0.8)
        
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Execution Time (ms)')
        ax1.set_title('Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(configs)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 加速比对比
        flash_v1_speedups = [n/t if t > 0 else 0 for n, t in zip(naive_times, flash_v1_times)]
        flash_v2_speedups = [n/t if t > 0 else 0 for n, t in zip(naive_times, flash_v2_times)]
        
        ax2.plot(configs, flash_v1_speedups, 'o-', label='Flash v1', linewidth=2, markersize=8)
        ax2.plot(configs, flash_v2_speedups, 's-', label='Flash v2', linewidth=2, markersize=8)
        
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Speedup vs Naive')
        ax2.set_title('Speedup Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('flash_attention_performance.png', dpi=300, bbox_inches='tight')
        print(f"\n性能图表已保存为 'flash_attention_performance.png'")
        
        # 显示图表
        plt.show()
    
    def save_results(self, filename=None):
        """保存基准测试结果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flash_attention_benchmark_{timestamp}.json"
        
        # 准备可序列化的数据
        serializable_results = {}
        for config_name, config_data in self.results.items():
            serializable_results[config_name] = {
                'config': config_data['config'],
                'results': {}
            }
            
            for impl_name, result in config_data['results'].items():
                serializable_results[config_name]['results'][impl_name] = result.copy()
                # 移除无法序列化的张量
                if 'output_shape' in serializable_results[config_name]['results'][impl_name]:
                    del serializable_results[config_name]['results'][impl_name]['output_shape']
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"基准测试结果已保存为 '{filename}'")
    
    def load_results(self, filename):
        """加载基准测试结果"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
            print(f"基准测试结果已从 '{filename}' 加载")
            return True
        except Exception as e:
            print(f"加载基准测试结果失败: {e}")
            return False

def main():
    """主函数"""
    print("Flash Attention 性能基准测试工具")
    print("=" * 60)
    
    # 创建基准测试实例
    benchmark = FlashAttentionBenchmark()
    
    # 运行基准测试
    results = benchmark.run_comprehensive_benchmark()
    
    # 生成报告
    benchmark.generate_report()
    
    # 绘制图表
    benchmark.plot_results()
    
    # 保存结果
    benchmark.save_results()
    
    print("\n" + "=" * 60)
    print("基准测试完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()