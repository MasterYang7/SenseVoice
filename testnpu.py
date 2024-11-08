import torch 
import time
import torch_npu 
def test_cpu(): 
    start_time = time.perf_counter()
    input = torch.randn(2000, 1000).detach().requires_grad_() 
    output = torch.sum(input) 
    output.backward(torch.ones_like(output)) 
    end_time = time.perf_counter()
    # 计算运行时间
    execution_time = end_time - start_time
    print(f"代码运行时间: {execution_time} 秒")
def test_npu(): 
    start_time = time.perf_counter()
    input = torch.randn(2000, 1000).detach().requires_grad_().npu() 
    output = torch.sum(input) 
    output.backward(torch.ones_like(output)) 
    end_time = time.perf_counter()
    # 计算运行时间
    execution_time = end_time - start_time
    print(f"代码运行时间: {execution_time} 秒")
if __name__ == "__main__": 
    test_cpu() 
    start_time = time.perf_counter()
    torch_npu.npu.set_device("npu:0") 
    end_time = time.perf_counter()
    # 计算运行时间
    execution_time = end_time - start_time
    print(f"代码运行时间: {execution_time} 秒")
    test_npu()