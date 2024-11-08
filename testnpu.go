import torch 
import torch_npu 
def test_cpu(): 
    input = torch.randn(2000, 1000).detach().requires_grad_() 
    output = torch.sum(input) 
    output.backward(torch.ones_like(output)) 
def test_npu(): 
    input = torch.randn(2000, 1000).detach().requires_grad_().npu() 
    output = torch.sum(input) 
    output.backward(torch.ones_like(output)) 
if __name__ == "__main__": 
    test_cpu() 
    torch_npu.npu.set_device("npu:0") 
    test_npu()