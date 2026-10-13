import torch
from torch import nn


device = "cuda:0"

def mixed_precision_accumulation():

    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float32)
    print(s)

    s = torch.tensor(0,dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s)

    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s)

    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01,dtype=torch.float16)
        s += x.type(torch.float32)
    print(s)


def benckmarking_mixed_precision():

    class ToyModel(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.fc1 = nn.Linear(in_features, 10, bias=False)
            self.ln = nn.LayerNorm(10)
            self.fc2 = nn.Linear(10, out_features, bias=False)
            self.relu = nn.ReLU()
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.ln(x)
            x = self.fc2(x)
            return x

    model : torch.nn.Module = ToyModel(10, 5).to(device)
    dtype : torch.dtype = torch.float16
    x : torch.Tensor = torch.randn(1, 10, dtype=torch.float32).to(device)
    l : torch.Tensor = torch.randn(1, 5, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss()
    print(x.dtype)

    with torch.autocast(device_type="cuda", dtype=dtype):
        print(f"model parameter:\t {model.fc1.weight.data.dtype}")
        x = model.fc1(x)
        print(f"output of fc1:\t {x.dtype}")
        x = model.relu(x)
        x = model.ln(x)
        print(f"output of ln:\t {x.dtype}")
        x = model.fc2(x)
        print(f"output of model:\t {x.dtype}")
        loss = criterion(x, l)
        print(f"loss dtype:\t {loss.dtype}")
        loss.backward()
        print(f"grad dtype:\t {model.fc1.weight.grad.dtype}")
        
        

if __name__ == "__main__":

    print(f"cuda available is {torch.cuda.is_available()}, {torch.cuda.get_device_name(0)}")     # 应该是 True, 打印 GPU 名字

    # mixed_precision_accumulation()
    benckmarking_mixed_precision()