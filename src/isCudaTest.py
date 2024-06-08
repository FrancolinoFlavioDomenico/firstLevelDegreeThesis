import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())

# import torch
# foo = torch.tensor([1,2,3])
# foo = foo.to('cuda')