import torch
criteria = torch.nn.CrossEntropyLoss(ignore_index=0)

output = torch.rand(3, 5, requires_grad=True)
target = torch.zeros(3, dtype=torch.long)
target[1] = 1

loss = criteria(output, target)
print(loss)