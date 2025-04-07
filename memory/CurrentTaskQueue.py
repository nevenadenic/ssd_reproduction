from collections import defaultdict

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CurrentTaskQueue:
    def __init__(self, max_size=64):
        self.queues = defaultdict(list)
        self.max_size = max_size

    def add(self, inputs, labels):
        for x, y in zip(inputs, labels):
            y = y.item()
            if len(self.queues[y]) >= self.max_size:
                self.queues[y].pop(0)
            self.queues[y].append(x)



    def get_class(self, label):
        label = label.item()
        if len(self.queues[label]) == 0:
            return None, None
        inputs = torch.stack(self.queues[label])
        labels = torch.full((len(inputs),), label).to(device)

        return inputs, labels