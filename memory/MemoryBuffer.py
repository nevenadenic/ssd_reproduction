import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MemoryBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.total_seen = 0

    def add_batch(self, inputs, labels):
        """
        Implements reservoir sampling:
        - If the buffer is not full, add new sample.
        - If the buffer is full, replace a random existing sample with decreasing probability.
        """
        for (x, y) in zip(inputs, labels):
            self.total_seen += 1

            if len(self.buffer) < self.capacity:
                self.buffer.append((x, y))
            else:
                replace_idx = random.randint(0, self.total_seen - 1)
                if replace_idx < self.capacity:
                    self.buffer[replace_idx] = (x, y)

    def sample_batch(self, batch_size):
        """
        Randomly samples a batch from the memory buffer.
        """
        if len(self.buffer) == 0:
            return None, None
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        batch_inputs, batch_labels = zip(*batch)
        batch_inputs = torch.stack(batch_inputs)
        batch_labels = torch.tensor(batch_labels)
        return batch_inputs.to(device), batch_labels.to(device)


    def is_not_empty(self):
        return len(self.buffer) > 0

    def get_all_with_label(self, label):
        selected_indices = []
        for idx in range(len(self.buffer)):
            if self.buffer[idx][1] == label:
                selected_indices.append(idx)

        return self._get_tensors_for_indices(selected_indices)

    def _get_tensors_for_indices(self, indices):
        if len(indices) == 0:
            return None, None
        sampled_inputs, sampled_labels = zip(*[self.buffer[i] for i in indices])
        sampled_inputs, sampled_labels = torch.stack(sampled_inputs), torch.tensor(sampled_labels).to(device)

        return sampled_inputs, sampled_labels