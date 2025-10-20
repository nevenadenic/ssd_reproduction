import random

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DynamicMemoryBuffer:
    def __init__(self, total_capacity, summarized_per_class, number_of_classes):

        self.total_capacity = total_capacity
        self.summarized_per_class = summarized_per_class
        self.number_of_classes = number_of_classes

        self.buffer = []

        self.current_index = 0

        self.indices_ms_per_class: list[list[int]] = [[] for _ in range(number_of_classes)]

        self.indices_mo = []
        self.total_seen_mo = 0


    def add_batch(self, inputs, labels):
        for (x, y) in zip(inputs, labels):
            self.total_seen_mo += 1
            if len(self.buffer) < self.total_capacity:
                self.buffer.append((x, y))
                if len(self.indices_ms_per_class[y]) < self.summarized_per_class:
                    self.indices_ms_per_class[y].append(self.current_index)
                else:
                    self.indices_mo.append(self.current_index)

                self.current_index += 1

            else:
                if len(self.indices_ms_per_class[y]) < self.summarized_per_class:
                    replace_num = random.randint(0, len(self.indices_mo) - 1)
                    buffer_replace_index = self.indices_mo[replace_num]
                    self.indices_mo.remove(buffer_replace_index)

                    self.buffer[buffer_replace_index] = (x, y)
                    self.indices_ms_per_class[y].append(buffer_replace_index)

                else:
                    replace_num = random.randint(0, self.total_seen_mo - 1)
                    if replace_num < len(self.indices_mo):
                        buffer_replace_index = self.indices_mo[replace_num]
                        self.buffer[buffer_replace_index] = (x, y)

    def sample_batch(self, batch_size, discard_current_task=False):
        if len(self.buffer) == 0:
            return None, None
        sampled_indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))

        return self._get_tensors_for_indices(sampled_indices)


    def sample_mo_for_summarizing_model_training(self, batch_size, discard_current_task=False):
        sampled_indices = random.sample(self.indices_mo, min(batch_size, len(self.indices_mo)))

        return self._get_tensors_for_indices(sampled_indices)


    def get_mc(self, label):
        indices = self.indices_ms_per_class[label]
        return self._get_tensors_for_indices(indices)


    def get_ms_diff_mc(self, label_to_discard, batch_size, current_task, discard_current_task=True):
        labels_to_discard = [label_to_discard]
        if discard_current_task:
            labels_to_discard.extend(range(current_task * 10, (current_task + 1) * 10))

        all_other_summarizing_indices = []
        for label in range((current_task + 1) * 10):
            if label in labels_to_discard:
                continue

            all_other_summarizing_indices.extend(self.indices_ms_per_class[label])

        sampled_indices = random.sample(all_other_summarizing_indices, min(batch_size, len(all_other_summarizing_indices)))
        return self._get_tensors_for_indices(sampled_indices)

    def _get_tensors_for_indices(self, indices):
        if len(indices) == 0:
            return None, None
        sampled_inputs, sampled_labels = zip(*[self.buffer[i] for i in indices])
        sampled_inputs, sampled_labels = torch.stack(sampled_inputs), torch.tensor(sampled_labels).to(device)

        return sampled_inputs, sampled_labels

    def update_mc(self, new_tensors, label):
        for idx, new_tensor in enumerate(new_tensors):
            index_in_buffer = self.indices_ms_per_class[label][idx]

            self.buffer[index_in_buffer] = (new_tensor, label)

    def is_full_mc(self, label):
        return len(self.indices_ms_per_class[label]) == self.summarized_per_class

    def is_not_empty(self):
        return len(self.buffer) > 0

    def get_all_with_label(self, label):
        selected_indices = []
        for idx in range(len(self.buffer)):
            if self.buffer[idx][1] == label:
                selected_indices.append(idx)

        return self._get_tensors_for_indices(selected_indices)