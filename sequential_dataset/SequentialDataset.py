import random

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

from sequential_dataset.BalancedSampler import BalancedBatchSampler
from sequential_dataset.imagenet import load_imagenet_data


class SequentialDataset:
    def __init__(self, total_num_tasks, batch_size, dataset, do_balanced_sampling=True):
        self.total_num_tasks = total_num_tasks
        self.batch_size = batch_size
        self.do_balanced_sampling = do_balanced_sampling
        self.initial_transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        if dataset=="cifar-100":
            train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=self.initial_transform)
            test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=self.initial_transform)

            self.x_train = torch.stack(
                [train_set[example][0] for example in range(len(train_set))])
            self.y_train = torch.tensor([train_set[example][1] for example in range(len(train_set))])

            self.x_test = torch.stack([test_set[example][0] for example in range(len(test_set))])
            self.y_test = torch.tensor([test_set[example][1] for example in range(len(test_set))])

        elif dataset=="food-101":
            transform = transforms.Compose([
                transforms.Resize((84, 84)),
                transforms.ToTensor()
            ])
            train_set = torchvision.datasets.Food101(root='data', split='train', transform=transform, download=True)
            test_set = torchvision.datasets.Food101(root='data', split='test', transform=transform, download=True)
            self.x_train = torch.stack([train_set[example][0] for example in range(len(train_set))])
            self.y_train = torch.tensor([train_set[example][1] for example in range(len(train_set))])

            self.x_test = torch.stack([test_set[example][0] for example in range(len(test_set))])
            self.y_test = torch.tensor([test_set[example][1] for example in range(len(test_set))])

        else:
            self.x_train, self.y_train, self.x_test, self.y_test = load_imagenet_data(transform = self.initial_transform, dataset=dataset)

        self.total_num_classes = torch.unique(self.y_train).size(0)
        self.num_classes_per_task = self.total_num_classes // self.total_num_tasks

        class_labels = [label for label in range(self.total_num_classes)]
        self.random_class_order = class_labels[:]
        random.shuffle(self.random_class_order)


    def get_train_dataloader_for_task(self, task_number):
        classes_per_task = torch.tensor(self.random_class_order[
                           task_number * self.num_classes_per_task: (task_number + 1) * self.num_classes_per_task])

        selected_train_idx = torch.isin(self.y_train, classes_per_task)

        x_train_per_task, y_train_per_task = self.x_train[selected_train_idx], self.y_train[selected_train_idx]


        print(f"Task {task_number + 1}: Classes {classes_per_task}")
        print(f"Train Samples: {x_train_per_task.shape[0]}")

        dataset = TensorDataset(x_train_per_task, y_train_per_task)

        if self.do_balanced_sampling:
            train_loader = DataLoader(
                dataset,
                sampler=BalancedBatchSampler(dataset, y_train_per_task),
                batch_size=self.batch_size
            )
        else:
            train_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True
            )

        print(f"Train batches = {len(train_loader)}\n")
        return train_loader

    def get_test_loader_for_task(self, task_number):
        labels_for_task = self.random_class_order[task_number * 10: (task_number + 1) * 10]
        mask = torch.tensor([label in labels_for_task for label in self.y_test], dtype=torch.bool)

        x_filtered = self.x_test[mask]
        y_filtered = self.y_test[mask]

        test_loader = DataLoader(
            TensorDataset(x_filtered, y_filtered), batch_size=self.batch_size, shuffle=False
        )

        return test_loader


    def get_full_test_loader(self):
        return DataLoader(
            TensorDataset(self.x_test, self.y_test), batch_size=self.batch_size, shuffle=False
        )

    def get_image_width_and_height(self):
        return self.x_train[0].shape[1], self.x_train[0].shape[1]

    def get_total_num_classes(self):
        return self.total_num_classes
