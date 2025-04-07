from training import train
from config import Config
import random
from collections import defaultdict
from statistics import mean

class AccuracyLogger:
    def __init__(self):
        self.data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    def append(self, dataset, method, size, value):
        self.data[dataset][method][size].append(value)

    def get_values(self, dataset, method, size):
        return self.data[dataset][method][size]

    def get_average(self, dataset, method, size):
        values = self.get_values(dataset, method, size)
        return mean(values) if values else None

    def get_all_averages(self):
        averages = {}
        for dataset, methods in self.data.items():
            averages[dataset] = {}
            for method, sizes in methods.items():
                averages[dataset][method] = {}
                for size, values in sizes.items():
                    averages[dataset][method][size] = mean(values) if values else None
        return averages

    def log_all_averages(self, seeds):
        with open("averages.txt", "a") as f:
            f.write(f"\n\nSeeds: {seeds}\n")
            for dataset, methods in self.data.items():
                for method, sizes in methods.items():
                    for size, values in sizes.items():
                        avg = mean(values) if values else None
                        f.write(
                            f"- Dataset: {dataset}, Method: {method}, Size: {size} → Avg: {avg:.4f}\n" if avg is not None else
                            f"- Dataset: {dataset}, Method: {method}, Size: {size} → No values logged.")


if __name__ == "__main__":
    num_seeds = 1
    random_seeds = [random.randint(0, 10000) for _ in range(num_seeds)]
    dataset_filenames = ["cifar-100", "mini-imagenet", "tiny-imagenet", "food-101"]
    buffer_sizes = [100, 500, 1000]

    configs = []

    final_test_accs = AccuracyLogger()

    for dataset_filename in dataset_filenames:
        for buffer_size in buffer_sizes:
            for random_seed in random_seeds:
                num_classes = 200 if dataset_filename=="tiny-imagenet" else 100
                num_tasks = 20 if dataset_filename=="tiny-imagenet" else 10
                buffer_size = buffer_size*2 if dataset_filename=="tiny-imagenet" else buffer_size
                summarized_per_class = buffer_size/num_classes

                configs.append(Config(dataset=dataset_filename, buffer_size=buffer_size, seed=random_seed, summarized_per_class=summarized_per_class, incremental_softmax=False))  # SSD
                configs.append(Config(dataset=dataset_filename, buffer_size=buffer_size, seed=random_seed, summarized_per_class=0)) # SCR

                configs.append(Config(dataset=dataset_filename, buffer_size=buffer_size, seed=random_seed, summarized_per_class=0, contrastive_learning=False)) # ER
                configs.append(Config(dataset=dataset_filename, buffer_size=buffer_size, seed=random_seed, summarized_per_class=0, contrastive_learning=False, use_memory=False)) # finetune


    for config in configs:
        final_test_acc = train(config)

        if config.summarized_per_class > 0:
            method = "SSD"
        elif config.contrastive_learning == True:
            method = "SCR"
        elif config.use_memory == True:
            method = "ER"
        else:
            method = "finetune"

        final_test_accs.append(config.dataset, method, config.buffer_size, final_test_acc)


        print(f"Current average for {config.dataset} with {config.buffer_size} size, method: {method}: {final_test_accs.get_average(config.dataset, method, config.buffer_size)}%")

    final_test_accs.log_all_averages(random_seeds)

