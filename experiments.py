from training import train
from config import Config
from collections import defaultdict
from statistics import mean
import time

class AccuracyLogger:
    def __init__(self):
        self.data = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(list)  # <-- added fourth level for True/False keys
                )
            )
        )

    def append(self, dataset, method, size, incremental_softmax, value):
        if method == "SSD":
            self.data[dataset][method][size][incremental_softmax].append(value)
        else:
            self.data[dataset][method][size][True].append(value)

    def get_values(self, dataset, method, size, incremental_softmax=True):
        return self.data[dataset][method][size][incremental_softmax]

    def get_average(self, dataset, method, size, incremental_softmax=True):
        values = self.get_values(dataset, method, size, incremental_softmax)
        return mean(values) if values else None

    def get_all_averages(self):
        averages = {}
        for dataset, methods in self.data.items():
            averages[dataset] = {}
            for method, sizes in methods.items():
                averages[dataset][method] = {}
                for size, values in sizes.items():
                    averages[dataset][method][size] = {}
                    if method != "SSD":
                        averages[dataset][method][size] = mean(values)
                    else:
                        bool_values = values
                        for flag, values in bool_values.items():
                            averages[dataset][method][size][flag] = mean(values) if values else None
        return averages

    def log_all_averages(self, seeds):
        with open(f"averages_{time.time}.txt", "a") as f:
            f.write(f"\n\nSeeds: {seeds}\n")
            for dataset, methods in self.data.items():
                for method, sizes in methods.items():
                    for size, values in sizes.items():
                        if method != "SSD":
                            average = mean(values)
                            if average:
                                f.write(
                                    f"- Dataset: {dataset}, Method: {method}, Size: {size} → Avg: {average:.4f}\n"
                                )
                            else:
                                f.write(f"- Dataset: {dataset}, Method: {method}, Size: {size} → Avg: None\n")
                        else:
                            bool_values = values
                            for flag, values in bool_values.items():
                                average = mean(values)
                                if average:
                                    f.write(
                                        f"- Dataset: {dataset}, Method: {method}, Size: {size}, Incremental softmax: {flag} → Avg: {average:.4f}\n"
                                    )
                                else:
                                    f.write(f"- Dataset: {dataset}, Method: {method}, Size: {size}, Incremental softmax {flag} → Avg: None\n")


if __name__ == "__main__":
    # num_seeds = 5
    seeds = [0]
    dataset_filenames = ["cifar-100"]
    buffer_sizes = [100]

    configs = []

    final_test_accs = AccuracyLogger()

    for dataset_filename in dataset_filenames:
        for buffer_size in buffer_sizes:
            for seed in seeds:
                num_classes = 200 if dataset_filename=="tiny-imagenet" else 100
                num_tasks = 20 if dataset_filename=="tiny-imagenet" else 10
                buffer_size = buffer_size*2 if dataset_filename=="tiny-imagenet" else buffer_size
                summarized_per_class = buffer_size/num_classes

                configs.append(Config(dataset=dataset_filename,
                                      buffer_size=buffer_size,
                                      seed=seed,
                                      summarized_per_class=summarized_per_class,
                                      incremental_softmax=False))  # SSD
                configs.append(Config(dataset=dataset_filename,
                                      buffer_size=buffer_size,
                                      seed=seed,
                                      summarized_per_class=0)) # SCR

                configs.append(Config(dataset=dataset_filename,
                                      buffer_size=buffer_size,
                                      seed=seed,
                                      summarized_per_class=summarized_per_class,
                                      incremental_softmax=True))

                configs.append(Config(dataset=dataset_filename,
                                      buffer_size=buffer_size,
                                      seed=seed,
                                      summarized_per_class=0,
                                      contrastive_learning=False)) # ER
                configs.append(Config(dataset=dataset_filename,
                                      buffer_size=buffer_size,
                                      seed=seed,
                                      summarized_per_class=0,
                                      contrastive_learning=False,
                                      use_memory=False)) # finetune


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

        final_test_accs.append(config.dataset, method, config.buffer_size, config.incremental_softmax, final_test_acc)

    final_test_accs.log_all_averages(seeds)

