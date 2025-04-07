import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NearestClassMeanClassifier:
    def __init__(self, num_classes=100, feature_dim=160):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.class_means = torch.zeros(num_classes, feature_dim).to(device)
        self.class_counts = torch.zeros(num_classes).to(device)

    def update_means(self, memory, task_number, model):
        self.empty()
        for label in range(10 * (task_number + 1)):
            ncm_inputs, ncm_labels = memory.get_all_with_label(label)
            with torch.no_grad():
                if ncm_labels is not None:
                    ncm_embeddings = model.features(ncm_inputs).detach().clone()
                else:
                    ncm_embeddings = torch.normal(0, 1, size=(1, self.feature_dim)).to(device)

                self._calculate_mean_for_class(ncm_embeddings, label)


    def _calculate_mean_for_class(self, features, label):
        features = features / features.norm(dim=1, keepdim=True)
        feature_mean = features.mean(dim=0)
        self.class_means[label] = feature_mean / feature_mean.norm()
        self.class_counts[label] += features.size(0)


    def empty(self):
        self.class_means = torch.zeros(self.num_classes, self.feature_dim).to(device)
        self.class_counts = torch.zeros(self.num_classes).to(device)

    def __call__(self, features):
        features = features / features.norm(dim=1, keepdim=True)
        dists = torch.cdist(features, self.class_means)

        return torch.argmin(dists, dim=1)

    def print_means(self):
        for i in range(len(self.class_means)):
            print(f"Class {i} mean: {self.class_means[i]}")
            print(f"Class {i} counts: {self.class_counts[i]}")