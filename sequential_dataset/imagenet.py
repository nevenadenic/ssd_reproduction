import pickle
import numpy as np
import torch

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_imagenet_data(transform, dataset):

    if dataset=="mini-imagenet":

        train_data = load_pickle('data/mini-imagenet/train.pkl')
        val_data = load_pickle('data/mini-imagenet/val.pkl')
        test_data = load_pickle('data/mini-imagenet/test.pkl')

        datasets = [train_data, val_data, test_data]

        all_images = np.concatenate([train_data['image_data'], val_data['image_data'], test_data['image_data']], axis=0)

        class_dict = {}
        offset = 0

        for dataset in datasets:
            for class_name, indices in dataset['class_dict'].items():
                if class_name not in class_dict:
                    class_dict[class_name] = []
                class_dict[class_name].extend([idx + offset for idx in indices])
            offset += dataset['image_data'].shape[0]

        class_names = sorted(class_dict.keys())
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

        labels = []
        for class_name, indices in class_dict.items():
            numeric_label = class_to_idx[class_name]
            labels.extend([(idx, numeric_label) for idx in indices])

        np.random.shuffle(labels)
        image_indices, image_labels = zip(*labels)
        image_labels = torch.tensor(image_labels, dtype=torch.long)
        image_tensors = torch.stack([transform(all_images[idx]) for idx in image_indices])

    else:
        train_data = load_pickle('data/tiny-imagenet-200/train.pkl')
        val_data = load_pickle('data/tiny-imagenet-200/val.pkl')

        all_images = np.concatenate([train_data['data'], val_data['data']], axis=0)
        image_tensors = torch.stack([transform(all_images[idx]) for idx in range(len(all_images))])
        image_tensors = image_tensors.permute(0, 2, 1, 3)

        image_labels =  np.concatenate([train_data['target'], val_data['target']], axis=0)
        image_labels = torch.tensor(image_labels, dtype=torch.long)

    from collections import defaultdict

    class_images = defaultdict(list)
    for idx, (img, label) in enumerate(zip(image_tensors, image_labels)):
        class_images[label.item()].append(idx)

    x_train, y_train = [], []
    x_test, y_test = [], []

    for class_label, image_indices in class_images.items():
        images = []
        for idx in image_indices:
            images.append(image_tensors[idx])

        x_train.extend(images[:500])
        y_train.extend([class_label] * 500)

        x_test.extend(images[500:])
        y_test.extend([class_label] * (len(images) - 500))

    x_train = torch.stack(x_train)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_test = torch.stack(x_test)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return x_train, y_train, x_test, y_test

