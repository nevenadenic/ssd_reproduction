from collections import defaultdict

import numpy as np
import torch

def maybe_cuda(what, use_cuda=True, **kw):
    if use_cuda is not False and torch.cuda.is_available():
        what = what.cuda()
    return what

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.sum += val * n
        self.count += n

    def avg(self):
        if self.count == 0:
            return 0
        return float(self.sum) / self.count


def njihov_evaluate(test_loaders, model, buffer, class_mapping):
    model.eval()
    acc_array = np.zeros(len(test_loaders))
    exemplar_means = {}
    # cls_exemplar = {cls: [] for cls in self.old_labels}
    # buffer_filled = buffer.current_index
    # for x, y in zip(buffer.buffer_img[:buffer_filled], buffer.buffer_label[:buffer_filled]):
    #     cls_exemplar[y.item()].append(x)

    cls_exemplar = {cls: [] for cls in range(10*len(test_loaders))}

    # print(cls_exemplar)

    for x,y in buffer.buffer:
        # print("Labele u bufferu:")
        # print(y)
        cls_exemplar[y.item()].append(x) # ovde je ok jer su smesteni ok ovamo u buffer

    #
    # print(cls_exemplar)

    for cls, exemplar in cls_exemplar.items():
        features = []
        # Extract feature for each exemplar in p_y
        for ex in exemplar:
            feature = model.features(ex.unsqueeze(0)).detach().clone()
            feature = feature.squeeze()
            feature.data = feature.data / feature.data.norm()  # Normalize
            features.append(feature)
        if len(features) == 0:
            mu_y = torch.normal(0, 1, size=tuple(model.features(x.unsqueeze(0)).detach().size())).cuda()

            mu_y = mu_y.squeeze()
        else:
            features = torch.stack(features)
            mu_y = features.mean(0).squeeze()
        mu_y.data = mu_y.data / mu_y.data.norm()  # Normalize
        exemplar_means[cls] = mu_y


    with torch.no_grad():
        for task, test_loader in enumerate(test_loaders):
            acc = AverageMeter()
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = maybe_cuda(batch_x)
                batch_y = maybe_cuda(batch_y)

                batch_y = class_mapping[batch_y] # ovo je neophodno ovde



                feature = model.features(batch_x)  # (batch_size, feature_size)
                for j in range(feature.size(0)):  # Normalize
                    feature.data[j] = feature.data[j] / feature.data[j].norm()
                feature = feature.unsqueeze(2)  # (batch_size, feature_size, 1)
                # means = torch.stack([exemplar_means[cls] for cls in self.old_labels])  # (n_classes, feature_size)
                means = torch.stack([exemplar_means[cls] for cls in range(10*len(test_loaders))])  # (n_classes, feature_size)



                # old ncm
                means = torch.stack([means] * batch_x.size(0))  # (batch_size, n_classes, feature_size)
                means = means.transpose(1, 2)
                feature = feature.expand_as(means)  # (batch_size, feature_size, n_classes)
                dists = (feature - means).pow(2).sum(1).squeeze()  # (batch_size, n_classes)
                _, pred_label = dists.min(1)
                # may be faster
                # feature = feature.squeeze(2).T
                # _, preds = torch.matmul(means, feature).max(0)
                correct_cnt = (np.array(range(10*len(test_loaders)))[
                                   pred_label.tolist()] == batch_y.cpu().numpy()).sum().item() / batch_y.size(0)

                acc.update(correct_cnt, batch_y.size(0))
            acc_array[task] = acc.avg()
    print(acc_array)
    return acc_array