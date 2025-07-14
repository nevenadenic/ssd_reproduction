import copy
import os

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from config import Config
from memory.DynamicMemoryBuffer import DynamicMemoryBuffer
from scr_training.ResNets import SupConResNet, Reduced_ResNet18
from scr_training.SupervisedContrastiveLoss import SupConLoss
from sequential_dataset.SequentialDataset import SequentialDataset
from summarizing.DifferentiableAugmentation import DifferentiableAugmentation
from summarizing.SummarizingModel import SummarizingModel

from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale

from scr_training.NCM import NearestClassMeanClassifier
from memory.CurrentTaskQueue import CurrentTaskQueue
import matplotlib.pyplot as plt
import numpy as np

from scr_training.njihov_evaluate import njihov_evaluate

def train(config, plot_graph=False):
    seed = config.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Initializations

    sequential_dataset = SequentialDataset(total_num_tasks=config.num_tasks,
                                           batch_size=config.streaming_data_batch_size,
                                           dataset=config.dataset)
    augmentation = nn.Sequential(
                RandomResizedCrop(size=sequential_dataset.get_image_width_and_height(), scale=(0.2, 1.)),
                RandomHorizontalFlip(),
                ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                RandomGrayscale(p=0.2)
            )

    total_num_classes = sequential_dataset.get_total_num_classes()
    num_classes_per_task = total_num_classes // config.num_tasks

    if config.summarized_per_class == 1:
        lr_for_summarizing_images_update = 2e-4
    elif config.summarized_per_class == 5:
        lr_for_summarizing_images_update = 1e-3 # 1e-3 # 1e-3? kako su sad ovolike brojke?? something is off here
    elif config.summarized_per_class == 10:
        lr_for_summarizing_images_update = 4e-3
    else:
        lr_for_summarizing_images_update = 3e-3 # source: I made it up


    memory = DynamicMemoryBuffer(total_capacity=config.buffer_size,
                                 number_of_classes=total_num_classes,
                                 summarized_per_class=config.summarized_per_class)

    resnet_dimension = 160 if config.dataset == "cifar-100" else 640 # ovo moze i treba da se racuna drugacije, al neka ga za sad ovako

    if config.contrastive_learning:
        main_model = SupConResNet(resnet_dimension).to(device) # because of image sizes
        ncm_classifier = NearestClassMeanClassifier(num_classes=total_num_classes, feature_dim=resnet_dimension)
        main_model_criterion = SupConLoss(temperature=0.07) # da li mozda ovo 0.07?
        main_model_optimizer = optim.SGD(main_model.parameters(), lr=0.1)

    else:
        main_model = Reduced_ResNet18(nclasses=total_num_classes, dim_in=resnet_dimension).to(device)
        main_model_criterion = nn.CrossEntropyLoss()
        main_model_optimizer = optim.SGD(main_model.parameters(), lr=0.01, momentum=0.9)

    if config.summarized_per_class:
        summarizing_model = SummarizingModel(input_image_width_and_height=sequential_dataset.get_image_width_and_height()).to(device)
        summarizing_model_criterion = nn.CrossEntropyLoss()
        summarizing_model_optimizer = optim.SGD(summarizing_model.parameters(), lr=0.01, momentum=0.9)
        summarizing_model.train()



    main_model.train()
    class_mapping = torch.full((total_num_classes,), -1, dtype=torch.long).to(device)
    reverse_class_mapping = torch.full((total_num_classes,), -1, dtype=torch.long).to(device)
    curr_class = 0
    accuracy_through_time = [[] for _ in range(config.num_tasks)]

    # Sequential processing: 10 classes per task (10 or 20 tasks total)
    for task_number in range(config.num_tasks):
        train_loader = sequential_dataset.get_train_dataloader_for_task(task_number)

        current_task_queue = CurrentTaskQueue(config.queue_size)
        main_model.train()

        total_loss = 0

        this_task_assigned_optimizer_to_images = False

        for batch_idx, (inputs, labels) in enumerate(train_loader):

            if batch_idx % 100 == 0:
                print(f"Started batch {batch_idx}...")


            inputs, labels = inputs.to(device), labels.to(device)

            # 0) Mapping labels

            initial_unique_class_labels = torch.unique(labels)
            num_new_classes_in_batch = 0
            for initial_unique_class_label in initial_unique_class_labels:
                if class_mapping[initial_unique_class_label] == -1:
                    class_mapping[initial_unique_class_label] = curr_class
                    reverse_class_mapping[curr_class] = initial_unique_class_label

                    curr_class += 1
                    num_new_classes_in_batch += 1

            labels = class_mapping[labels]
            labels_from_this_batch = torch.unique(labels)

            # 1) Main model data

            main_model_inputs = inputs
            main_model_labels = labels

            if config.use_memory and memory.is_not_empty():
                memory_inputs, memory_labels = memory.sample_batch(config.main_task_memory_batch_size)
                # memory_inputs = memory_inputs.detach()
                main_model_inputs = torch.cat([main_model_inputs, memory_inputs], dim=0)
                main_model_labels = torch.cat([main_model_labels, memory_labels], dim=0)  # Labels remain the same

                augmented_main_model_inputs = augmentation(main_model_inputs)

                # 2) Main model step
                if config.contrastive_learning:
                    embeddings = torch.cat([main_model(main_model_inputs).unsqueeze(1),
                                                   main_model(augmented_main_model_inputs).unsqueeze(1)], dim=1)
                else:
                    embeddings = torch.cat([main_model(main_model_inputs),
                                                   main_model(augmented_main_model_inputs)], dim=1)

                loss = main_model_criterion(embeddings, main_model_labels)

                main_model_optimizer.zero_grad()
                loss.backward()
                main_model_optimizer.step()

                total_loss += loss.item()


            if config.summarized_per_class==0:
                memory.add_batch(inputs, labels)
                continue

            # 3) Summarizing model update

            if config.incremental_softmax:
                if num_new_classes_in_batch > 0:
                    summarizing_model.add_classes(num_new_classes_in_batch)
                    summarizing_model.to(device)
                    summarizing_model_optimizer = optim.SGD(summarizing_model.parameters(), lr=0.01, momentum=0.9)
                    summarizing_model.train()
            else:
                if batch_idx == 0:
                    summarizing_model = SummarizingModel(sequential_dataset.get_image_width_and_height()).to(device)
                    summarizing_model.add_classes((task_number + 1) * num_classes_per_task)
                    summarizing_model = summarizing_model.to(device)
                    summarizing_model_optimizer = optim.SGD(summarizing_model.parameters(), lr=0.01, momentum=0.9)
                    summarizing_model.train()


            # 4) Potential summarizing model step

            if batch_idx % config.tau == 5:

                if memory.is_full_mc(labels_from_this_batch[0]):

                    if not this_task_assigned_optimizer_to_images:

                        this_task_Ms_images = []
                        this_task_Ms_labels = []

                        for label in labels_from_this_batch:
                            inputs_Mc, labels_Mc = memory.get_mc(label)

                            this_task_Ms_images.append(inputs_Mc)
                            this_task_Ms_labels.append(labels_Mc)

                        this_task_Ms_images = copy.deepcopy(torch.cat(this_task_Ms_images, dim=0)).requires_grad_()
                        this_task_Ms_labels = torch.cat(this_task_Ms_labels)

                        summarizing_image_optimizer = optim.SGD([this_task_Ms_images],
                                                                lr=lr_for_summarizing_images_update,
                                                                momentum=0.9)

                        this_task_assigned_optimizer_to_images = True
                else:
                    continue



                for label in labels_from_this_batch:
                    if config.start_summarization_when_mc_and_queue_full and not memory.is_full_mc(label):
                        continue

                    mask = (labels == label)
                    inputs_Bc, labels_Bc = inputs[mask], labels[mask]

                    queue_inputs_c, queue_labels_c = current_task_queue.get_class(label)
                    if config.start_summarization_when_mc_and_queue_full and len(queue_labels_c) < config.queue_size:
                        continue
                    elif queue_labels_c is not None:
                        inputs_Bc = torch.cat([inputs_Bc, queue_inputs_c], dim=0)
                        labels_Bc = torch.cat([labels_Bc, queue_labels_c], dim=0)


                    for iteration in range(config.summarization_iterations):
                        #
                        # inputs_Mc, labels_Mc = memory.get_mc(label)
                        #
                        # if inputs_Mc is not None:
                        #     inputs_Mc = inputs_Mc.detach().requires_grad_(True)
                        # else:
                        #     continue

                        inputs_Mc = this_task_Ms_images[this_task_Ms_labels == label]
                        labels_Mc = this_task_Ms_labels[this_task_Ms_labels == label]

                        cloned_inputs_Mc = inputs_Mc.clone().detach()




                        # inputs_Bc = inputs_Bc.detach()



                        inputs_Ms_diff_Mc, labels_Ms_diff_Mc = memory.get_ms_diff_mc(label_to_discard=label,
                                                                                     current_task=task_number,
                                                                                     batch_size=config.ms_diff_mc_batch_size)

                        # summarizing_image_optimizer = optim.SGD([inputs_Mc],
                        #                                         lr=lr_for_summarizing_images_update,
                        #                                         momentum=0.9)  # Only updating Mc

                        diff_aug = DifferentiableAugmentation(strategy='color_crop', batch=False)
                        match_aug = transforms.Compose([diff_aug])
                        inputs_aug = match_aug(torch.cat((inputs_Bc, inputs_Mc), dim=0))
                        augmented_inputs_Bc = inputs_aug[:len(inputs_Bc)]
                        augmented_inputs_Mc = inputs_aug[len(inputs_Bc):]

                        if inputs_Ms_diff_Mc is not None:
                            # inputs_Ms_diff_Mc.detach()
                            with torch.no_grad():
                                emb_Ms_diff_Mc = summarizing_model.embedding(inputs_Ms_diff_Mc)
                        _, out_Bc = summarizing_model(augmented_inputs_Bc)
                        with torch.no_grad():
                            emb_Bc = summarizing_model.embedding(augmented_inputs_Bc)

                        emb_Mc, out_Mc = summarizing_model(augmented_inputs_Mc)


                        # Gradient-based loss
                        criterion_gradient_loss = nn.CrossEntropyLoss()

                        L_Bc = criterion_gradient_loss(out_Bc, labels_Bc)
                        grads_Bc = torch.autograd.grad(L_Bc, summarizing_model.parameters())
                        grads_Bc = list((_.detach().clone() for _ in grads_Bc))

                        L_Mc = criterion_gradient_loss(out_Mc, labels_Mc)
                        grads_Mc = torch.autograd.grad(L_Mc, summarizing_model.parameters(), create_graph=True)

                        if config.gradient_matching_distance=="sse":
                            grad_diffs = [(gwr - gws).pow(2).sum() for gwr, gws in zip(grads_Bc, grads_Mc) if gwr.ndim > 2]
                        else:
                            grad_diffs = [torch.norm(g1 - g2, p=2) for g1, g2 in zip(grads_Mc, grads_Bc)]

                        gradient_matching_loss = sum(grad_diffs)


                        if inputs_Ms_diff_Mc is not None:
                            emb_Ms_diff_Mc = F.normalize(emb_Ms_diff_Mc, dim=1)
                            emb_Mc = F.normalize(emb_Mc, dim=1)
                            emb_Bc = F.normalize(emb_Bc, dim=1)


                            # Relationship matching loss
                            mean_emb_Mc = emb_Mc.mean(dim=0, keepdim=True)
                            mean_emb_Bc = emb_Bc.mean(dim=0, keepdim=True)
                            emb_d_Bc = torch.norm(mean_emb_Bc - emb_Ms_diff_Mc, p=2, dim=1)
                            emb_d_Mc = torch.norm(mean_emb_Mc - emb_Ms_diff_Mc, p=2, dim=1)

                            if config.relationship_matching_distance=="l1":
                                relationship_matching_loss = torch.norm(emb_d_Bc - emb_d_Mc, p=1)
                            else:
                                relationship_matching_loss = torch.norm(emb_d_Bc - emb_d_Mc, p=2)

                        else:
                            relationship_matching_loss = 0

                        summarizing_image_loss = config.gamma * relationship_matching_loss + gradient_matching_loss

                        # Compute gradients of final loss w.r.t Mc and update
                        summarizing_image_optimizer.zero_grad()
                        summarizing_image_loss.backward()
                        summarizing_image_optimizer.step()

                        img_new = this_task_Ms_images[this_task_Ms_labels == label]

                        close = torch.abs(cloned_inputs_Mc - img_new) <= 0.0001
                        # all_close = close.all()
                        # print("All elements close:", all_close)
                        # exact_match = torch.equal(cloned_inputs_Mc, img_new)
                        # print("Exact match:", exact_match)


                        memory.update_mc(new_tensors=img_new.detach(), label=label)

                        # memory.update_mc(new_tensors=inputs_Mc.detach(), label=label)


            memory.add_batch(inputs, labels)
            current_task_queue.add(inputs, labels)

            mo_inputs, mo_labels = memory.sample_mo_for_summarizing_model_training(10)
            if mo_labels is not None:

                inputs = torch.cat([inputs, mo_inputs], dim=0)
                labels = torch.cat([labels, mo_labels], dim=0)

            # PART 5: SUMMARIZING MODEL UPDATE
            indices = torch.randperm(len(inputs))

            inputs = inputs[indices]
            labels = labels[indices]

            input_groups = inputs.chunk(config.number_of_chunks_for_summarizing_model)
            label_groups = labels.chunk(config.number_of_chunks_for_summarizing_model)

            for i, (batch_inputs, batch_labels) in enumerate(zip(input_groups, label_groups)):
                _, batch_outputs = summarizing_model(batch_inputs)
                loss = summarizing_model_criterion(batch_outputs, batch_labels)

                summarizing_model_optimizer.zero_grad()
                loss.backward()
                summarizing_model_optimizer.step()

        print(f"Main Model Loss: {total_loss / len(train_loader):.4f}")


        main_model.eval()

        if config.contrastive_learning:
            ncm_classifier.update_means(memory, task_number, main_model)


        test_loaders = []
        for past_task_number in range(task_number + 1):  # past tasks, including this one
            test_loader = sequential_dataset.get_test_loader_for_task(past_task_number)

            test_loaders.append(test_loader)

            correct, total = 0, 0
            with torch.no_grad():
                for idx, (images, labels) in enumerate(test_loader):
                    labels = class_mapping[labels]
                    images, labels = images.to(device), labels.to(device)

                    if config.contrastive_learning:
                        embeddings = main_model.features(images)
                        predictions = ncm_classifier(embeddings)
                    else:
                        outputs = main_model(images)
                        predictions = outputs.argmax(dim=1)

                    correct += predictions.eq(labels).sum().item()
                    total += labels.size(0)
                test_acc = 100 * correct / total
                accuracy_through_time[past_task_number].append(test_acc)
                print(f"Accuracy on {past_task_number} after {task_number}: {test_acc}%")
        #
        # test_loaders = []
        # for past_task_number in range(task_number + 1):
        #     test_loader = sequential_dataset.get_test_loader_for_task(past_task_number)
        #     test_loaders.append(test_loader)
        #
        # njihov_evaluate(test_loaders=test_loaders, model=main_model, buffer=memory, class_mapping=class_mapping)


    # Final evaluation

    final_accuracies = [accuracy_through_time[task][-1] for task in range(config.num_tasks)]

    test_acc = sum(final_accuracies)/len(final_accuracies)
    with open("results.txt", "a") as file:
        file.write(f"\n\n\n\nDataset: {config.dataset}, buffer size: {config.buffer_size}, SSD: {True if config.summarized_per_class > 0 else False}\n")
        file.write(f"Test Accuracy On All Test Data After {config.num_tasks} Tasks: {test_acc:.2f}%")

    print(f"\n\n\n\nTest Accuracy On All Test Data After {config.num_tasks} Tasks: {test_acc:.2f}%\n\n")

    task_list = [task for task in range(config.num_tasks)]
    for task in range(config.num_tasks):
        accuracy_through_time[task] = [None] * (len(task_list) - len(accuracy_through_time[task])) + accuracy_through_time[task]

        if plot_graph:
            plt.plot(task_list, accuracy_through_time[task], marker='o', label=f'Task {task}')

    from PIL import Image

    for i, (image, label) in enumerate(memory.buffer):
        # Ensure it's on CPU and convert to NumPy
        img_array = image.to("cpu").numpy()

        # If tensor is in CHW format, convert to HWC
        if img_array.ndim == 3 and img_array.shape[0] in [1, 3]:  # assuming CxHxW
            img_array = img_array.transpose(1, 2, 0)  # CHW to HWC

        # Convert to uint8 if not already
        if img_array.dtype != 'uint8':
            img_array = (img_array * 255).clip(0, 255).astype('uint8')

        # Create output directory
        output_dir = f"summarized_images/class_{label}"
        os.makedirs(output_dir, exist_ok=True)

        # Save image
        img = Image.fromarray(img_array)
        img.save(os.path.join(output_dir, f"image_{i}.png"))





    if plot_graph:
        plt.xlabel('Time')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy through time on {config.dataset}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"plots/{config.dataset}/{config.buffer_size}/plot_{config.seed}_{random.randint(1, 1_000_000)}.png")
        # plt.show()
        plt.close()

    return test_acc


if __name__ == "__main__":
    ssd_config = Config()
    train(ssd_config)