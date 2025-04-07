# A Reproduction of "Summarizing Stream Data for Memory-Constrained Online Continual Learning"

Online class-incremental continual learning is a challenging setting in which models must adapt 
to tasks consisting of new classes over time without forgetting previously learned ones. In this work, we reproduce the 
results of **SCR** [(Mai et al., 2021)](https://arxiv.org/pdf/2103.13885), a strong baseline that leverages contrastive learning 
along with memory replay. Next, we implement **SSD** [(Gu et al., 2024)](https://arxiv.org/pdf/2305.16645), which enhances SCR using 
dataset distillation techniques to generate compact memory representations. While we observe consistent 
improvements over the base SCR method, some experiments currently show smaller performance gains than those reported
by Gu et al., 2024.


We design four datasets to experiment on (by splitting them into tasks of 10 classes): 
- Sequential CIFAR-100 (CIFAR-100 is available in torch datasets)
- Sequential Mini-ImageNet (download Mini-ImageNet from [here](https://www.kaggle.com/whitemoon/miniimagenet) and place it in `/data/mini-imagenet`)
- Sequential Tiny-ImageNet (download Tiny-ImageNet from [here](https://drive.google.com/file/d/1xF-_ew39BzhWG0OyOrzlwNEBP_89YJ-z/view?usp=sharing) and place in `/data/tiny-imagenet-200`)
- Sequential Food-101 (Food-101 is available in torch datasets)

After each task ends, the model is evaluated on all tasks up to the current one:

![Plot](/plots/mini-imagenet/500/plot_519_303902.png)