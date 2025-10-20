from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 0
    num_runs: int = 1

    # Data
    dataset: str = "cifar-100"  # Options: "cifar-100", "tiny-imagenet", "mini-imagenet", "food-101"
    streaming_data_batch_size: int = 10
    num_tasks: int = 10  # 20 for tiny-imagenet

    # Main model config
    contrastive_learning: bool = True  # If False, use regular ResNet
    use_augmentations: bool = True
    use_memory: bool = True

    # Memory
    buffer_size: int = 500
    main_task_memory_batch_size: int = 100  # Samples to replay per batch

    # SSD config
    summarized_per_class: int = 5 # Examples per class for gradient & relationship matching
    tau: int = 6  # Each tau batches, one summarizing step is done
    ms_diff_mc_batch_size: int = 10
    relationship_matching_distance: str = "l1"
    gradient_matching_distance: str = "sse"
    incremental_softmax: bool = True  # Use same summarizing model across tasks
    start_summarization_when_mc_and_queue_full: bool = True
    gamma: int = 1
    queue_size: int = 64
    number_of_chunks_for_summarizing_model: int = 2
    summarization_iterations: int = 1

config = Config()

ssd_config = Config()

scr_config = Config(summarized_per_class=0)

er_config = Config(summarized_per_class=0,
                   contrastive_learning=False)

finetune_config = Config(summarized_per_class=0,
                         contrastive_learning=False,
                         use_memory=False)

