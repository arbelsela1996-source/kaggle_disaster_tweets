import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    # Model and tokenizer
    model_name: str = "bert-base-uncased"
    max_length: int = 128

    # Training hyperparameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 3

    # Data and output paths
    data_dir: str = os.path.join(PROJECT_ROOT, "data")
    train_file: str = os.path.join(data_dir, "train.csv")
    test_file: str = os.path.join(data_dir, "test.csv")

    outputs_dir: str = os.path.join(PROJECT_ROOT, "outputs")
    model_dir: str = os.path.join(outputs_dir, "model")
    plots_dir: str = os.path.join(outputs_dir, "plots")

    # Misc
    random_seed: int = 42
    val_size: float = 0.1


config = Config()

