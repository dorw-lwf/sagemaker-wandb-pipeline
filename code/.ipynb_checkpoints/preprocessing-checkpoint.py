import tensorflow as tf
import numpy as np
import os
import sys
import subprocess


# Function to install wandb if not present
def install_wandb():
    try:
        import wandb
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    finally:
        import wandb
    return wandb


# Install and import wandb
wandb = install_wandb()


def prepare_data():
    # Load CIFAR-10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Flatten y_train and y_test to make them 1D
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Filter cats and dogs (classes 3 and 5)
    train_mask = (y_train == 3) | (y_train == 5)
    test_mask = (y_test == 3) | (y_test == 5)

    x_train, y_train = x_train[train_mask], y_train[train_mask]
    x_test, y_test = x_test[test_mask], y_test[test_mask]

    # Convert to binary classification (1 for class 5, 0 for class 3)
    y_train = (y_train == 5).astype(int)
    y_test = (y_test == 5).astype(int)

    # Resize and normalize
    x_train = tf.image.resize(x_train, (160, 160)).numpy()
    x_test = tf.image.resize(x_test, (160, 160)).numpy()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":

    wandb.login(key="cbae9bb59542d60f1f9dbdb9ff89599479e24b71")
    # Initialize W&B
    wandb.init(project="sagemaker-demo", job_type="preprocessing")

    # Get the W&B run ID
    preprocessing_run_id = wandb.run.id

    # Save the run ID to a file in the designated output directory
    with open("/opt/ml/processing/output/run_id.txt", "w") as f:
        f.write(preprocessing_run_id)

    # Prepare data
    x_train, y_train, x_test, y_test = prepare_data()

    # Save processed datasets locally
    os.makedirs("/opt/ml/processing/train", exist_ok=True)
    os.makedirs("/opt/ml/processing/test", exist_ok=True)
    np.save("/opt/ml/processing/train/x_train.npy", x_train)
    np.save("/opt/ml/processing/train/y_train.npy", y_train)
    np.save("/opt/ml/processing/test/x_test.npy", x_test)
    np.save("/opt/ml/processing/test/y_test.npy", y_test)

    # Create W&B Table for training data
    train_table = wandb.Table(data=[], columns=[])
    train_table.add_column("x_train", x_train)
    train_table.add_column("y_train", y_train)
    train_table.add_computed_columns(lambda ndx, row: {"img": wandb.Image(row["x_train"])})

    # Create W&B Table for eval data
    eval_table = wandb.Table(data=[], columns=[])
    eval_table.add_column("x_test", x_test)
    eval_table.add_column("y_test", y_test)
    eval_table.add_computed_columns(lambda ndx, row: {"img": wandb.Image(row["x_test"])})

    model_use_case_id = "cifar10"
    # Create an artifact object
    artifact_name = "{}_dataset".format(model_use_case_id)
    artifact = wandb.Artifact(name=artifact_name, type="dataset")

    # Add wandb.WBValue obj to the artifact.
    artifact.add(train_table, "train_table")
    artifact.add(eval_table, "eval_table")

    # Persist any changes made to the artifact.
    artifact.save()

    wandb.finish()