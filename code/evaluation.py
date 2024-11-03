import tensorflow as tf
import numpy as np
import json
import os
import tarfile
import sys
import subprocess


# Function to install wandb if not present
def install_wandb():
    try:
        import wandb
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "wandb"])
        import wandb
        from wandb.integration.keras import WandbMetricsLogger
    return wandb, WandbMetricsLogger


def evaluate_model(model, x_test, y_test):
    # Evaluate model on test data
    loss, accuracy = model.evaluate(x_test, y_test)
    metrics = {
        'test_loss': float(loss),
        'test_accuracy': float(accuracy)
    }
    return metrics


if __name__ == "__main__":
    
    # Install and import wandb, then import WandbCallback
    wandb, WandbMetricsLogger = install_wandb()
    # Load test data
    x_test = np.load("/opt/ml/processing/test/x_test.npy")
    y_test = np.load("/opt/ml/processing/test/y_test.npy")

    # Ensure x_test has the correct shape
    print(f"Original x_test shape: {x_test.shape}")
    if x_test.ndim == 5 and x_test.shape[0] == 1:
        x_test = np.squeeze(x_test, axis=0)
    print(f"Reshaped x_test shape: {x_test.shape}")

    # Ensure y_test has the correct shape
    print(f"Original y_test shape: {y_test.shape}")
    if y_test.ndim > 2:
        y_test = np.squeeze(y_test)  # Squeeze extra dimensions if present
    print(f"Reshaped y_test shape: {y_test.shape}")

    # Extract the model.tar.gz file
    model_tar_path = "/opt/ml/processing/model/model.tar.gz"
    extract_path = "/opt/ml/processing/model"
    with tarfile.open(model_tar_path, "r:gz") as tar:
        tar.extractall(path=extract_path)

    # Load the model from the extracted SavedModel directory
    model = tf.keras.models.load_model(os.path.join(extract_path, "model.keras"))
    
    # Evaluate the model
    metrics = evaluate_model(model, x_test, y_test)

    # Save evaluation results to a JSON file
    evaluation_output_path = "/opt/ml/processing/evaluation/evaluation.json"
    os.makedirs(os.path.dirname(evaluation_output_path), exist_ok=True)
    with open(evaluation_output_path, "w") as f:
        json.dump(metrics, f)
