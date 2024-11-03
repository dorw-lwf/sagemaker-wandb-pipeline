import tensorflow as tf
import argparse
import os
import json
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


def create_model():
    # Load pre-trained MobileNetV2 as base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(160, 160, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze the pre-trained weights

    # Build the complete model architecture
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),  # Convert features to vector
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification head
    ])

    # Compile model with binary classification settings
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    # Install and import wandb, then import WandbCallback
    wandb, WandbMetricsLogger = install_wandb()
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default="/opt/ml/model")  # Default model directory
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    args, _ = parser.parse_known_args()

    wandb.login(key="cbae9bb59542d60f1f9dbdb9ff89599479e24b71")
    # Read the W&B run ID from the file
    run_id_file = "/opt/ml/input/data/run_id/run_id.txt"
    with open(run_id_file, "r") as f:
        preprocessing_run_id = f.read().strip()

    # Initialize W&B
    wandb.init(project="sagemaker-demo", job_type="training", id=preprocessing_run_id, resume="allow")
    wandb.config.update({
        "epochs": args.epochs,
        "batch_size": args.batch_size
    })
    # Retrieve the dataset artifact
    model_use_case_id = "cifar10"
    version = "latest"
    name = "{}:{}".format("{}_dataset".format(model_use_case_id), version)
    artifact = wandb.use_artifact(artifact_or_name=name)

    # Get specific content from the dataframe
    train_table = artifact.get("train_table")
    x_train = train_table.get_column("x_train", convert_to="numpy")
    y_train = train_table.get_column("y_train", convert_to="numpy")
    # Get specific content from the dataframe
    eval_table = artifact.get("eval_table")
    x_test = eval_table.get_column("x_test", convert_to="numpy")
    y_test = eval_table.get_column("y_test", convert_to="numpy")

    # Create and train model
    model = create_model()
    history = model.fit(
        x_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test, y_test),
        callbacks=[WandbMetricsLogger(log_freq=10)]  # Integrate W&B callback to log metrics
    )

    # Evaluate model
    eval_results = model.evaluate(x_test, y_test)
    wandb.log({
        "test_loss": eval_results[0],
        "test_accuracy": eval_results[1]
    })

    # Save evaluation metrics to /opt/ml/model/evaluation/evaluation.json
    evaluation_dir = os.path.join(args.model_dir, 'evaluation')
    os.makedirs(evaluation_dir, exist_ok=True)

    metrics = {
        'test_loss': float(eval_results[0]),
        'test_accuracy': float(eval_results[1])
    }
    with open(os.path.join(evaluation_dir, 'evaluation.json'), 'w') as f:
        json.dump(metrics, f)

    # Save the model in the specified model directory
    model_path = os.path.join(args.model_dir, 'model.keras')
    model.save(model_path)
    registered_model_name = "cifar10-MobileNetV2"
    wandb.link_model(path=model_path, registered_model_name=registered_model_name)

    # Finish the W&B run
    wandb.finish()