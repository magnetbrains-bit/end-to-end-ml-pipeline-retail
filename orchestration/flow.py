# orchestration/flow.py

from prefect import flow, task
import subprocess
import os
import json
import sys

# Get the absolute path of the project's root directory.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_script(script_name: str):
    """A helper function to run a script using the correct venv Python."""
    script_path = os.path.join(project_root, "scripts", script_name)
    print(f"--- Running Script: {script_path} ---")
    
    # Construct the full path to the Python executable inside the venv
    if sys.platform == "win32":
        python_executable = os.path.join(project_root, "venv", "scripts", "python.exe")
    else:
        python_executable = os.path.join(project_root, "venv", "bin", "python")

    # Run the script using the full path to the python executable
    # check=True will automatically raise an error if the script fails
    subprocess.run([python_executable, script_path], check=True)

@task(name="Process Data Task")
def process_data():
    """Task to run the data processing script."""
    run_script("process_data.py")

@task(name="Train Model Task")
def train_model():
    """Task to run the model training script."""
    run_script("train_model.py")
    
    # If the script succeeds, return the path to the metrics file for the next task.
    metrics_path = os.path.join(project_root, "app", "metrics.json")
    return metrics_path

@task(name="Evaluate Model Task")
def evaluate_model(metrics_path: str):
    """Task to evaluate the model based on its performance metric."""
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    roc_auc = metrics['validation_roc_auc']
    
    print(f"--- Evaluating Model ---")
    print(f"Current model validation ROC AUC: {roc_auc:.4f}")
    
    if roc_auc > 0.75:
        print("✅ Model performance is good. Promoting to 'production'.")
        return "Model Promoted"
    else:
        print("❌ Model performance did not meet the threshold. Not promoting.")
        return "Model Not Promoted"

@flow(name="ML Training Flow")
def ml_training_flow():
    """The main flow to orchestrate the ML model training pipeline."""
    print("--- Starting ML Training Flow ---")
    
    # Define the dependency graph: process -> train -> evaluate
    process_data_task = process_data()
    
    # wait_for ensures this task won't start until process_data is complete.
    train_model_task = train_model(wait_for=[process_data_task])
    
    # This task waits for the training to finish.
    evaluation_result = evaluate_model(wait_for=[train_model_task], metrics_path=train_model_task)
    
    print(f"--- Flow finished with status: {evaluation_result} ---")

# The if __name__ == "__main__": block has been removed.
# The flow is now only intended to be run by Prefect, not directly as a script.