import subprocess
import os
import shutil
import tarfile

model_repo_path = r"D:\E\Copy\PyCharm\Hometask\ml_hard_models_2025\hw3\model_repository"
env_name = "triton_cv_env310"
python_version = "3.10"  # Or whatever Python version you need

def create_environment_yml(model_repo_path, env_name, python_version):
    """Creates the environment.yml file."""
    environment_yml_path = os.path.join(model_repo_path, "environment.yml")

    # Updated environment.yml content, ensuring it has tritonclient and explicitly includes pytorch.
    environment_yml_content = f"""
name: {env_name}
channels:
  - defaults
dependencies:
  - python={python_version}
  - numpy
  - pillow
  - pytorch #Added pytorch because torchvision depends on it
  - torchvision
  - pip:
      - tritonclient[all]  # Add tritonclient (important for interacting with the server!)
"""

    with open(environment_yml_path, "w") as f:
        f.write(environment_yml_content)
    print(f"Created environment.yml at {environment_yml_path}")

def create_conda_env(env_name, model_repo_path):
    """
    Creates a conda environment from environment.yml
    """
    env_file_path = os.path.join(model_repo_path, "environment.yml")
    try:
        subprocess.run(["conda", "env", "create", "-f", env_file_path, "-n", env_name], check=True, capture_output=True, text=True)
        print(f"Successfully created conda environment {env_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating conda environment:\n{e.stderr}")
        raise

def create_tar_gz(env_name, model_repo_path):
    """
    Creates a tar.gz archive of the conda environment.  Crucially, it copies the env yml to the environment as well, which Triton requires.

    Args:
        env_name: The name of the conda environment.
        model_repo_path: The path to the model repository.
    """
    tar_filename = os.path.join(model_repo_path, f"{env_name}.tar.gz")
    conda_env_path = os.path.join(os.environ['CONDA_PREFIX'], 'envs', env_name) # Locate conda env

    # Copy environment.yml into the environment directory (required by triton)
    try:
        shutil.copy(os.path.join(model_repo_path, "environment.yml"), conda_env_path)
    except FileNotFoundError:
        print("environment.yml not found.  Continuing without it.")

    with tarfile.open(tar_filename, "w:gz") as tar:
        tar.add(conda_env_path, arcname=env_name)  # Use arcname to preserve folder structure

    print(f"Successfully created archive: {tar_filename}")


if __name__ == "__main__":
    try:
        create_environment_yml(model_repo_path, env_name, python_version)  # Create environment.yml first
        create_conda_env(env_name, model_repo_path)  # Create the conda environment
        create_tar_gz(env_name, model_repo_path)  # Create the archive
        print("Conda environment creation and archiving complete.")

    except Exception as e:
        print(f"An error occurred: {e}")