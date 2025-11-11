# FactorVAE: Improved Disentanglement and Performance in Generative Models

## Project Overview
FactorVAE is a novel approach to enhance disentanglement in generative models by introducing a Total Correlation (TC) penalty. This project implements the FactorVAE model as described in the paper "FactorVAE: Improved Disentanglement and Performance in Generative Models".

## Installation Instructions
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd FactorVAE
   ```
2. **Set up the environment**:
   - Ensure you have Python 3.8+ installed.
   - Install the required packages:
     ```bash
     pip install -r requirements.txt
     ```
   - Ensure CUDA drivers are up to date for GPU support.

## Usage Guide
- **Training the Model**:
  Run the main script to start training:
  ```bash
  python src/main.py --config config/config.yaml
  ```
- **Evaluating the Model**:
  Use the experiment script to evaluate and visualize results:
  ```bash
  python src/experiment.py --config config/config.yaml
  ```

## Configuration Details
- The configuration file `config/config.yaml` contains hyperparameters and settings for the model. Modify this file to change datasets, model parameters, and training settings.

## Results Interpretation
- Results are stored in the `results/` directory, organized by experiment name.
- Check `logs.txt` for training logs and `figures/` for visualizations of the results.

## Contact Information
For questions or contributions, please contact the project maintainers at [email@example.com].

---
This README provides a comprehensive guide to using the FactorVAE implementation. For more detailed information, refer to the comments and documentation within the code files.