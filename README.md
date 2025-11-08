
# Logger Module README

## Overview
The LoggerFormatter class is a comprehensive logging utility designed for experiment tracking and data management. It automatically creates timestamped directories, saves hyperparameters, handles figure exports, and manages large variable storage.

## Installation
```python
from Logger import LoggerFormatter
```

## Basic Usage

### Initialize LoggerFormatter
```python
# Basic initialization
logger = LoggerFormatter(project_name="my_experiment", base_path="./logs")

# Default initialization
logger = LoggerFormatter()  # Uses project_name="exp", base_path="./logs"
```

When initialized, the LoggerFormatter will:
- Create a timestamped directory (format: `YYYY-MM-DD-HH-MM-SS-project_name`)
- Automatically save the source file (Python script or Jupyter notebook)
- Print initialization details

#### For Jupyter Notebook Users

**Important**: The logger saves the notebook file from disk, not from memory. To ensure the latest version is saved:

**Manual Save Before Running**:
```python
# Step 1: Save your notebook (Ctrl+S or Cmd+S)
# Step 2: Run your code
logger = LoggerFormatter(project_name="my_experiment")
```
   
## Core Functions

### 1. `seed_everything(seed=42)`
Sets random seeds for reproducibility across different libraries.

**Parameters:**
- `seed` (int): Random seed value, default is 42

**Returns:**
- The seed value used

**Example:**
```python
# Set seed for reproducibility
logger.seed_everything(2024)
```

**Supported libraries:**
- Python's `random`
- NumPy (if installed)
- PyTorch (if installed, includes CUDA seeds and deterministic settings)

### 2. `save_hyperparameters(save_filename="hyperparameters.json")`
Automatically detects and saves all global variables starting with uppercase letters as hyperparameters.

**Parameters:**
- `save_filename` (str): Name of the JSON file to save, default is "hyperparameters.json"

**Returns:**
- Dictionary of saved hyperparameters

**Example:**
```python
# Define hyperparameters (must start with uppercase)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
MODEL_NAME = "ResNet50"

# Save them
saved_params = logger.save_hyperparameters()
```

**What gets saved:**
- Variables starting with uppercase letters
- Basic types: int, float, str, bool, None
- Lists/tuples of basic types
- Dictionaries with basic type keys and values
- Complex objects are converted to strings

**What gets ignored:**
- Variables starting with lowercase or underscore
- Modules, classes, functions, methods
- Jupyter special variables (In, Out)

### 3. `save_fig(fig_name, dpi=300)`
Saves matplotlib figures to the log directory with high quality.

**Parameters:**
- `fig_name` (str): **Required**. Name of the figure file
- `dpi` (int): Resolution, default is 300. **Must be >= 300**

**Returns:**
- Path to the saved figure

**Example:**
```python
import matplotlib.pyplot as plt
import numpy as np

# Create a plot
plt.figure(figsize=(8, 6))
x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))
plt.title("Sine Wave")

# Save the figure
logger.save_fig("sine_wave.png", dpi=300)
plt.close()
```

**Supported formats:**
- Automatically adds `.png` if no extension provided
- Supports: `.png`, `.jpg`, `.jpeg`, `.pdf`, `.svg`

**Error cases:**
```python
# These will raise errors:
logger.save_fig("")  # ValueError: Figure name must be provided
logger.save_fig("plot", dpi=100)  # ValueError: DPI must be at least 300
```

### 4. `save_large_vars(var_obj, var_name=None)`
Saves large variables (arrays, models, datasets) to a dedicated `large_vars` folder using pickle.

**Parameters:**
- `var_obj`: The variable object to save
- `var_name` (str, optional): Name for the saved file. If None, attempts to auto-detect

**Returns:**
- Path to the saved file

**Example:**
```python
import numpy as np

# Create large variables
large_matrix = np.random.randn(1000, 1000)
data_dict = {"data": np.random.randn(500, 500), "labels": list(range(1000))}

# Save with auto-detected name
logger.save_large_vars(large_matrix)  # Saves as "large_matrix.pkl"

# Save with custom name
logger.save_large_vars(data_dict, "my_dataset")  # Saves as "my_dataset.pkl"
```

**Loading saved variables:**
```python
import pickle

# Load saved variable
with open(logger.log_dir / "large_vars" / "large_matrix.pkl", 'rb') as f:
    loaded_matrix = pickle.load(f)
```

### 5. `rm_logs()`
Removes all timestamped log directories in the base path. **Use with caution!**

**Example:**
```python
# This will prompt for confirmation before deletion
logger.rm_logs()
```

**Behavior:**
- Scans for directories matching pattern: `YYYY-MM-DD-HH-MM-SS-*`
- Lists all directories to be deleted
- Requires user confirmation ("yes" to proceed)
- Deletes directories one by one with status updates

## Complete Example

```python
import numpy as np
import matplotlib.pyplot as plt
from Logger import LoggerFormatter

# Define hyperparameters (uppercase for auto-detection)
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50
MODEL_TYPE = "CNN"
DROPOUT_RATE = 0.5

# Initialize logger
logger = LoggerFormatter(project_name="mnist_training")

# Set random seed for reproducibility
logger.seed_everything(42)

# Save hyperparameters
logger.save_hyperparameters()

# Generate and save training data
train_data = np.random.randn(1000, 784)
train_labels = np.random.randint(0, 10, 1000)

# Save large variables
logger.save_large_vars(train_data, "training_data")
logger.save_large_vars(train_labels, "training_labels")

# Create and save training curve
epochs = range(1, EPOCHS + 1)
train_loss = np.exp(-np.array(epochs) * 0.1) + np.random.randn(EPOCHS) * 0.01
val_loss = np.exp(-np.array(epochs) * 0.08) + np.random.randn(EPOCHS) * 0.02

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True)

# Save figure with high quality
logger.save_fig("training_curves.png", dpi=300)
plt.close()

print(f"All logs saved to: {logger.log_dir}")
```

## Directory Structure
After running, your log directory will look like:
```
logs/
└── 2024-01-15-14-30-45-mnist_training/
    ├── hyperparameters.json
    ├── your_script.py (or notebook.ipynb)
    ├── training_curves.png
    └── large_vars/
        ├── training_data.pkl
        └── training_labels.pkl
```

## Common Errors and Solutions

### Error: "Figure name must be provided"
```python
# Wrong
logger.save_fig("")

# Correct
logger.save_fig("my_plot.png")
```

### Error: "DPI must be at least 300"
```python
# Wrong
logger.save_fig("plot.png", dpi=100)

# Correct
logger.save_fig("plot.png", dpi=300)
```

### Error: "matplotlib is required for save_fig"
```python
# Install matplotlib first
# pip install matplotlib
```

### Variable name not detected for save_large_vars
```python
# If auto-detection fails, specify name explicitly
anonymous_array = np.random.randn(100, 100)
logger.save_large_vars(anonymous_array, "my_array")
```

## Best Practices

1. **Hyperparameter naming**: Use UPPERCASE for variables you want to track
2. **Figure quality**: Always use DPI >= 300 for publication-quality figures
3. **Large variables**: Use `save_large_vars()` for arrays > 1000MB to keep the main directory clean
4. **Cleanup**: Use `rm_logs()` carefully - it permanently deletes directories
5. **Reproducibility**: Always call `seed_everything()` at the beginning of experiments

## Requirements
- Python 3.6+
- Optional dependencies:
  - `matplotlib` for `save_fig()`
  - `numpy` for enhanced `seed_everything()`
  - `torch` for PyTorch seed setting
  - `ipynbname` for Jupyter notebook saving
