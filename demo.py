import numpy as np
import matplotlib.pyplot as plt
from Logger import LoggerFormtter

# Define hyperparameters (uppercase for auto-detection)
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50
MODEL_TYPE = "CNN"
DROPOUT_RATE = 0.5

# Initialize logger
logger = LoggerFormtter(project_name="mnist_training")

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
