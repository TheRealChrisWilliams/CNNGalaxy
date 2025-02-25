import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class MetricsLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_file="metrics_log.csv"):
        super().__init__()
        self.log_file = log_file
        self.history = {"epoch": [], "mse": [], "recall": []}

    def on_epoch_end(self, epoch, logs=None):
        mse = logs.get("mse")
        recall = logs.get("recall")

        self.history["epoch"].append(epoch + 1)
        self.history["mse"].append(mse)
        self.history["recall"].append(recall)

        # Save to CSV
        df = pd.DataFrame(self.history)
        df.to_csv(self.log_file, index=False)

        # Plot metrics
        self.plot_metrics()

    def plot_metrics(self):
        plt.figure(figsize=(10, 5))

        # Plot MSE
        plt.subplot(1, 2, 1)
        plt.plot(self.history["epoch"], self.history["mse"], marker="o", label="MSE", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("MSE over Epochs")
        plt.legend()

        # Plot Recall
        plt.subplot(1, 2, 2)
        plt.plot(self.history["epoch"], self.history["recall"], marker="o", label="Recall", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Recall")
        plt.title("Recall over Epochs")
        plt.legend()

        plt.show()
