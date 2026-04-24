#!/usr/bin/env python3
import argparse
import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def read_metrics(path):
    epochs = []
    loss = []
    train_acc = []
    val_acc = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epochs.append(int(row["epoch"]))
            loss.append(float(row["loss"]))
            train_acc.append(float(row["training_accuracy"]))
            val_acc.append(float(row["validation_accuracy"]))
    return np.array(epochs), np.array(loss), np.array(train_acc), np.array(val_acc)


def plot_learning_curves(path, output_dir):
    epochs, loss, train_acc, val_acc = read_metrics(path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, loss, label="Loss", color="#b91c1c", linewidth=2)
    axes[0].set_title("Learning Curve: Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, train_acc, label="Train Accuracy", color="#1d4ed8", linewidth=2)
    axes[1].plot(epochs, val_acc, label="Validation Accuracy", color="#15803d", linewidth=2)
    axes[1].set_title("Learning Curve: Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=160)
    plt.close(fig)


def read_confusion_pairs(path):
    truth = []
    predicted = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            truth.append(int(row["true_label"]))
            predicted.append(int(row["predicted_label"]))
    classes = sorted(set(truth) | set(predicted))
    index = {label: i for i, label in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(truth, predicted):
        matrix[index[t], index[p]] += 1
    return matrix, classes


def plot_confusion_matrix(path, output_dir):
    matrix, classes = read_confusion_pairs(path)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def read_grid(path):
    grid_points = []
    grid_labels = []
    sample_points = []
    sample_labels = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            point = (float(row["x1"]), float(row["x2"]))
            label = int(row["label"])
            if row["kind"] == "grid":
                grid_points.append(point)
                grid_labels.append(label)
            else:
                sample_points.append(point)
                sample_labels.append(label)
    return np.array(grid_points), np.array(grid_labels), np.array(sample_points), np.array(sample_labels)


def plot_decision_boundaries(path, output_dir):
    grid_points, grid_labels, sample_points, sample_labels = read_grid(path)
    x_values = np.unique(grid_points[:, 0])
    y_values = np.unique(grid_points[:, 1])
    xx, yy = np.meshgrid(x_values, y_values)
    zz = np.zeros_like(xx)
    lookup = {(x, y): label for (x, y), label in zip(grid_points, grid_labels)}
    for row in range(xx.shape[0]):
        for col in range(xx.shape[1]):
            zz[row, col] = lookup[(xx[row, col], yy[row, col])]

    fig, ax = plt.subplots(figsize=(7, 5))
    contour = ax.contourf(xx, yy, zz, levels=np.arange(zz.min(), zz.max() + 2) - 0.5, alpha=0.35, cmap="coolwarm")
    if sample_points.size > 0:
        scatter = ax.scatter(sample_points[:, 0], sample_points[:, 1], c=sample_labels, cmap="coolwarm", edgecolors="black", s=50)
        legend = ax.legend(*scatter.legend_elements(), title="Class")
        ax.add_artist(legend)
    ax.set_title("Decision Boundary")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    fig.colorbar(contour, ax=ax, label="Predicted Class")
    fig.tight_layout()
    fig.savefig(output_dir / "decision_boundary.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot metrics and decision-boundary outputs from the C++ library.")
    parser.add_argument("--metrics", default="output/metrics.csv")
    parser.add_argument("--confusion", default="output/confusion_pairs.csv")
    parser.add_argument("--grid", default="output/prediction_grid.csv")
    parser.add_argument("--output-dir", default="output/plots")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plot_learning_curves(args.metrics, output_dir)
    plot_confusion_matrix(args.confusion, output_dir)
    plot_decision_boundaries(args.grid, output_dir)
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
