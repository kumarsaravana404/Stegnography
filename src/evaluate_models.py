"""
Comprehensive Model Evaluation Script
Evaluates all trained models and generates comparison reports
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_train import SteganographyDetector
from feature_extraction import AudioFeatureExtractor, process_directory


class ModelEvaluator:
    """Evaluate and compare multiple models"""

    def __init__(self, data_clean_dir: str, data_stego_dir: str):
        self.data_clean_dir = data_clean_dir
        self.data_stego_dir = data_stego_dir
        self.results = {}
        self.X_test = None
        self.y_test = None

    def load_test_data(self):
        """Load and prepare test data"""
        print("Loading test data...")
        extractor = AudioFeatureExtractor()

        X_clean, y_clean = process_directory(self.data_clean_dir, 0, extractor)
        X_stego, y_stego = process_directory(self.data_stego_dir, 1, extractor)

        X = np.vstack([X_clean, X_stego])
        y = np.hstack([y_clean, y_stego])

        # Use 20% as test set
        from sklearn.model_selection import train_test_split

        _, self.X_test, _, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Test set size: {len(self.X_test)} samples")

    def evaluate_model(self, model_path: str, model_name: str):
        """Evaluate a single model"""
        print(f"\nEvaluating {model_name}...")

        try:
            detector = SteganographyDetector()
            detector.load_model(model_path)

            # Get predictions
            X_scaled = detector.scaler.transform(self.X_test)
            y_pred = detector.model.predict(X_scaled)
            y_pred_proba = detector.model.predict_proba(X_scaled)[:, 1]

            # Calculate metrics
            metrics = {
                "model_name": model_name,
                "accuracy": accuracy_score(self.y_test, y_pred),
                "precision": precision_score(self.y_test, y_pred),
                "recall": recall_score(self.y_test, y_pred),
                "f1": f1_score(self.y_test, y_pred),
                "roc_auc": roc_auc_score(self.y_test, y_pred_proba),
            }

            # Store results
            self.results[model_name] = {
                "metrics": metrics,
                "y_pred": y_pred,
                "y_pred_proba": y_pred_proba,
                "confusion_matrix": confusion_matrix(self.y_test, y_pred),
            }

            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1 Score: {metrics['f1']:.4f}")

        except Exception as e:
            print(f"  Error: {e}")

    def evaluate_all_models(self):
        """Evaluate all models in the models directory"""
        models_dir = "models"

        if not os.path.exists(models_dir):
            print("No models directory found")
            return

        for filename in os.listdir(models_dir):
            if filename.endswith(".pkl"):
                model_name = filename.replace("_model.pkl", "").replace(".pkl", "")
                model_path = os.path.join(models_dir, filename)
                self.evaluate_model(model_path, model_name)

    def generate_comparison_table(self, save_path: str = None):
        """Generate comparison table of all models"""
        if not self.results:
            print("No results to compare")
            return

        # Create DataFrame
        metrics_list = [result["metrics"] for result in self.results.values()]
        df = pd.DataFrame(metrics_list)
        df = df.set_index("model_name")

        # Sort by F1 score
        df = df.sort_values("f1", ascending=False)

        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        print(df.to_string())
        print("=" * 80)

        if save_path:
            df.to_csv(save_path)
            print(f"\nComparison table saved to {save_path}")

        return df

    def plot_roc_curves(self, save_path: str = None):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))

        for model_name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result["y_pred_proba"])
            auc_score = result["metrics"]["roc_auc"]
            plt.plot(
                fpr, tpr, label=f"{model_name} (AUC = {auc_score:.3f})", linewidth=2
            )

        plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title("ROC Curves - Model Comparison", fontsize=14, fontweight="bold")
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"ROC curves saved to {save_path}")

        plt.close()

    def plot_confusion_matrices(self, save_path: str = None):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        if n_models == 0:
            return

        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, (model_name, result) in enumerate(self.results.items()):
            cm = result["confusion_matrix"]
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=axes[idx],
                xticklabels=["Clean", "Stego"],
                yticklabels=["Clean", "Stego"],
            )
            axes[idx].set_title(
                f'{model_name}\nAccuracy: {result["metrics"]["accuracy"]:.3f}'
            )
            axes[idx].set_ylabel("True Label")
            axes[idx].set_xlabel("Predicted Label")

        # Hide extra subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Confusion matrices saved to {save_path}")

        plt.close()

    def plot_metrics_comparison(self, save_path: str = None):
        """Plot bar chart comparing all metrics"""
        if not self.results:
            return

        metrics_list = [result["metrics"] for result in self.results.values()]
        df = pd.DataFrame(metrics_list)
        df = df.set_index("model_name")

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        for idx, metric in enumerate(metrics):
            df[metric].plot(kind="bar", ax=axes[idx], color=colors[idx], alpha=0.7)
            axes[idx].set_title(
                metric.upper().replace("_", " "), fontsize=12, fontweight="bold"
            )
            axes[idx].set_ylabel("Score")
            axes[idx].set_ylim([0, 1.1])
            axes[idx].grid(axis="y", alpha=0.3)
            axes[idx].tick_params(axis="x", rotation=45)

        # Hide last subplot
        axes[5].axis("off")

        plt.suptitle(
            "Model Performance Comparison", fontsize=16, fontweight="bold", y=1.02
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Metrics comparison saved to {save_path}")

        plt.close()

    def generate_report(self, output_dir: str = "evaluation_reports"):
        """Generate comprehensive evaluation report"""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save comparison table
        table_path = os.path.join(output_dir, f"model_comparison_{timestamp}.csv")
        self.generate_comparison_table(table_path)

        # Save plots
        roc_path = os.path.join(output_dir, f"roc_curves_{timestamp}.png")
        self.plot_roc_curves(roc_path)

        cm_path = os.path.join(output_dir, f"confusion_matrices_{timestamp}.png")
        self.plot_confusion_matrices(cm_path)

        metrics_path = os.path.join(output_dir, f"metrics_comparison_{timestamp}.png")
        self.plot_metrics_comparison(metrics_path)

        # Save JSON report
        json_path = os.path.join(output_dir, f"evaluation_report_{timestamp}.json")
        report = {
            "timestamp": timestamp,
            "test_set_size": len(self.y_test),
            "models": {},
        }

        for model_name, result in self.results.items():
            report["models"][model_name] = {
                "metrics": result["metrics"],
                "confusion_matrix": result["confusion_matrix"].tolist(),
            }

        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Evaluation report generated in: {output_dir}")
        print(f"{'='*80}")


def main():
    """Main evaluation function"""
    DATA_CLEAN = os.path.join("data", "clean")
    DATA_STEGO = os.path.join("data", "stego")

    print("=" * 80)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 80)

    evaluator = ModelEvaluator(DATA_CLEAN, DATA_STEGO)
    evaluator.load_test_data()
    evaluator.evaluate_all_models()
    evaluator.generate_report()

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
