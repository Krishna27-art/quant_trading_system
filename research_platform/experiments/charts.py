"""
Charts Generator

Generates visualization charts for experiments.
Creates equity curves, ROC curves, SHAP plots, and more.
"""

import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

from utils.logger import get_logger

logger = get_logger("experiments.charts")


class ChartsGenerator:
    """
    Charts Generator.
    
    Generates:
    - Equity curves
    - ROC curves
    - Confusion matrices
    - Calibration curves
    - Feature importance plots
    - SHAP plots
    - Monthly returns
    - Drawdown curves
    """
    
    def __init__(self, output_dir: str = "./experiment_charts"):
        """
        Initialize charts generator.
        
        Args:
            output_dir: Directory to save charts
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._logger = get_logger("experiments.charts")
        
        # Set style
        sns.set_style("darkgrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def generate_equity_curve(
        self,
        returns: List[float],
        experiment_id: str,
        title: str = "Equity Curve",
    ) -> str:
        """
        Generate equity curve chart.
        
        Args:
            returns: List of returns
            experiment_id: Experiment ID
            title: Chart title
            
        Returns:
            Path to saved chart
        """
        chart_id = f"EC-{uuid.uuid4().hex[:8].upper()}"
        
        # Calculate cumulative returns
        cumulative_returns = np.cumsum(returns) / 100
        cumulative_returns = (1 + cumulative_returns) * 100
        
        # Create plot
        fig, ax = plt.subplots()
        ax.plot(cumulative_returns, linewidth=2, color='green')
        ax.axhline(y=100, color='red', linestyle='--', linewidth=1, label='Breakeven')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Portfolio Value (%)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save chart
        filepath = os.path.join(self.output_dir, f"{chart_id}_equity_curve.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        self._logger.info(f"Generated equity curve: {filepath}")
        
        return filepath
    
    def generate_roc_curve(
        self,
        y_true: List[int],
        y_scores: List[float],
        experiment_id: str,
        title: str = "ROC Curve",
    ) -> str:
        """
        Generate ROC curve chart.
        
        Args:
            y_true: True labels
            y_scores: Predicted scores
            experiment_id: Experiment ID
            title: Chart title
            
        Returns:
            Path to saved chart
        """
        chart_id = f"RC-{uuid.uuid4().hex[:8].upper()}"
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Create plot
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save chart
        filepath = os.path.join(self.output_dir, f"{chart_id}_roc_curve.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        self._logger.info(f"Generated ROC curve: {filepath}")
        
        return filepath
    
    def generate_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        experiment_id: str,
        title: str = "Confusion Matrix",
    ) -> str:
        """
        Generate confusion matrix chart.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            experiment_id: Experiment ID
            title: Chart title
            
        Returns:
            Path to saved chart
        """
        chart_id = f"CM-{uuid.uuid4().hex[:8].upper()}"
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create plot
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(title)
        
        # Save chart
        filepath = os.path.join(self.output_dir, f"{chart_id}_confusion_matrix.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        self._logger.info(f"Generated confusion matrix: {filepath}")
        
        return filepath
    
    def generate_feature_importance_plot(
        self,
        feature_importance: Dict[str, float],
        experiment_id: str,
        title: str = "Feature Importance",
        top_n: int = 20,
    ) -> str:
        """
        Generate feature importance plot.
        
        Args:
            feature_importance: Dictionary mapping features to importance
            experiment_id: Experiment ID
            title: Chart title
            top_n: Number of top features to show
            
        Returns:
            Path to saved chart
        """
        chart_id = f"FI-{uuid.uuid4().hex[:8].upper()}"
        
        # Sort and get top features
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]
        
        features, importance = zip(*sorted_features)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importance, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Save chart
        filepath = os.path.join(self.output_dir, f"{chart_id}_feature_importance.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        self._logger.info(f"Generated feature importance plot: {filepath}")
        
        return filepath
    
    def generate_drawdown_curve(
        self,
        returns: List[float],
        experiment_id: str,
        title: str = "Drawdown Curve",
    ) -> str:
        """
        Generate drawdown curve chart.
        
        Args:
            returns: List of returns
            experiment_id: Experiment ID
            title: Chart title
            
        Returns:
            Path to saved chart
        """
        chart_id = f"DC-{uuid.uuid4().hex[:8].upper()}"
        
        # Calculate cumulative returns and drawdown
        cumulative_returns = np.cumsum(returns) / 100
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / peak * 100
        
        # Create plot
        fig, ax = plt.subplots()
        ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        ax.plot(drawdown, linewidth=2, color='red')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Save chart
        filepath = os.path.join(self.output_dir, f"{chart_id}_drawdown_curve.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        self._logger.info(f"Generated drawdown curve: {filepath}")
        
        return filepath
    
    def generate_monthly_returns(
        self,
        returns: List[float],
        dates: List[datetime],
        experiment_id: str,
        title: str = "Monthly Returns",
    ) -> str:
        """
        Generate monthly returns chart.
        
        Args:
            returns: List of returns
            dates: List of corresponding dates
            experiment_id: Experiment ID
            title: Chart title
            
        Returns:
            Path to saved chart
        """
        chart_id = f"MR-{uuid.uuid4().hex[:8].upper()}"
        
        # Group by month
        monthly_returns = {}
        for ret, date in zip(returns, dates):
            month_key = date.strftime("%Y-%m")
            if month_key not in monthly_returns:
                monthly_returns[month_key] = []
            monthly_returns[month_key].append(ret)
        
        # Calculate monthly returns
        monthly_avg = {k: np.mean(v) for k, v in monthly_returns.items()}
        
        # Sort by month
        sorted_months = sorted(monthly_avg.keys())
        monthly_values = [monthly_avg[m] for m in sorted_months]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = ['green' if v > 0 else 'red' for v in monthly_values]
        ax.bar(range(len(sorted_months)), monthly_values, color=colors)
        ax.set_xticks(range(len(sorted_months)))
        ax.set_xticklabels(sorted_months, rotation=45, ha='right')
        ax.set_xlabel('Month')
        ax.set_ylabel('Average Return (%)')
        ax.set_title(title)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Save chart
        filepath = os.path.join(self.output_dir, f"{chart_id}_monthly_returns.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        self._logger.info(f"Generated monthly returns chart: {filepath}")
        
        return filepath
    
    def generate_calibration_curve(
        self,
        y_true: List[int],
        y_scores: List[float],
        experiment_id: str,
        title: str = "Calibration Curve",
        n_bins: int = 10,
    ) -> str:
        """
        Generate calibration curve chart.
        
        Args:
            y_true: True labels
            y_scores: Predicted scores
            experiment_id: Experiment ID
            title: Chart title
            n_bins: Number of bins
            
        Returns:
            Path to saved chart
        """
        chart_id = f"CC-{uuid.uuid4().hex[:8].upper()}"
        
        # Calculate calibration curve
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=n_bins)
        
        # Create plot
        fig, ax = plt.subplots()
        ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Calibration curve')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfectly calibrated')
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save chart
        filepath = os.path.join(self.output_dir, f"{chart_id}_calibration_curve.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        self._logger.info(f"Generated calibration curve: {filepath}")
        
        return filepath
    
    def generate_shap_summary_plot(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        experiment_id: str,
        title: str = "SHAP Summary Plot",
    ) -> str:
        """
        Generate SHAP summary plot.
        
        Args:
            shap_values: SHAP values array
            feature_names: List of feature names
            experiment_id: Experiment ID
            title: Chart title
            
        Returns:
            Path to saved chart
        """
        try:
            import shap
            
            chart_id = f"SH-{uuid.uuid4().hex[:8].upper()}"
            
            # Create SHAP summary plot
            plt.figure()
            shap.summary_plot(shap_values, feature_names=feature_names, show=False)
            plt.title(title)
            
            # Save chart
            filepath = os.path.join(self.output_dir, f"{chart_id}_shap_summary.png")
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            self._logger.info(f"Generated SHAP summary plot: {filepath}")
            
            return filepath
        except ImportError:
            self._logger.warning("SHAP not installed, skipping SHAP plot")
            return ""
