"""
Model evaluation script with detailed metrics.

Provides accuracy, precision, recall, F1, confusion matrix,
and threshold calibration analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from training.train_resnet import LogoDataset, get_transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model and collect predictions.
    
    Returns:
        Tuple of (true_labels, predicted_labels, probabilities)
    """
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of genuine
    
    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict:
    """Compute comprehensive evaluation metrics."""
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # ROC-AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold (Youden's J index)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'optimal_threshold': float(optimal_threshold),
        'threshold_metrics': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
    }


def evaluate_by_forgery_level(
    data_dir: Path,
    model: torch.nn.Module,
    device: torch.device
) -> Dict:
    """
    Evaluate performance separately for each forgery level.
    
    Useful for understanding model performance on hard negatives.
    """
    from PIL import Image
    from torchvision import transforms
    
    transform = get_transforms('val')
    
    val_forged = data_dir / 'val' / 'forged'
    
    level_predictions = {1: [], 2: [], 3: []}
    
    model.eval()
    
    for img_path in val_forged.glob('*.png'):
        # Extract level from filename (format: forged_L{level}_...)
        filename = img_path.name
        if '_L' in filename:
            try:
                level = int(filename.split('_L')[1][0])
            except:
                continue
        else:
            continue
        
        # Predict
        image = Image.open(img_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1)
            pred = output.argmax(1).item()
        
        # Correct detection = predicting as forged (0)
        correct = (pred == 0)
        confidence = prob[0, 0].item() if pred == 0 else prob[0, 1].item()
        
        if level in level_predictions:
            level_predictions[level].append({
                'correct': correct,
                'confidence': confidence
            })
    
    # Compute per-level metrics
    level_metrics = {}
    for level, preds in level_predictions.items():
        if preds:
            accuracy = sum(p['correct'] for p in preds) / len(preds)
            avg_confidence = sum(p['confidence'] for p in preds) / len(preds)
            level_metrics[f'level_{level}'] = {
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'num_samples': len(preds)
            }
    
    return level_metrics


def calibrate_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_metrics: Dict = None
) -> Dict:
    """
    Find optimal thresholds for different operating points.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities for genuine class
        target_metrics: Optional target metrics to optimize for
        
    Returns:
        Dictionary of threshold recommendations
    """
    if target_metrics is None:
        target_metrics = {
            'high_precision': 0.95,  # Minimize false positives
            'high_recall': 0.95,     # Minimize false negatives
            'balanced': None         # F1 optimal
        }
    
    thresholds = np.arange(0.1, 1.0, 0.01)
    results = []
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        
        if len(np.unique(y_pred)) < 2:
            continue
        
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results.append({
            'threshold': thresh,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })
    
    if not results:
        return {}
    
    # Find optimal thresholds
    recommendations = {}
    
    # High precision threshold
    high_prec = [r for r in results if r['precision'] >= target_metrics.get('high_precision', 0.95)]
    if high_prec:
        recommendations['high_precision'] = min(high_prec, key=lambda x: x['threshold'])
    
    # High recall threshold
    high_rec = [r for r in results if r['recall'] >= target_metrics.get('high_recall', 0.95)]
    if high_rec:
        recommendations['high_recall'] = max(high_rec, key=lambda x: x['threshold'])
    
    # Balanced (max F1)
    recommendations['balanced'] = max(results, key=lambda x: x['f1'])
    
    return recommendations


def run_evaluation(
    model_path: Union[str, Path],
    data_dir: Union[str, Path],
    output_path: Union[str, Path] = None
) -> Dict:
    """
    Run complete evaluation pipeline.
    
    Args:
        model_path: Path to trained model checkpoint
        data_dir: Data directory with val split
        output_path: Optional path to save results
        
    Returns:
        Complete evaluation results
    """
    from models.resnet_classifier import LogoClassifier
    
    model_path = Path(model_path)
    data_dir = Path(data_dir)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model = LogoClassifier()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create validation dataloader
    val_dataset = LogoDataset(data_dir, 'val', get_transforms('val'))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Evaluate
    y_true, y_pred, y_prob = evaluate(model, val_loader, device)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    
    # Per-level metrics
    level_metrics = evaluate_by_forgery_level(data_dir, model, device)
    
    # Threshold calibration
    threshold_rec = calibrate_thresholds(y_true, y_prob)
    
    # Compile results
    results = {
        'overall_metrics': metrics,
        'per_level_metrics': level_metrics,
        'threshold_recommendations': {
            k: {
                'threshold': v['threshold'],
                'precision': v['precision'],
                'recall': v['recall'],
                'f1': v['f1']
            } for k, v in threshold_rec.items()
        },
        'model_path': str(model_path),
        'num_samples': len(y_true)
    }
    
    # Log summary
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    if level_metrics:
        logger.info("Per-level accuracy:")
        for level, lm in level_metrics.items():
            logger.info(f"  {level}: {lm['accuracy']:.4f} ({lm['num_samples']} samples)")
    
    # Save if path provided
    if output_path:
        with open(output_path, 'w') as f:
            # Remove large arrays for JSON
            results_json = results.copy()
            del results_json['overall_metrics']['threshold_metrics']
            json.dump(results_json, f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    return results


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate classifier')
    parser.add_argument('--model', type=str, default=None,
                        help='Model checkpoint path (defaults based on element)')
    parser.add_argument('--data', type=str, default='data/logo',
                        help='Data directory')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='Output path for results')
    parser.add_argument('--element', type=str, default='logo',
                        choices=['logo', 'stamp'], help='Element type')
    
    args = parser.parse_args()
    
    # Adjust defaults based on element
    element = args.element
    if args.model is None:
        args.model = f'models/{element}_resnet50.pt'
    
    if element == 'stamp' and args.data == 'data/logo':
        args.data = 'data/stamp'
        
    run_evaluation(args.model, args.data, args.output)


if __name__ == '__main__':
    main()
