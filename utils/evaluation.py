# evaluation.py

import numpy as np

def calculate_accuracy(true_labels, predictions):
    """Calculate accuracy of predictions."""
    correct = np.sum(np.argmax(true_labels, axis=1) == np.argmax(predictions, axis=1))
    total = len(true_labels)
    return correct / total

def calculate_wer(true_labels, predictions):
    """Calculate Word Error Rate (WER)."""
    # Placeholder implementation for WER calculation
    wer = 0.0
    for true, pred in zip(true_labels, predictions):
        true_words = true.split()
        pred_words = pred.split()
        distance = np.zeros((len(true_words) + 1, len(pred_words) + 1))
        
        for i in range(len(true_words) + 1):
            distance[i][0] = i
        for j in range(len(pred_words) + 1):
            distance[0][j] = j
            
        for i in range(1, len(true_words) + 1):
            for j in range(1, len(pred_words) + 1):
                if true_words[i - 1] == pred_words[j - 1]:
                    cost = 0
                else:
                    cost = 1
                distance[i][j] = min(distance[i - 1][j] + 1,
                                     distance[i][j - 1] + 1,
                                     distance[i - 1][j - 1] + cost)
        
        wer += distance[len(true_words)][len(pred_words)] / len(true_words)
    
    return wer / len(true_labels)
