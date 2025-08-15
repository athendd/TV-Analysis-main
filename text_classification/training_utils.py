import evaluate
import torch

metric = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)

    from sklearn.metrics import f1_score, accuracy_score
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_micro': f1_score(labels, preds, average='micro'),
        'f1_macro': f1_score(labels, preds, average='macro'),
    }
