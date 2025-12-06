from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np

def evaluate_model(model, generator):
    # generator must be with shuffle=False
    y_true = generator.classes
    y_pred_prob = model.predict(generator).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except:
        auc = None
    return {'report': report, 'confusion_matrix': cm, 'auc': auc}
