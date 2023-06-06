import numpy as np
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, zero_one_loss, precision_recall_fscore_support
import torch
from torch import nn

def validate(model, test_data, label_dict, embedding_database, label_database):
    predictions = []
    labels = []
    paths = []
    model.eval()
    embedding_database_tensor = torch.Tensor(embedding_database).cuda()

    for it, (img, label, path) in enumerate(test_data):
        b_images = img.cuda()
        label = label.cuda()
        with torch.no_grad():
            emb, logits = model(b_images)
            #emb = torch.div(emb,torch.linalg.norm(emb, dim=1).view(-1,1))
            logit = nn.Softmax(dim=-1)(logits)
            pred = torch.argmax(logit, dim=1)
            actual_pred = [label_dict[k] for k in pred.cpu().numpy()]
            predictions.extend(actual_pred)


        labels.extend(label.cpu().numpy())
        paths.extend(path)

    acc = accuracy_score(np.array(labels), np.array(predictions))
    C = confusion_matrix(np.array(labels), np.array(predictions))
    C_norm = C/C.astype(np.float).sum(axis=1, keepdims=True)

    return acc, C_norm

