import torch
from sklearn.metrics import precision_score, recall_score


def prec_recall_score(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
):


    # Load saved model using appropriate function based on framework (e.g., torch.load for PyTorch)
    model.load_state_dict(torch.load('F:\\Auto_PCOS_Challenge\\Final_Code\\Module\\pco_net_best_32_84.2875.pth'))
    model.eval()
    # Setup test loss, test accuracy, precision, and recall values
    test_loss, test_acc, precision, recall = 0, 0, 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # 3. Calculate and accumulate accuracy (adjust based on task type, e.g., binary vs. multi-class)
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

            # y is binary
            tp = ((test_pred_labels == 1) & (y == 1)).sum().item()
            fp = ((test_pred_labels == 1) & (y == 0)).sum().item()
            fn = ((test_pred_labels == 0) & (y == 1)).sum().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn + 0.00005)
            
    return precision, recall
