import torch
from metrics import spearman_corr


def to_gpu(obj, device):
    if isinstance(obj, torch.Tensor):
        try:
            return obj.to(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [to_gpu(i, device=device) for i in obj]
    elif isinstance(obj, tuple):
        return (to_gpu(i, device=device) for i in obj)
    elif isinstance(obj, dict):
        return {i: to_gpu(j, device=device) for i, j in obj.items()}
    else:
        return obj


def train_model(model, optimizer, loader):

    model.train()
    device = next(model.parameters()).device
    epoch_loss = 0
    for wt, mut, label, _ in loader:
        wt, mut, label = to_gpu(wt, device), to_gpu(mut, device), to_gpu(label, device)
        optimizer.zero_grad()
        _, loss = model(wt, mut, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), norm_type=2, max_norm=10, error_if_nonfinite=True)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def validation_model(model, loader):

    model.eval()
    device = next(model.parameters()).device
    epoch_loss = 0
    with torch.no_grad():
        for wt, mut, label, _ in loader:
            wt, mut, label = to_gpu(wt, device), to_gpu(mut, device), to_gpu(label, device)
            pred = model(wt, mut)
            loss = -spearman_corr(pred, label)
            epoch_loss += loss.item()
    return epoch_loss / len(loader)


def test_model(model, loader):

    model.eval()
    device = next(model.parameters()).device
    results = {}
    results["forward"] = {}
    results["backward"] = {}
    preds = []
    trues = []
    with torch.no_grad():
        for wt, mut, label, name in loader:
            name = name[0]
            wt, mut = to_gpu(wt, device), to_gpu(mut, device)
            label = label.item()

            pred = model(wt, mut).item()
            results["forward"][name] = pred
            preds.append(pred)
            trues.append(label)

            pred = model(mut, wt).item()
            results["backward"][name] = pred
            preds.append(pred)
            trues.append(-label)

    # spearman correlation
    spearman_correlation = spearman_corr(torch.tensor(preds), torch.tensor(trues))

    # mse
    rmse = torch.nn.functional.mse_loss(torch.tensor(preds), torch.tensor(trues)) ** 0.5

    # mae
    mae = torch.nn.functional.l1_loss(torch.tensor(preds), torch.tensor(trues))

    return spearman_correlation.item(), rmse.item(), mae.item(), results
