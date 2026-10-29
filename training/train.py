"""Repeats the seven Part 1 training steps and records fixed-weight measurements."""

import copy

import pandas as pd
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def train_epoch(model, batches, loss_function, optimizer):
    """Updates network parameters once for each batch of training examples."""
    model.train()
    for features, labels in batches:
        # 2. Forward propagation produces one raw score per supported class.
        scores = model(features)
        # 3. Cross-entropy compares the scores with the recorded class indices.
        loss = loss_function(scores, labels)
        if not torch.isfinite(loss):
            raise ValueError("Training loss became non-finite; inspect inputs and learning rate.")
        # 4–5. Backpropagation calculates parameter gradients for this batch.
        optimizer.zero_grad()
        loss.backward()
        # 6. SGD subtracts learning_rate × gradient from each parameter.
        optimizer.step()


def evaluate(model, features, labels):
    """Measures one fixed set of weights without changing it or building gradients."""
    model.eval()
    with torch.no_grad():
        scores = model(features)
        loss = nn.functional.cross_entropy(scores, labels).item()
        accuracy = (scores.argmax(dim=1) == labels).float().mean().item()
    # The declared output classes remain in F1 averages even when none is predicted.
    truth, predicted = labels.cpu().numpy(), scores.argmax(dim=1).cpu().numpy()
    classes = list(range(scores.shape[1]))
    return {
        "loss": loss, "accuracy": accuracy,
        "macro_f1": float(f1_score(
            truth, predicted, labels=classes, average="macro", zero_division=0
        )),
        "weighted_f1": float(f1_score(
            truth, predicted, labels=classes, average="weighted", zero_division=0
        )),
    }


def fit_network(model, train_x, train_y, validation_x, validation_y, settings, on_epoch=None):
    """Records each epoch and restores the weights with lowest validation loss."""
    batches = DataLoader(
        TensorDataset(train_x, train_y), batch_size=settings["batch_size"], shuffle=True
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=settings["learning_rate"])
    history, best_loss, best_epoch = [], float("inf"), 0
    best_weights = copy.deepcopy(model.state_dict())
    # 7. Each epoch repeats Steps 2–6; the history supports a convergence check.
    for epoch in range(settings["epochs"] + 1):
        if epoch > 0:
            train_epoch(model, batches, nn.CrossEntropyLoss(), optimizer)
        for split, features, labels in [
            ("train", train_x, train_y), ("validation", validation_x, validation_y)
        ]:
            result = evaluate(model, features, labels)
            history.append({"epoch": epoch, "split": split, **result})
            if split == "validation" and result["loss"] < best_loss:
                best_loss, best_epoch = result["loss"], epoch
                best_weights = copy.deepcopy(model.state_dict())
        # Observers receive copies after both splits are measured at fixed weights.
        if on_epoch is not None:
            on_epoch(epoch, [dict(row) for row in history[-2:]])
        if epoch % 20 == 0 or epoch == settings["epochs"]:
            print(f"Epoch {epoch}: validation loss {result['loss']:.4f}", flush=True)
    model.load_state_dict(best_weights)
    return pd.DataFrame(history), best_epoch
