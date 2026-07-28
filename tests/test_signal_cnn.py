"""Tests for the signal-only SignalCNN classifier + train_signal_classifier."""

import numpy as np
import torch

from leech.dataset import SignalDataset, collate_fn
from leech.models import MODEL_REGISTRY, SignalCNN
from leech.training import compute_class_weights_from_labels, train_signal_classifier


def test_registered():
    assert MODEL_REGISTRY["SignalCNN"] is SignalCNN


def test_forward_ignores_sequence():
    m = SignalCNN(num_classes=5, signal_len=256)
    out = m(torch.randn(3, 1, 256), sequence=torch.zeros(3, 1), features=None)
    assert out.shape == (3, 5)


def test_signal_dataset_batch_contract():
    ds = SignalDataset(
        np.random.randn(8, 256).astype(np.float32), np.array([0, 1, 2, 3, 4, 0, 1, 2])
    )
    batch = collate_fn([ds[i] for i in range(8)])
    assert set(batch) >= {"signal", "sequence", "label"}
    assert batch["signal"].shape == (8, 1, 256)
    assert batch["label"].shape == (8,)


def test_class_weights_from_labels():
    w = compute_class_weights_from_labels(np.array([0, 0, 0, 0, 1, 2, 2]), num_classes=3)
    assert w.shape == (3,)
    assert w[1] > w[2] > w[0]  # rarer class -> larger weight


def test_train_signal_classifier_separable():
    rng = np.random.default_rng(0)
    X = np.concatenate(
        [
            np.full((150, 128), k, np.float32)
            + 0.3 * rng.standard_normal((150, 128)).astype(np.float32)
            for k in range(3)
        ]
    )
    y = np.repeat(np.arange(3), 150)
    idx = rng.permutation(len(X))
    tr, va = idx[:350], idx[350:]
    model, _, _ = train_signal_classifier(
        X[tr], y[tr], X[va], y[va], num_classes=3, signal_len=128, epochs=8, device="cpu"
    )
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X[va][:, None, :])).argmax(1).numpy()
    assert (pred == y[va]).mean() > 0.9
