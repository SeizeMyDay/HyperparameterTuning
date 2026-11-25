
from __future__ import annotations

"""
사용법
data = prepare_data(DataConfig())       # 탐색 전체에서 한 번만 실행
result = train_candidate(data, HyperParameters(1e-3, 1e-4),
                         TrainConfig(), seed=42)
score = result.val_accuracy             # 최대화, 범위 [0, 1]
save_result(result, "runs/candidate_001")  # 필요한 후보만 저장
"""

"""
FashionMNIST 공통 모델. 실행: python base_model.py

후보마다 같은 seed로 새로 학습하고 마지막 epoch의 검증 정확도를 비교한다.
탐색 RNG는 별도로 관리하고, patience는 탐색 코드에서 적용한다.
테스트 평가는 탐색 종료 후 evaluate_test()로 실행한다.
"""

import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

# CUDA 재현성 설정은 초기화 전에 적용
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class HyperParameters:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    def __post_init__(self):
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")


@dataclass(frozen=True)
class DataConfig:
    root: str = "./data"
    split_seed: int = 2026
    val_per_class: int = 1000
    download: bool = True


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    num_workers: int = 0  # Windows 호환 기본값
    device: str = "auto"  # auto: CUDA가 있으면 CUDA, 없으면 CPU
    deterministic: bool = True

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("Invalid epochs, batch_size, or num_workers")


@dataclass
class DataBundle:
    train: Dataset
    val: Dataset
    metadata: dict


@dataclass
class TrialResult:
    val_accuracy: float
    val_loss: float
    elapsed_seconds: float
    history: list[dict]
    metadata: dict
    state_dict: dict[str, torch.Tensor] = field(repr=False)


class FashionCNN(nn.Module):
    #입력 (N, 1, 28, 28)
    # 출력 (N, 10) logits
    # Softmax 없음

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                         # 32 x 14 x 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                         # 64 x 7 x 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def image_transform():
    # 픽셀값을 [-1, 1]로 변환
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def stratified_indices(targets, val_per_class: int, seed: int):
    # 클래스별 분할
    # 전역 RNG 변경 안 함
    labels = np.asarray(targets)
    if val_per_class < 1:
        raise ValueError("val_per_class must be positive")
    rng = np.random.default_rng(seed)
    train_parts, val_parts = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if val_per_class >= len(indices):
            raise ValueError("Each class must retain at least one training item")
        rng.shuffle(indices)
        val_parts.append(indices[:val_per_class])
        train_parts.append(indices[val_per_class:])
    train_indices = np.concatenate(train_parts)
    val_indices = np.concatenate(val_parts)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def prepare_data(config: DataConfig = DataConfig()) -> DataBundle:
    # 공식 학습 세트 train/validation 분할
    dataset = datasets.FashionMNIST(
        root=config.root, train=True, download=config.download,
        transform=image_transform(),
    )
    train_idx, val_idx = stratified_indices(
        dataset.targets, config.val_per_class, config.split_seed
    )
    return DataBundle(
        Subset(dataset, train_idx.tolist()),
        Subset(dataset, val_idx.tolist()),
        {"dataset": "FashionMNIST", "config": asdict(config),
         "train_size": len(train_idx), "val_size": len(val_idx),
         "normalization": {"mean": 0.5, "std": 0.5}},
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in {"cpu", "cuda", "xpu"}:
        raise ValueError("Supported devices: auto, cpu, cuda, xpu")
    if device.type != "cpu":
        backend = getattr(torch, device.type, None)
        if backend is None or not backend.is_available():
            raise RuntimeError(f"Device unavailable in this environment: {name}")
    return device


def set_seed(seed: int, deterministic: bool = True):
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset, config, seed, shuffle, device):
    # 후보마다 다시 생성. 후보 평가 순서에 영향 받지 않음
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=config.batch_size, shuffle=shuffle,
        num_workers=config.num_workers, worker_init_fn=seed_worker,
        generator=generator, pin_memory=device.type == "cuda",
        drop_last=False,
    )


def synchronize(device):
    if device.type != "cpu":
        getattr(torch, device.type).synchronize(device)


def run_epoch(model, loader, device, optimizer=None):
    # 학습/평가 공통 루프
    # 마지막 작은 batch도 표본 수로 가중 평균
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite loss; candidate training diverged")
            if training:
                loss.backward()
                optimizer.step()
            batch_n = labels.size(0)
            total_loss += loss.item() * batch_n
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += batch_n
    if count == 0:
        raise ValueError("Empty dataset")
    return {"loss": total_loss / count, "accuracy": correct / count}


def train_candidate(
    data: DataBundle,
    hyperparameters: HyperParameters = HyperParameters(),
    config: TrainConfig = TrainConfig(),
    seed: int = 42,
    verbose: bool = False,
) -> TrialResult:
    # 새 모델 학습
    device = resolve_device(config.device)
    set_seed(seed, config.deterministic)
    synchronize(device)
    start = perf_counter()
    model = FashionCNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hyperparameters.learning_rate,
        weight_decay=hyperparameters.weight_decay,
        betas=(0.9, 0.999), eps=1e-8, foreach=False,
    )  # bias를 포함한 모든 학습 파라미터에 동일한 weight decay 적용
    train_loader = make_loader(data.train, config, seed, True, device)
    val_loader = make_loader(data.val, config, seed, False, device)
    history = []
    for epoch in range(1, config.epochs + 1):
        train = run_epoch(model, train_loader, device, optimizer)
        val = run_epoch(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train["loss"],
               "train_accuracy": train["accuracy"],
               "val_loss": val["loss"], "val_accuracy": val["accuracy"]}
        history.append(row)
        if verbose:
            print(f"Epoch {epoch:02d}/{config.epochs} | "
                  f"train loss={train['loss']:.4f} acc={train['accuracy']:.4%} | "
                  f"val loss={val['loss']:.4f} acc={val['accuracy']:.4%}", flush=True)
    synchronize(device)
    elapsed = perf_counter() - start
    metadata = {
        "model": "FashionCNN-v1",
        "hyperparameters": asdict(hyperparameters), "train_config": asdict(config),
        "seed": seed, "data": data.metadata, "actual_device": str(device),
    }
    weights = {key: value.detach().cpu().clone()
               for key, value in model.state_dict().items()}
    return TrialResult(val["accuracy"], val["loss"], elapsed, history, metadata, weights)


def save_result(result: TrialResult, directory: str | Path):
    # 가중치와 기록 저장
    # 기존 결과 덮어쓰지 않음
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "model.pt"
    report_path = directory / "metrics.json"
    if checkpoint.exists() or report_path.exists():
        raise FileExistsError(f"Result already exists in {directory}")
    report = {"val_accuracy": result.val_accuracy, "val_loss": result.val_loss,
              "elapsed_seconds": result.elapsed_seconds,
              "metadata": result.metadata, "history": result.history}
    torch.save({"state_dict": result.state_dict, **report}, checkpoint)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                      allow_nan=False), encoding="utf-8")
    return checkpoint


def evaluate_test(checkpoint: str | Path, root: str = "./data",
                  device: str = "auto", download: bool = True):
    # 재학습 없이 저장된 모델 테스트
    # 탐색 종료 후에만 호출됨
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    if metadata["model"] != "FashionCNN-v1":
        raise ValueError("Checkpoint model architecture does not match")
    config = TrainConfig(**{**metadata["train_config"], "device": device})
    set_seed(metadata["seed"], config.deterministic)
    actual_device = resolve_device(device)
    model = FashionCNN().to(actual_device)
    model.load_state_dict(payload["state_dict"])
    test = datasets.FashionMNIST(root=root, train=False, download=download,
                                 transform=image_transform())
    loader = make_loader(test, config, metadata["seed"], False, actual_device)
    metrics = run_epoch(model, loader, actual_device)
    return {"test_accuracy": metrics["accuracy"], "test_loss": metrics["loss"],
            "test_size": len(test), "checkpoint": str(checkpoint),
            "seed": metadata["seed"], "actual_device": str(actual_device)}


def main():
    output = Path("runs/baseline")
    if any((output / name).exists() for name in ("model.pt", "metrics.json")):
        raise FileExistsError(f"Result already exists in {output}")
    data = prepare_data(DataConfig())
    result = train_candidate(
        data, HyperParameters(), TrainConfig(), seed=42, verbose=True,
    )
    checkpoint = save_result(result, output)
    print(f"Saved: {checkpoint} | val accuracy={result.val_accuracy:.4%} | "
          f"time={result.elapsed_seconds:.1f}s")


if __name__ == "__main__":
    main()