"""GP 기반 Bayesian Optimization. 실행: python bayesian_optimization.py"""

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from tqdm.auto import tqdm

from base_model import (
    DataConfig, TrainConfig, HyperParameters,
    prepare_data, train_candidate, save_result, evaluate_test,
)

import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood


@dataclass(frozen=True)
class SearchConfig:
    lr_bounds: tuple = (1e-4, 1e-2)
    weight_decay_bounds: tuple = (1e-6, 1e-2)
    max_evaluations: int = 50
    check_every: int = 5
    patience: int = 3
    min_delta: float = 0.001  # 정확도 0.1%p


class SearchSession:
    def __init__(self, data, output, seed, search_seed, config, train_config):
        bounds = np.array([config.lr_bounds, config.weight_decay_bounds], dtype=float)
        if not np.isfinite(bounds).all() or np.any(bounds <= 0) or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Bounds must be finite, positive and increasing")
        if config.max_evaluations < 5 or config.check_every < 1 or config.patience < 1 or config.min_delta < 0:
            raise ValueError("Invalid budget or patience settings")
        self.data, self.config, self.train_config = data, config, train_config
        self.seed, self.search_seed = seed, search_seed
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.log_bounds = np.log10(bounds)
        self.rng = np.random.default_rng(search_seed)
        self.rows, self.cache, self.checks = [], {}, []
        self.best, self.best_path, self.reference = None, None, None
        self.bad_windows, self.cache_hits = 0, 0
        self.reason = None
        self.start = perf_counter()

    def evaluate(self, point, branch=0, iteration=1):
        key = tuple(np.round(point, 12))
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        if self.reason is not None:
            return None
        values = 10 ** (self.log_bounds[:, 0] + np.array(key) * np.diff(self.log_bounds)[:, 0])
        hp = HyperParameters(*map(float, values))
        started = perf_counter()
        row = {"evaluation": len(self.rows) + 1, "point": list(key),
               "hyperparameters": asdict(hp), "seed": self.seed,
               "branch": branch, "iteration": iteration}
        try:
            result = train_candidate(self.data, hp, self.train_config, self.seed)
            row.update(status="ok", val_accuracy=result.val_accuracy,
                       val_loss=result.val_loss, training_seconds=result.elapsed_seconds)
            if self.best is None or row["val_accuracy"] > self.best["val_accuracy"]:
                path = save_result(result, self.output / f"candidate_{row['evaluation']:03d}")
                if self.best_path is not None:
                    shutil.rmtree(self.best_path.parent)
                self.best, self.best_path = row, path
        except FloatingPointError as error:
            row.update(status="failed", val_accuracy=None, val_loss=None,
                       training_seconds=perf_counter() - started, error=str(error))
        self.rows.append(row)
        self.cache[key] = row
        best_score = self.best["val_accuracy"] if self.best else None
        row.update(best_val_accuracy=best_score, search_seconds=perf_counter() - self.start)
        with (self.output / "trials.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        # 신규 학습만 예산에 포함. 발산 포함, 중복 제외
        if len(self.rows) % self.config.check_every == 0:
            score = best_score if best_score is not None else -1.0
            improved = self.reference is None or (
                score > self.reference and score + 1e-12 >= self.reference + self.config.min_delta
            )
            if improved:
                self.reference, self.bad_windows = score, 0
            else:
                self.bad_windows += 1
            self.checks.append({"evaluation": len(self.rows), "best_val_accuracy": best_score,
                                "bad_windows": self.bad_windows})
            if self.bad_windows >= self.config.patience:
                self.reason = "patience"
        if len(self.rows) >= self.config.max_evaluations and self.reason is None:
            self.reason = "max_evaluations"
        tqdm.write(
            f"[{len(self.rows):02d}/{self.config.max_evaluations}] "
            f"val={row['val_accuracy']} best={best_score}"
        )
        return row

    def finish(self, reason):
        report = {"method": METHOD, "seed": self.seed, "search_seed": self.search_seed,
                  "search_config": asdict(self.config), "train_config": asdict(self.train_config),
                  "data": self.data.metadata, "evaluations": len(self.rows),
                  "cache_hits": self.cache_hits, "stop_reason": self.reason or reason,
                  "search_seconds": perf_counter() - self.start,
                  "training_seconds": sum(row["training_seconds"] for row in self.rows),
                  "patience_checks": self.checks, "best": self.best,
                  "checkpoint": str(self.best_path) if self.best_path else None}
        (self.output / "search.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
        )
        return report


METHOD = "bayesian"


def propose_point(session):
    # GP: CPU float64. 학습 RNG와 분리
    acquisition_seed = int(session.rng.integers(0, 2**31 - 1))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(acquisition_seed)
        x = torch.tensor([row["point"] for row in session.rows], dtype=torch.double)
        # 발산 후보: GP에 정확도 0으로 반영
        y = torch.tensor([[row["val_accuracy"] if row["status"] == "ok" else 0.0]
                          for row in session.rows], dtype=torch.double)
        model = SingleTaskGP(x, y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        sampler = SobolQMCNormalSampler(torch.Size([128]), seed=acquisition_seed)
        acquisition = qLogNoisyExpectedImprovement(model, X_baseline=x, sampler=sampler)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
        candidate, _ = optimize_acqf(acquisition, bounds, q=1, num_restarts=8, raw_samples=128)
        point = candidate.detach().numpy()[0]
        if tuple(np.round(point, 12)) in session.cache:
            # 중복 제안: 새 후보 풀에서 획득함수 최댓값 선택
            pool = torch.quasirandom.SobolEngine(2, scramble=True, seed=acquisition_seed).draw(512).double()
            mask = [tuple(np.round(p.numpy(), 12)) not in session.cache for p in pool]
            pool = pool[mask]
            if not len(pool):
                raise RuntimeError("No unique acquisition candidates")
            with torch.no_grad():
                values = torch.cat([acquisition(chunk.unsqueeze(1)) for chunk in pool.split(64)])
            point = pool[values.argmax()].numpy()
    return point


def search(data, output, seed=42, search_seed=None, config=SearchConfig(), train_config=TrainConfig()):
    search_seed = seed if search_seed is None else search_seed
    session = SearchSession(data, output, seed, search_seed, config, train_config)
    # 초기 5개 Sobol 후보. 이후 1개씩 순차 제안
    initial = torch.quasirandom.SobolEngine(2, scramble=True, seed=search_seed).draw(5).double().numpy()
    for point in tqdm(initial, desc=f"Seed {seed} | Initial candidates", unit="candidate", leave=False):
        session.evaluate(point)
        if session.reason:
            break
    if session.best is None:
        return session.finish("no_valid_candidates")

    with tqdm(total=config.max_evaluations, initial=len(session.rows), desc=f"Seed {seed} | BO evaluations", 
              unit="model", leave=False) as pbar:
        while session.reason is None:
            before = len(session.rows)
            point = propose_point(session)
            session.evaluate(point, iteration=len(session.rows) - 3)
            pbar.update(len(session.rows) - before)

    return session.finish("max_evaluations")


def run_experiment(output, run_seeds=(42, 43, 44, 45, 46), final_seeds=(1000, 1001, 1002),
                   search_config=SearchConfig(), train_config=TrainConfig(), data_config=DataConfig()):
    if not run_seeds or len(set(run_seeds)) != len(run_seeds):
        raise ValueError("Use nonempty, distinct run_seeds")
    if len(set(final_seeds)) != len(final_seeds) or set(run_seeds) & set(final_seeds):
        raise ValueError("Final seeds must be distinct and separate from search training seeds")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = prepare_data(data_config)
    reports = []
    # 전체 탐색 완료 후 테스트 평가
    for seed in tqdm(run_seeds, desc=f"{METHOD} | Search runs", unit="run"):
        reports.append(search(data, output / f"run_{seed}" / "search", seed=seed,
                              config=search_config, train_config=train_config))
    for report in tqdm(reports, desc=f"{METHOD} | Final evaluation", unit="run"):
        report["final_evaluations"] = []
        if report["best"] is None:
            continue
        hp = HyperParameters(**report["best"]["hyperparameters"])
        for seed in tqdm(final_seeds, desc=f"Run {report['seed']} | Retraining", unit="model", leave=False):
            result = train_candidate(data, hp, train_config, seed)
            checkpoint = save_result(result, output / f"run_{report['seed']}" / f"final_{seed}")
            metrics = evaluate_test(checkpoint, root=data_config.root, device=train_config.device,
                                    download=data_config.download)
            metrics["training_seconds"] = result.elapsed_seconds
            report["final_evaluations"].append(metrics)
        scores = [item["test_accuracy"] for item in report["final_evaluations"]]
        report["mean_test_accuracy"] = float(np.mean(scores)) if scores else None
        (output / f"run_{report['seed']}" / "result.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
        )
    # 비교 단위: 탐색 실행별 재학습 평균
    means = [r["mean_test_accuracy"] for r in reports if r.get("mean_test_accuracy") is not None]
    summary = {"method": METHOD, "run_seeds": list(run_seeds), "final_seeds": list(final_seeds),
               "mean_test_accuracy": float(np.mean(means)) if means else None,
               "std_test_accuracy": float(np.std(means, ddof=1)) if len(means) > 1 else None,
               "failed_searches": sum(r["best"] is None for r in reports), "runs": reports}
    (output / "experiment.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    run_experiment("runs/bayesian")
