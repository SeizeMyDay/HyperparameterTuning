from base_model import DataConfig, TrainConfig

import bayesian_optimization as bo
import dynamic_grid_search as grid
import dynamic_random_search as rand


if __name__ == "__main__":
    settings = {
        "data_config": DataConfig(),
        "train_config": TrainConfig(epochs=20, device="cuda"),
        "run_seeds": (42, 43, 44),
        "final_seeds": (1000, 1001),
    }

    bayesian_opt = bo.run_experiment(
        "runs/bayesian",
        search_config=bo.SearchConfig(max_evaluations=25),
        **settings,
    )

    dynamic_grid = grid.run_experiment(
        "runs/dynamic_grid",
        search_config=grid.SearchConfig(max_evaluations=25),
        **settings,
    )

    dynamic_rand = rand.run_experiment(
        "runs/dynamic_random",
        search_config=rand.SearchConfig(max_evaluations=25),
        **settings,
    )