import ray
from ray import tune
# from ray.rllib.contrib.alpha_zero.core.alpha_zero_trainer import AlphaZeroTrainer
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.registry import register_env
from env import HeartsAlphaEnv1
from models import AlphaHeartsModel

# from ray.rllib.utils.framework import try_import_torch

# torch, nn = try_import_torch()

if __name__ == "__main__":

    ray.init()

    ModelCatalog.register_custom_model("custom_model", AlphaHeartsModel)

    register_env(
        "HeartsEnv",
        lambda _: HeartsAlphaEnv1(random_players=False)
    )

    config_default = {
                 "env"                    : "HeartsEnv",
                 "model"                  : {
                     "custom_model": "custom_model",
                 },
             }

    config_tuned = {
                 "env"                    : "HeartsEnv",
                 "rollout_fragment_length": 200,
                 "train_batch_size"       : 4000,
                 "sgd_minibatch_size"     : 128,
                 "lr"                     : 1e-4,
                 "num_sgd_iter"           : 30,
                 "mcts_config"            : {
                     "puct_coefficient"   : 1.0,
                     "num_simulations"    : 30,
                     "temperature"        : 1.5,
                     "dirichlet_epsilon"  : 0.25,
                     "dirichlet_noise"    : 0.03,
                     "argmax_tree_policy" : False,
                     "add_dirichlet_noise": True,
                 },
                 "ranked_rewards"         : {
                     "enable": True,
                 },
                 "model"                  : {
                     "custom_model": "custom_model",
                 },
             }

    config = config_default
    config["num_workers"] = 0

    results = tune.run(
        "contrib/AlphaZero",
        stop={"training_iteration": 5},
        checkpoint_at_end = True,
        max_failures=0,
        config=config,
    )


    ray.shutdown()