"""Example of using training on CartPole."""

# ToDo: report Error in compute actions

import argparse

import ray
from ray import tune
from ray.rllib.contrib.alpha_zero.models.custom_torch_models import DenseModel
from ray.rllib.contrib.alpha_zero.environments.cartpole import CartPole
from ray.rllib.contrib.alpha_zero.core.alpha_zero_trainer import AlphaZeroTrainer
from ray.rllib.models.catalog import ModelCatalog

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-iteration", default=2, type=int)
    args = parser.parse_args()
    ray.init()

    ModelCatalog.register_custom_model("dense_model", DenseModel)

    config = {
            "env": CartPole,
            "num_workers": 5,
            "rollout_fragment_length": 50,
            "train_batch_size": 500,
            "sgd_minibatch_size": 64,
            "lr": 1e-4,
            "num_sgd_iter": 1,
            "mcts_config": {
                "puct_coefficient": 1.5,
                "num_simulations": 100,
                "temperature": 1.0,
                "dirichlet_epsilon": 0.20,
                "dirichlet_noise": 0.03,
                "argmax_tree_policy": False,
                "add_dirichlet_noise": True,
            },
            "ranked_rewards": {
                "enable": True,
            },
            "model": {
                "custom_model": "dense_model",
            },
        }

    results = tune.run(
        "contrib/AlphaZero",
        stop={"training_iteration": args.training_iteration},
        checkpoint_at_end=True,
        max_failures=0,
        config=config
    )

    best_checkpoint = results.get_best_checkpoint(trial=results.get_best_trial(metric="episode_reward_mean",
                                                                               mode="max"),
                                                  metric="episode_reward_mean",
                                                  mode="max")

    config["num_workers"] = 0

    agent = AlphaZeroTrainer(config=config, env=CartPole)
    agent.restore(best_checkpoint)

    # instantiate env class
    env = CartPole()

    # run until episode ends
    episode_reward = 0
    done = False
    obs = env.reset
    while not done:
        print(obs)
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

    print(episode_reward)

    ray.shutdown()