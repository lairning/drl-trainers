"""The SlateQ algorithm for recommendation"""

import argparse

import ray
from ray.rllib.agents import slateq
from ray.rllib.env.wrappers.recsim_wrapper import env_name as recsim_env_name

ALL_SLATEQ_STRATEGIES = [
    # RANDOM: Randomly select documents for slates.
    "RANDOM",
    # MYOP: Select documents that maximize user click probabilities. This is
    # a myopic strategy and ignores long term rewards. This is equivalent to
    # setting a zero discount rate for future rewards.
    "MYOP",
    # SARSA: Use the SlateQ SARSA learning algorithm.
    "SARSA",
    # QL: Use the SlateQ Q-learning algorithm.
    "QL",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="SARSA")
    parser.add_argument("--stop", type=int, default=1)

    args = parser.parse_args()

    env_config = {
        "slate_size"                      : 2,
        "seed"                            : 0,
        "convert_to_discrete_action_space": False,
    }

    config = {}
    config["num_workers"] = 0
    config["slateq_strategy"] = args.strategy
    config["env_config"] = env_config

    ray.init()

    trainer = slateq.SlateQTrainer(config=config, env=recsim_env_name)

    for i in range(args.stop):
        trainer.train()

    ray.shutdown()


if __name__ == "__main__":
    main()
