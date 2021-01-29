"""The SlateQ algorithm for recommendation"""

import argparse
from datetime import datetime

import ray
from ray import tune
from ray.rllib.env.wrappers.recsim_wrapper import env_name as recsim_env_name

ALL_SLATEQ_STRATEGIES = [
    # RANDOM: Randomly select documents for slates.
    # "RANDOM",
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
    parser.add_argument("--env-slate-size", type=int, default=2)
    parser.add_argument("--env-seed", type=int, default=0)
    parser.add_argument("--stop", type=int, default=1)

    args = parser.parse_args()

    env_config = {
        "slate_size"                      : args.env_slate_size,
        "seed"                            : args.env_seed,
        "convert_to_discrete_action_space": False,
    }

    stop = {
        "training_iteration": args.stop
    }

    ray.init()

    time_signature = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    tune.run(
        "SlateQ",
        stop=stop,
        name="SlateQ-{}".format(time_signature),
        config={
            "env"            : recsim_env_name,
            "num_workers"    : 5,
            "slateq_strategy": tune.grid_search(ALL_SLATEQ_STRATEGIES),
            "env_config"     : env_config,
        }
    )

    ray.shutdown()


if __name__ == "__main__":
    main()
