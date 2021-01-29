"""The SlateQ algorithm for recommendation"""

import argparse

import ray
from ray.rllib.agents import slateq
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

    # config = slateq.DEFAULT_CONFIG.copy()
    # config["num_gpus"] = 0
    config = {}
    config["num_workers"] = 5
    config["slateq_strategy"] = "QL"
    config["env_config"] = env_config

    ray.init()

    trainer = slateq.SlateQTrainer(config=config, env=recsim_env_name)

    result = trainer.train()
    best_checkpoint = trainer.save()
    best_reward = result['episode_reward_mean']
    print("Mean Reward {}:{}".format(1, result['episode_reward_mean']))

    for i in range(1, args.stop):
        result = trainer.train()
        print("Mean Reward {}:{}".format(i+1, result['episode_reward_mean']))
        best_reward = max(best_reward, result['episode_reward_mean'])
        if best_reward == result['episode_reward_mean']:
            best_checkpoint = trainer.save()

    print("BEST Mean Reward  :", best_reward)
    print("BEST Checkpoint at:", best_checkpoint)
    ray.shutdown()


if __name__ == "__main__":
    main()
