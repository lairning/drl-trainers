import ray
from ray import tune
from ray.rllib.contrib.alpha_zero.models.custom_torch_models import ActorCriticModel, convert_to_tensor
from ray.rllib.contrib.alpha_zero.core.alpha_zero_trainer import AlphaZeroTrainer
from ray.rllib.models.catalog import ModelCatalog
from ray.tune.registry import register_env
from env import HeartsAlphaEnv

from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import FLOAT_MIN, FLOAT_MAX
from ray.rllib.models.modelv2 import restore_original_dimensions

torch, nn = try_import_torch()

class DenseModel(ActorCriticModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        ActorCriticModel.__init__(self, obs_space, action_space, num_outputs,
                                  model_config, name)

        #print("## DEBUG obs_space.original_space ###", obs_space.original_space)
        N_NEURONS = 256
        self.shared_layers = nn.Sequential(
            nn.Linear(
                in_features= obs_space.original_space["obs"].n,
                out_features=N_NEURONS),
            nn.Linear(in_features=N_NEURONS, out_features=N_NEURONS)
        )
        self.actor_layers = nn.Sequential(
            nn.Linear(in_features=N_NEURONS, out_features=action_space.n))
        self.critic_layers = nn.Sequential(
            nn.Linear(in_features=N_NEURONS, out_features=1))
        self._value_out = None

    def compute_priors_and_value(self, obs):
        obs = convert_to_tensor([self.preprocessor.transform(obs)])
        input_dict = restore_original_dimensions(obs, self.obs_space, "torch")

        with torch.no_grad():
            model_out = self.forward(input_dict, None, [1])
            logits, _ = model_out
            value = self.value_function()
            logits, value = torch.squeeze(logits), torch.squeeze(value)
            priors = nn.Softmax(dim=-1)(logits)

            priors = priors.cpu().numpy()
            value = value.cpu().numpy()

            return priors, value

    def forward(self, input_dict, state, seq_lens):

        action_mask = input_dict["action_mask"]

        x = input_dict["obs"]
        x = self.shared_layers(x)
        # actor outputs
        logits = self.actor_layers(x)

        # compute value
        self._value_out = self.critic_layers(x)

        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)
        return logits + inf_mask, None



if __name__ == "__main__":

    ray.init()

    ModelCatalog.register_custom_model("dense_model", DenseModel)

    register_env(
        "HeartsEnv",
        #lambda _: HeartsEnv()
        lambda _: HeartsAlphaEnv(10)
    )

    config_default = {
                 "env"                    : "HeartsEnv",
                 "num_workers"            : 0,
                 "model"                  : {
                     "custom_model": "dense_model",
                 },
             }

    config_tuned = {
                 "env"                    : "HeartsEnv",
                 "num_workers"            : 0,
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
                     "custom_model": "dense_model",
                 },
             }

    config = config_tuned
    config["num_workers"] = 5

    results = tune.run(
        "contrib/AlphaZero",
        stop={"training_iteration": 30},
        checkpoint_at_end = True,
        max_failures=0,
        config=config,
    )

    best_checkpoint = results.get_best_checkpoint(trial=results.get_best_trial(metric="episode_reward_mean",
                                                                               mode="max"),
                                                  metric="episode_reward_mean",
                                                  mode="max")

    print(best_checkpoint)

    config["num_workers"] = 0

    agent = AlphaZeroTrainer(config=config, env="HeartsEnv")
    agent.restore(best_checkpoint)

    # instantiate env class
    he = HeartsAlphaEnv(10)

    # run until episode ends
    episode_reward = 0
    done = False
    obs = he.reset()
    while not done:
        #print(obs)
        action = agent.compute_action(obs)
        print(he.env.me, he.env.table_card, he._decode_card(action))
        obs, reward, done, info = he.step(action)
        episode_reward += reward
        print(episode_reward,reward)

    ray.shutdown()