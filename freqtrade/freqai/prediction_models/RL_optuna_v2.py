import logging
from typing import Any, Dict  # , Tuple

# import numpy.typing as npt
import torch as th
import numpy as np
import optuna
import gym
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3 import PPO
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


from freqtrade.freqai.RL.hyperparams_opt import HYPERPARAMS_SAMPLER
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv, Positions
from freqtrade.freqai.RL.BaseReinforcementLearningModel import BaseReinforcementLearningModel
from pathlib import Path

logger = logging.getLogger(__name__)


N_TRIALS = 100
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = int(2e4)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 3

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy"
}

class ReinforcementLearner_optuna(BaseReinforcementLearningModel):
    """
    User created Reinforcement Learning Model prediction model.
    """

    def fit_rl(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen):

        train_df = data_dictionary["train_features"]
        total_timesteps = self.freqai_info["rl_config"]["train_cycles"] * len(train_df)


        th.set_num_threads(15)
        sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
        pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

        study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

        #func = lambda trial: self.objective(trial, data_dictionary)

        try:
            #study.optimize(func, n_trials=N_TRIALS, timeout=600, show_progress_bar=True, n_jobs=15)
            #study.optimize(self.objective, train_df, N_TRIALS, timeout=600, show_progress_bar=True, n_jobs=2)
            study.optimize(self.objective, N_TRIALS)
        except KeyboardInterrupt:
            pass
        
        
        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        print("  User attrs:")
        for key, value in trial.user_attrs.items():
            print("    {}: {}".format(key, value))

        #return 0



        policy_kwargs = dict(activation_fn=th.nn.ReLU,
                             net_arch=[512, 512, 256, 128])

        model = self.MODELCLASS(self.policy_type, self.train_env,
                                tensorboard_log=Path(dk.data_path / "tensorboard"),
                                **trial.params)
                                #**self.freqai_info['model_training_parameters']
                                #)

        model.learn(
            total_timesteps=int(total_timesteps),
            callback=self.eval_callback
        )

        if Path(dk.data_path / "best_model.zip").is_file():
            logger.info('Callback found a best model.')
            best_model = self.MODELCLASS.load(dk.data_path / "best_model")
            return best_model

        logger.info('Couldnt find best model, using final model instead.')

        return model
    
    def objective(self, trial: optuna.Trial) -> float:

        kwargs = {}
        algo = self.freqai_info["rl_config"]["model_type"]
        #train_df = data_dictionary["train_features"]
        total_timesteps = 5000

        sampled_hyperparams = HYPERPARAMS_SAMPLER[algo](trial)
        kwargs.update(sampled_hyperparams)



        model = PPO(policy="MlpPolicy", env=self.train_env,
                    tensorboard_log=f"/tmp/tensor/", **kwargs)
        # Create env used for evaluation
        eval_env = self.eval_env
        # Create the callback that will periodically evaluate
        # and report the performance
        eval_callback = TrialEvalCallback(
            eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
        )

        nan_encountered = False
        try:
            model.learn(total_timesteps, callback=eval_callback)
        except AssertionError as e:
            # Sometimes, random hyperparams can generate NaN
            print(e)
            nan_encountered = True
        finally:
            # Free memory
            model.env.close()
            eval_env.close()

        # Tell the optimizer that the trial failed
        if nan_encountered:
            return float("nan")

        if eval_callback.is_pruned:
            raise optuna.exceptions.TrialPruned()

        return eval_callback.last_mean_reward

class TrialEvalCallback(EvalCallback):
    from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv, Positions
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            Base5ActionRLEnv().step()
            #super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True

class MyRLEnv(Base5ActionRLEnv):
    """
    User can modify any part of the environment by overriding base
    functions
    """
    def calculate_reward(self, action):

        rw = 0

        # # first, penalize if the action is not valid
        # if not self._is_valid(action):
        #     return -15

        max_trade_duration = self.rl_config.get('max_trade_duration_candles', 20)
        min_profit = self.rl_config.get('min_profit', 0.005)
        pnl = self.get_unrealized_profit()

        # reward for opening trades
        if action == Actions.Short_enter.value and self._position == Positions.Neutral:
            return 10
        if action == Actions.Long_enter.value and self._position == Positions.Neutral:
            return 10

        # close long
        if action == Actions.Long_exit.value and self._position == Positions.Long:
            if pnl >= 0 and pnl < min_profit:
                self.long_winners += 1
                self.long_small_profit += 1
                return 0
            if pnl >= min_profit:
                if pnl > self.profit_aim * self.rr:
                    rw = 50
                if pnl > self.profit_aim * (self.rr * 2):
                    rw = 100

                # duration rules
                if self._current_tick - self._last_trade_tick <= max_trade_duration:
                    self.long_winners += 1
                    self.long_pnl += pnl
                    self.long_profit += 1
                    return rw
                if self._current_tick - self._last_trade_tick > max_trade_duration:
                    over = (self._current_tick - self._last_trade_tick) - max_trade_duration
                    for i in range(1, 9):
                        if over == i:
                            self.long_winners += 1
                            self.long_pnl += pnl
                            self.long_over_profit += 1
                            return rw * (1 - (i / 10))

                        elif over >= 9:
                            self.long_winners += 1
                            self.long_pnl += pnl
                            self.long_over_over_profit += 1
                            return rw * 0.1

            # punishment for losses
            if pnl < 0:
                self.long_losers += 1
                self.long_pnl += pnl
                self.long_loss += 1
                return -50
            if pnl < (self.profit_aim * -1) * self.rr:
                self.long_pnl += pnl
                self.long_losers += 1
                self.long_big_loss += 1
                return -100

        # close short
        if action == Actions.Short_exit.value and self._position == Positions.Short:
            if pnl >= 0 and pnl < min_profit:
                self.short_winners += 1
                self.short_small_profit += 1
                return 0
            if pnl >= min_profit:
                # aim x2 rw
                if pnl > self.profit_aim * self.rr:
                    rw = 50
                if pnl > self.profit_aim * (self.rr * 2):
                    rw = 100

                # duration rules
                if self._current_tick - self._last_trade_tick <= max_trade_duration:
                    self.short_winners += 1
                    self.short_pnl += pnl
                    self.short_profit += 1
                    return rw
                if self._current_tick - self._last_trade_tick > max_trade_duration:
                    over = (self._current_tick - self._last_trade_tick) - max_trade_duration
                    for i in range(1, 9):
                        if over == i:
                            self.short_winners += 1
                            self.short_pnl += pnl
                            self.short_over_profit += 1
                            return rw * (1 - (i / 10))
                        elif over >= 9:
                            self.short_winners += 1
                            self.short_pnl += pnl
                            self.short_over_over_profit += 1
                            return rw * 0.1

            # punishment for losses
            if pnl < 0:
                self.short_losers += 1
                self.short_pnl += pnl
                self.short_loss += 1
                return -50
            if pnl < (self.profit_aim * -1) * self.rr:
                self.short_losers += 1
                self.short_pnl += pnl
                self.short_big_loss += 1
                return -100
        return 0
