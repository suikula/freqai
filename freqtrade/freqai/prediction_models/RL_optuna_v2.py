import logging
from typing import Any, Dict  # , Tuple

# import numpy.typing as npt
import torch as th
import numpy as np
import optuna
import gym
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines3.common.callbacks import EvalCallback
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


        th.set_num_threads(8)
        sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
        pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)
        
        #func = lambda trial: self.objective(trial, train_df)

        #study = optuna.create_study(direction="maximize")
        study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
        #study.optimize(func, n_trials=45, show_progress_bar=True, n_jobs=15)

        #study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

        try:
            #study.optimize(func, n_trials=N_TRIALS, timeout=600, show_progress_bar=True, n_jobs=15)
            study.optimize(self.objective, data_dictionary, dk, n_trials=N_TRIALS, timeout=600, show_progress_bar=True, n_jobs=15)
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

        return 0



        # policy_kwargs = dict(activation_fn=th.nn.ReLU,
        #                      net_arch=[512, 512, 256, 128])

        # model = self.MODELCLASS(self.policy_type, self.train_env, policy_kwargs=policy_kwargs,
        #                         tensorboard_log=Path(dk.data_path / "tensorboard"),
        #                         **self.freqai_info['model_training_parameters']
        #                         )

        # model.learn(
        #     total_timesteps=int(total_timesteps),
        #     callback=self.eval_callback
        # )

        # if Path(dk.data_path / "best_model.zip").is_file():
        #     logger.info('Callback found a best model.')
        #     best_model = self.MODELCLASS.load(dk.data_path / "best_model")
        #     return best_model

        # logger.info('Couldnt find best model, using final model instead.')

        # return model
    
    def objective(self, trial: optuna.Trial, data_dictionary: Dict[str, Any]) -> float:

        kwargs = {}
        algo = self.freqai_info["rl_config"]["model_type"]
        train_df = data_dictionary["train_features"]
        total_timesteps = self.freqai_info["rl_config"]["train_cycles"] * len(train_df)

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

        #if self._last_trade_tick is None:
        #    return 0.
        rw = 0

        max_trade_duration = 20 # this should be set in config
        pnl = self.get_unrealized_profit()

        # # prevent unwanted actions during trades or when no trades are open
        if action == Actions.Short_enter.value and self._position == Positions.Long:
            return -1
        if action == Actions.Short_exit.value and self._position == Positions.Long:
            return -1
        if action == Actions.Long_enter.value and self._position == Positions.Short:
            return -1
        if action == Actions.Long_exit.value and self._position == Positions.Short:
            return -1

        if action == Actions.Short_exit.value and self._position == Positions.Neutral:
            return -1
        if action == Actions.Long_exit.value and self._position == Positions.Neutral:
            return -1


        # reward for opening trades
        if action == Actions.Short_enter.value and self._position == Positions.Neutral:
            return 1
        if action == Actions.Long_enter.value and self._position == Positions.Neutral:
            return 1

        # close long
        if action == Actions.Long_exit.value and self._position == Positions.Long:
            if pnl >= 0 and pnl < 0.005:
                self.long_winners += 1
                return 0
            if pnl >= 0.005: # this should be set in config
                # aim x2 rw
                if pnl > self.profit_aim * self.rr:
                    rw = 5
                if pnl > self.profit_aim * (self.rr * 2):
                    rw = 10
                if pnl < self.profit_aim * self.rr:
                    rw = 2
                
                # duration rules
                if self._current_tick - self._last_trade_tick <= max_trade_duration:
                    self.long_winners += 1
                    self.long_pnl += pnl
                    return rw
                if self._current_tick - self._last_trade_tick > max_trade_duration:
                    over = (self._current_tick - self._last_trade_tick) - max_trade_duration
                    for i in range(1,9):
                        if over == i:
                            self.long_winners += 1
                            self.long_pnl += pnl
                            return rw*(1-(i/10))
                        elif over >= 9:
                            self.long_winners += 1
                            self.long_pnl += pnl
                            return rw * 0.1

            #punishment for losses
            if pnl < 0:
                self.long_losers += 1
                self.long_pnl += pnl
                return -7
            if pnl < (self.profit_aim * -1) * self.rr:
                self.long_pnl += pnl
                self.long_losers += 1
                return -12


        # close short
        if action == Actions.Short_exit.value and self._position == Positions.Short:
            if pnl >= 0 and pnl < 0.005:
                self.long_winners += 1
                return 0
            if pnl >= 0.005:
                # aim x2 rw
                if pnl > self.profit_aim * self.rr:
                    rw = 5
                if pnl > self.profit_aim * (self.rr * 2):
                    rw = 10
                if pnl < self.profit_aim * self.rr:
                    rw = 2

                # duration rules
                if self._current_tick - self._last_trade_tick <= max_trade_duration:
                    self.short_winners += 1
                    self.short_pnl += pnl
                    return rw
                if self._current_tick - self._last_trade_tick > max_trade_duration:
                    over = (self._current_tick - self._last_trade_tick) - max_trade_duration
                    for i in range(1,9):
                        if over == i:
                            self.short_winners += 1
                            self.short_pnl += pnl
                            return rw*(1-(i/10))
                        elif over >= 9:
                            self.short_winners += 1
                            self.short_pnl += pnl
                            return rw * 0.1
            
            #punishment for losses
            if pnl < 0:
                self.short_losers += 1
                self.short_pnl += pnl
                return -7
            if pnl < (self.profit_aim * -1) * self.rr:
                self.short_losers += 1
                self.short_pnl += pnl
                return -12
        return 0