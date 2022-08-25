import logging
from typing import Any, Dict

import torch as th
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv, Positions
from freqtrade.freqai.RL.BaseReinforcementLearningModel import BaseReinforcementLearningModel
from pathlib import Path
from pandas import DataFrame
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
from typing import Callable

logger = logging.getLogger(__name__)


class ReinforcementLearner(BaseReinforcementLearningModel):
    """
    User created Reinforcement Learning Model prediction model.
    """
    def linear_schedule(self, initial_value: float) -> Callable[[float], float]:
        """
        Linear learning rate schedule.

        :param initial_value: Initial learning rate.
        :return: schedule that computes
        current learning rate depending on remaining progress
        """
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining:
            :return: current learning rate
            """
            return progress_remaining * initial_value

        return func

    def fit_rl(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen):

        train_df = data_dictionary["train_features"]
        total_timesteps = self.freqai_info["rl_config"]["train_cycles"] * len(train_df)

        policy_kwargs = dict(activation_fn=th.nn.Tanh,
                             net_arch=[512, 512, 256])

        if dk.pair not in self.dd.model_dictionary or not self.continual_retraining:
            model = self.MODELCLASS(self.policy_type, self.train_env, policy_kwargs=policy_kwargs,
                                    tensorboard_log=Path(dk.data_path / "tensorboard"),
                                    learning_rate=self.linear_schedule(0.001),
                                    clip_range=self.linear_schedule(0.5),
                                    **self.freqai_info['model_training_parameters']
                                    )
        else:
            logger.info('Continual training activated - starting training from previously '
                        'trained agent.')
            model = self.dd.model_dictionary[dk.pair]
            model.tensorboard_log = Path(dk.data_path / "tensorboard")
            model.set_env(self.train_env)

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

    def set_train_and_eval_environments(self, data_dictionary: Dict[str, DataFrame],
                                        prices_train: DataFrame, prices_test: DataFrame,
                                        dk: FreqaiDataKitchen):
        """
        User can override this if they are using a custom MyRLEnv
        """
        train_df = data_dictionary["train_features"]
        test_df = data_dictionary["test_features"]
        eval_freq = self.freqai_info["rl_config"]["eval_cycles"] * len(test_df)

        self.train_env = MyRLEnv(df=train_df, prices=prices_train, window_size=self.CONV_WIDTH,
                                 reward_kwargs=self.reward_params, config=self.config)
        self.eval_env = Monitor(MyRLEnv(df=test_df, prices=prices_test,
                                window_size=self.CONV_WIDTH,
                                reward_kwargs=self.reward_params, config=self.config))
        self.eval_callback = EvalCallback(self.eval_env, deterministic=True,
                                          render=False, eval_freq=eval_freq,
                                          best_model_save_path=str(dk.data_path))


class MyRLEnv(Base5ActionRLEnv):
    """
    User can override any function in BaseRLEnv and gym.Env. Here the user
    sets a custom reward based on profit and trade duration.
    """

    def calculate_reward(self, action):

        #if self._last_trade_tick is None:
        #    return 0.

        # # first, penalize if the action is not valid
        # if not self._is_valid(action):
        #     return -1
        rw = 0

        max_trade_duration = self.rl_config.get('max_trade_duration_candles', 20)
        pnl = self.get_unrealized_profit()

        # reward for opening trades
        if action == Actions.Short_enter.value and self._position == Positions.Neutral:
            return 1
        if action == Actions.Long_enter.value and self._position == Positions.Neutral:
            return 1


        # close long
        if action == Actions.Long_exit.value and self._position == Positions.Long:
            if pnl >= 0 and pnl < 0.005:
                self.long_winners += 1
                self.long_small_profit += 1
                return 0
            if pnl >= 0.005: # this should be set in config
                # aim x2 rw
                if pnl > self.profit_aim * self.rr:
                    rw = 50
                if pnl > self.profit_aim * (self.rr * 2):
                    rw = 100
                if pnl < self.profit_aim * self.rr:
                    rw = 2
                
                # duration rules
                if self._current_tick - self._last_trade_tick <= max_trade_duration:
                    self.long_winners += 1
                    self.long_pnl += pnl
                    self.long_profit += 1
                    return rw
                if self._current_tick - self._last_trade_tick > max_trade_duration:
                    over = (self._current_tick - self._last_trade_tick) - max_trade_duration
                    for i in range(1,9):
                        if over == i:
                            self.long_winners += 1
                            self.long_pnl += pnl
                            self.long_over_profit += 1
                            return rw*(1-(i/10))
                            
                        elif over >= 9:
                            self.long_winners += 1
                            self.long_pnl += pnl
                            self.long_over_over_profit += 1
                            return rw * 0.1

            #punishment for losses
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
            if pnl >= 0 and pnl < 0.005:
                self.short_winners += 1
                self.short_small_profit += 1
                return 0
            if pnl >= 0.005:
                # aim x2 rw
                if pnl > self.profit_aim * self.rr:
                    rw = 50
                if pnl > self.profit_aim * (self.rr * 2):
                    rw = 100
                if pnl < self.profit_aim * self.rr:
                    rw = 2

                # duration rules
                if self._current_tick - self._last_trade_tick <= max_trade_duration:
                    self.short_winners += 1
                    self.short_pnl += pnl
                    self.short_profit += 1
                    return rw
                if self._current_tick - self._last_trade_tick > max_trade_duration:
                    over = (self._current_tick - self._last_trade_tick) - max_trade_duration
                    for i in range(1,9):
                        if over == i:
                            self.short_winners += 1
                            self.short_pnl += pnl
                            self.short_over_profit += 1
                            return rw*(1-(i/10))
                        elif over >= 9:
                            self.short_winners += 1
                            self.short_pnl += pnl
                            self.short_over_over_profit += 1
                            return rw * 0.1
            
            #punishment for losses
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