from enum import IntEnum
from typing import Iterable, Optional
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

class TradeAction(IntEnum):
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class BrokerState:

    NON_TS_FEATURES = 3

    equity: float
    open_position_size: int
    entry_price: float
    ts_buffer: np.array
    ts_obs_len: int

    def to_obs(self):
        return np.r_[
            self.equity,
            self.open_position_size,
            self.entry_price,
            self.ts_buffer[-self.ts_obs_len:] if self.ts_obs_len > 0 else []
        ].astype(np.float32)


class BrokerEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
            self,
            price_feed: Iterable,
            initial_equity: float = 10_000,
            txn_cost: float = 0.1,
            position_size=1,
            ts_buffer_len: int = 100,
            ts_obs_len: int = 30,
    ):
        super().__init__()

        self._price_feed = price_feed
        self._price_feed_iter = None
        self._initial_equity = initial_equity
        self._txn_cost = txn_cost / 100
        self._position_size = position_size

        assert ts_obs_len <= ts_buffer_len
        self._ts_buffer_len = ts_buffer_len
        self._ts_obs_len = ts_obs_len

        self._state = None  # type: Optional[BrokerState]
        self._terminated = False

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(ts_obs_len + BrokerState.NON_TS_FEATURES,),
            dtype=np.float32
        )

    def step(self, action):
        reward = 0
        truncated = False

        if self._terminated:
            return None, reward, self._terminated, truncated, self._get_info(None, reward)

        price = None
        open_position = self._state.open_position_size

        try:
            price = self._update_price()
            self._state.equity += (price - self._state.ts_buffer[-1]) * self._state.open_position_size

            if self._state.equity <= 0:
                self._terminated = True
                reward = -self._initial_equity
            elif open_position:
                if action != TradeAction.HOLD:
                    reward = (price - self._state.entry_price) * open_position
                    close_position = (
                        (open_position > 0 and action == TradeAction.SELL) or
                        (open_position < 0 and action == TradeAction.BUY)
                    )
                    if close_position:
                        self._exit_position()
                        reward -= self._get_txn_cost(price)
                    else:
                        self._enter_position(action, price)
                        reward -= 2*self._get_txn_cost(price)
            else:
                if action != TradeAction.HOLD:
                    self._enter_position(action, price)
                    reward -= self._get_txn_cost(price)
        except StopIteration:
            truncated = True
            self._terminated = True

        obs = self._state.to_obs()
        return obs, reward, self._terminated, truncated, self._get_info(price=price, reward=reward)

    def _enter_position(self, action: TradeAction, price: float):
        self._state.open_position_size = (
            self._position_size if action == TradeAction.BUY
            else -1 * self._position_size
        )
        self._state.entry_price = price

    def _exit_position(self):
        self._state.open_position_size = 0
        self._state.entry_price = 0

    def _get_txn_cost(self, price):
        return price * self._position_size * self._txn_cost

    def reset(self, seed=None, options=None):
        np.random.seed(seed)
        self._price_feed_iter = iter(self._price_feed)
        self._state = BrokerState(
            equity=self._initial_equity,
            entry_price=0,
            open_position_size=0,
            ts_buffer=np.array([next(self._price_feed_iter) for _ in range(self._ts_buffer_len)]),
            ts_obs_len=self._ts_obs_len,
        )
        self._terminated = False
        return self._state.to_obs(), self._get_info(price=None, reward=0)

    def render(self):
        return None

    def close(self):
        pass

    def _update_price(self):
        price = next(self._price_feed_iter)
        price_series = self._state.ts_buffer
        price_series = np.roll(price_series, -1)
        price_series[-1] = price
        self._state.ts_buffer = price_series
        return price

    def _get_info(self, price: float|None, reward: float):
        return {'raw_price': price, 'raw_reward': reward}