import os

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

from price_feeds import ARMAPriceFeed, RandomWalkPriceFeed
from broker_env import BrokerEnv, TradeAction
from policy import TraderActorCriticPolicy


OUTPUT_DIR = './output'
MODEL_FILE = 'saved_model_rw'
VEC_NORM_FILE = 'vec_norm_rw'

broker_config = dict(
    initial_equity=1_000_000,
    position_size=100,
    txn_cost=0.02,
    ts_buffer_len=30,
    ts_obs_len=30,
)
time_series_config = dict(
    mean=100,
    std=2.,
    ar=np.array([0.8]),
    ma=np.array([]),
)
episode_len = 1000



def train_model(save_to=OUTPUT_DIR):
    # price_feed = ARMAPriceFeed(
    #     **time_series_config,
    #     n_samples=episode_len
    # )
    price_feed = RandomWalkPriceFeed(
        initial_price=422.8,
        drift=0.0012,
        volatility=0.00784,
        n_samples=282
    )
    env = DummyVecEnv([lambda:BrokerEnv(price_feed, **broker_config)])
    env = VecNormalize(env, training=True)

    policy_kwargs = dict(net_arch=dict(
        ts_len=broker_config['ts_obs_len'],
        conv_channels=(4, 2),
        conv_kernels=(2, 4),
        mlp_dims=(128, 64)
    ))
    model = A2C(TraderActorCriticPolicy, env, verbose=1, tensorboard_log='./tb_log', policy_kwargs=policy_kwargs)
    print(model.policy)

    model.learn(total_timesteps=500_000, log_interval=1000)
    model.save(os.path.join(save_to, MODEL_FILE))
    env.training = False
    env.save(os.path.join(save_to, VEC_NORM_FILE))
    return model, env


def load_model():
    price_feed = ARMAPriceFeed(n_samples=episode_len, **time_series_config)
    env = DummyVecEnv([lambda: BrokerEnv(price_feed, **broker_config)])
    env = VecNormalize.load(os.path.join(OUTPUT_DIR, VEC_NORM_FILE), env)

    model = A2C.load(os.path.join(OUTPUT_DIR, MODEL_FILE))
    return model, env


def test_model():
    model, env = train_model()
    # model, env = load_model()
    obs = env.reset()
    bar = 0
    ep_rewards = []
    cum_reward = 0
    for i in range(100_000):
        action, pred_state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        raw_reward = info[0]['raw_reward']
        cum_reward += raw_reward

        trade_action = TradeAction(action[0])
        if trade_action != TradeAction.HOLD:
            print(f'#{bar}', trade_action.name, raw_reward)

        if done:
            bar = 0
            ep_rewards.append(cum_reward)
            cum_reward = 0
            print("==========")
        else:
            bar += 1

    perm_stats = {
        'ep_rws:': ep_rewards,
        'ep_rw_mean': np.mean(ep_rewards),
    }
    print(perm_stats['ep_rw_mean'])


if __name__ == '__main__':
    test_model()
