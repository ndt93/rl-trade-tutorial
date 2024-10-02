import pandas as pd
import numpy as np
import statsmodels.api as sm


class CSVPriceFeed:

    def __init__(self, file_path: str):
        self.data = pd.read_csv(file_path)
        self.close = self.data['Close']
        self.current_bar = 0

    def __iter__(self):
        self.current_bar = 0
        return self

    def __next__(self):
        if self.current_bar < len(self.close):
            close = self.close.iloc[self.current_bar]
            self.current_bar += 1
            return close
        else:
            raise StopIteration


class RandomWalkPriceFeed:
    
    def __init__(self, initial_price: float, drift: float, volatility: float, n_samples: int = 252):
        """
        :param initial_price: Initial stock price
        :param drift: expected return per bar
        :param volatility: standard deviation of return per bar
        :param n_samples: total number of bars
        """
        self.initial_price = initial_price
        self.s = initial_price
        self.mu = drift
        self.sigma = volatility
        self.n_samples = n_samples
        self._samples_left = n_samples

    def __iter__(self):
        self.s = self.initial_price
        self._samples_left = self.n_samples
        return self

    def __next__(self):
        if self._samples_left is not None and self._samples_left <= 0:
            raise StopIteration
        self.s = self.s * (1 + np.random.normal(self.mu, self.sigma))
        if self._samples_left is not None:
            self._samples_left -= 1
        return self.s


class ARMAPriceFeed:

    def __init__(self, mean: float, std: float, ar: np.array, ma: np.array, n_samples: int = 1000):
        """
        :param mean: mean value of the series
        :param std: standard deviation of random shock
        :param ar: autoregressive coefficients
        :param ma: moving average coefficients
        :param n_samples: total number of bars
        """
        self.std = std
        self.mean = mean
        ar = np.r_[1, -ar]
        ma = np.r_[1, ma]
        self.arma_process = sm.tsa.ArmaProcess(ar=ar, ma=ma)

        self._n_samples = n_samples

    def __iter__(self):
        samples = self.arma_process.generate_sample(nsample=self._n_samples, scale=self.std)
        samples = np.clip(self.mean + samples, a_min=0., a_max=None)
        return iter(samples)
