import numpy as np


# We'll use a standard interface similar to gym, but avoiding direct gym dependency
# for the skeleton to keep requirements clean until PPO/SB3 is integrated.
class TradingEnv:
    """
    A reinforcement learning environment for execution and position sizing.
    The agent receives the Meta-Ensemble's probability and the current risk metrics,
    and decides on position sizing (continuous action space).
    """

    def __init__(
        self, prices: np.ndarray, initial_balance: float = 100_000.0, min_balance: float = 50_000.0
    ):
        """
        Parameters
        ----------
        prices : np.ndarray
            1-D array of historical prices used to compute returns.
        initial_balance : float
            Starting cash balance.
        min_balance : float
            Balance threshold below which the episode terminates.
        """
        if prices.ndim != 1 or len(prices) < 2:
            raise ValueError("prices must be a 1-D array with at least 2 elements")
        self.prices = prices.astype(np.float64)
        self.initial_balance = float(initial_balance)
        self.min_balance = float(min_balance)
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = 0
        self.position = 0.0
        return self._get_obs()

    def _get_obs(self):
        price = self.prices[self.current_step]
        # Simple rolling volatility (std of last 20 returns, or 0 if not enough history)
        if self.current_step >= 21:
            returns_window = (
                np.diff(self.prices[self.current_step - 20 : self.current_step + 1])
                / self.prices[self.current_step - 20 : self.current_step]
            )
            volatility = float(np.std(returns_window))
        else:
            volatility = 0.0
        drawdown = (self.balance - self.initial_balance) / self.initial_balance
        return np.array(
            [price, self.balance, self.position, volatility, drawdown], dtype=np.float32
        )

    def step(self, action: float):
        """
        Execute one time-step.

        Parameters
        ----------
        action : float
            Position size as a fraction in [0, 1].

        Returns
        -------
        obs, reward, done, info
        """
        self.position = max(0.0, min(1.0, action))

        # Calculate reward = position_size * price_return
        price_current = self.prices[self.current_step]
        next_step = self.current_step + 1

        if next_step >= len(self.prices):
            # No more data — episode ends with zero reward for this step
            done = True
            reward = 0.0
        else:
            price_next = self.prices[next_step]
            price_return = (price_next - price_current) / price_current
            reward = self.position * price_return
            # Update balance based on PnL
            self.balance += self.balance * self.position * price_return
            done = self.balance < self.min_balance

        # Advance step
        self.current_step = min(next_step, len(self.prices) - 1)

        info = {"balance": self.balance, "position_size": self.position, "step": self.current_step}
        return self._get_obs(), reward, done, info
