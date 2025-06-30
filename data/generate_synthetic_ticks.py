
import numpy as np
import pandas as pd
import argparse

def generate_gbm_ticks(S0, mu, sigma, T, dt, n_ticks):
    """
    Generates synthetic tick data using Geometric Brownian Motion.

    Args:
        S0 (float): Initial stock price.
        mu (float): Drift coefficient.
        sigma (float): Volatility coefficient.
        T (float): Total time in years.
        dt (float): Time step in years.
        n_ticks (int): Number of ticks to generate.

    Returns:
        pd.DataFrame: A DataFrame with columns ['timestamp', 'price'].
    """
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps)
    W = np.random.standard_normal(size=n_steps)
    W = np.cumsum(W) * np.sqrt(dt)  # standard brownian motion
    X = (mu - 0.5 * sigma ** 2) * t + sigma * W
    S = S0 * np.exp(X)  # Geometric Brownian Motion

    # Generate random timestamps for the ticks
    timestamps = np.sort(np.random.uniform(0, T, n_ticks))
    prices = np.interp(timestamps, t, S)

    # Add some noise to the prices to make them more realistic
    noise = np.random.normal(0, S.mean() * 0.0001, n_ticks)
    prices += noise

    df = pd.DataFrame({'timestamp': timestamps, 'price': prices})
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic tick data.")
    parser.add_argument('--s0', type=float, default=100, help='Initial stock price')
    parser.add_argument('--mu', type=float, default=0.05, help='Drift')
    parser.add_argument('--sigma', type=float, default=0.2, help='Volatility')
    parser.add_argument('--T', type=float, default=1.0, help='Total time in years')
    parser.add_argument('--dt', type=float, default=1/252.0, help='Time step in years (1/252 is daily)')
    parser.add_argument('--n_ticks', type=int, default=10000, help='Number of ticks')
    parser.add_argument('--output', type=str, default='synthetic_ticks.csv', help='Output CSV file')

    args = parser.parse_args()

    ticks_df = generate_gbm_ticks(args.s0, args.mu, args.sigma, args.T, args.dt, args.n_ticks)
    ticks_df.to_csv(args.output, index=False)
    print(f"Generated {len(ticks_df)} ticks and saved to {args.output}")
