
import numpy as np
import pandas as pd
import argparse

def generate_gbm_ticks(S0, mu, sigma, T, dt, n_ticks):
    """
    Generates synthetic tick data using Geometric Brownian Motion, including bid/ask prices and volumes.

    Args:
        S0 (float): Initial stock price.
        mu (float): Drift coefficient.
        sigma (float): Volatility coefficient.
        T (float): Total time in years.
        dt (float): Time step in years.
        n_ticks (int): Number of ticks to generate.

    Returns:
        pd.DataFrame: A DataFrame with columns ['timestamp', 'price', 'bid_price', 'ask_price', 'bid_volume', 'ask_volume'].
    """
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps)
    W = np.random.standard_normal(size=n_steps)
    W = np.cumsum(W) * np.sqrt(dt)  # standard brownian motion
    X = (mu - 0.5 * sigma ** 2) * t + sigma * W
    S = S0 * np.exp(X)  # Geometric Brownian Motion - this will be our "mid_price"

    # Generate random timestamps for the ticks
    timestamps = np.sort(np.random.uniform(0, T, n_ticks))
    mid_prices = np.interp(timestamps, t, S)

    # Add some noise to the mid_prices to make them more realistic
    noise = np.random.normal(0, S.mean() * 0.0001, n_ticks)
    mid_prices += noise
    
    # Simulate bid-ask spread, making it proportional to volatility
    spread = mid_prices * sigma * np.random.uniform(0.0005, 0.0015, n_ticks)
    
    # Calculate bid and ask prices
    bid_prices = mid_prices - spread / 2
    ask_prices = mid_prices + spread / 2
    
    # Simulate a "last trade" price, which could be at the bid, ask, or within the spread
    trade_type = np.random.choice([-1, 0, 1], n_ticks, p=[0.4, 0.2, 0.4]) # -1: sell, 0: mid, 1: buy
    prices = mid_prices + (spread / 2) * trade_type

    # Simulate bid and ask volumes
    # Let's assume volume is inversely related to spread and has some randomness
    base_volume = 100
    bid_volumes = np.random.poisson(base_volume * abs(1 + 0.5 * np.random.randn(n_ticks)))
    ask_volumes = np.random.poisson(base_volume * abs(1 + 0.5 * np.random.randn(n_ticks)))
    bid_volumes[bid_volumes <= 0] = 1
    ask_volumes[ask_volumes <= 0] = 1


    df = pd.DataFrame({
        'timestamp': timestamps, 
        'price': prices,
        'bid_price': bid_prices,
        'ask_price': ask_prices,
        'bid_volume': bid_volumes,
        'ask_volume': ask_volumes
    })
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic tick data.")
    parser.add_argument('--s0', type=float, default=100, help='Initial stock price')
    parser.add_argument('--mu', type=float, default=0.05, help='Drift')
    parser.add_argument('--sigma', type=float, default=0.2, help='Volatility')
    parser.add_argument('--T', type=float, default=1.0, help='Total time in years')
    parser.add_argument('--dt', type=float, default=1/252.0, help='Time step in years (1/252 is daily)')
    parser.add_argument('--n_ticks', type=int, default=10000, help='Number of ticks')
    parser.add_argument('--output', type=str, default='synthetic_ticks_custom.csv', help='Output CSV file')

    args = parser.parse_args()

    ticks_df = generate_gbm_ticks(args.s0, args.mu, args.sigma, args.T, args.dt, args.n_ticks)
    ticks_df.to_csv(args.output, index=False)
    print(f"Generated {len(ticks_df)} ticks and saved to {args.output}")
    print("Columns:", ticks_df.columns.tolist())
