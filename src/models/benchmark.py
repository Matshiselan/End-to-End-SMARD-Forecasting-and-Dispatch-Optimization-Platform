from src.models.backtest import rolling_backtest

def benchmark():

    print("\nRUNNING MODEL BENCHMARKS\n")

    results = {}

    results["lgbm"] = rolling_backtest()

    print("\nFINAL RESULTS\n")

    for model, metrics in results.items():

        print(f"\n{model.upper()}")

        for k,v in metrics.items():

            print(f"{k}: {v:.4f}")

if __name__ == "__main__":

    benchmark()