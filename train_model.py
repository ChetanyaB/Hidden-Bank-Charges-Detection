
import argparse
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from src.model import train


def main():
    parser = argparse.ArgumentParser(description="Train Hidden Charge Detection Models")
    parser.add_argument("--data",  default="data/bank_transactions.csv",
                        help="Path to training CSV")
    parser.add_argument("--models", default="models",
                        help="Directory to save model artifacts")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"❌ Data file not found: {args.data}")
        print("   Download a sample dataset or point to your own CSV.")
        sys.exit(1)

    print(f"📂 Loading data from: {args.data}")
    df = pd.read_csv(args.data)
    print(f"   Loaded {len(df):,} transactions")

    required = ["is_hidden_charge", "hidden_charge_type"]
    for col in required:
        if col not in df.columns:
            print(f"❌ Column '{col}' not found. "
                  "Training data must have ground truth labels.")
            sys.exit(1)

    metrics = train(df, save_dir=args.models)
    print("\n📊 Final Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"   {k:30s}: {v:.4f}")
        else:
            print(f"   {k:30s}: {v}")


if __name__ == "__main__":
    main()
