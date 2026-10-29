"""Step 1: data exploration, cleaning and feature engineering.

Run:  python -m src.preprocess
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def extract_title(name: str) -> str:
    """'Braund, Mr. Owen Harris' -> 'Mr', grouped into a few common titles."""
    title = name.split(",")[1].split(".")[0].strip()
    mapping = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    title = mapping.get(title, title)
    return title if title in ("Mr", "Mrs", "Miss", "Master") else "Rare"


def explore(df: pd.DataFrame) -> None:
    """Print a quick exploration summary (shape, missing values, target balance)."""
    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0])
    print("\nTarget balance:\n", df[config.TARGET].value_counts(normalize=True).round(3))
    print("\nSurvival rate by Sex:\n", df.groupby("Sex")[config.TARGET].mean().round(3))
    print("\nSurvival rate by Pclass:\n", df.groupby("Pclass")[config.TARGET].mean().round(3))


def cap_outliers(series: pd.Series, lower_q=0.01, upper_q=0.99) -> pd.Series:
    """Clip extreme values (e.g. a few very high fares) to the 1st/99th percentile."""
    return series.clip(series.quantile(lower_q), series.quantile(upper_q))


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned DataFrame with the engineered features added."""
    df = df.copy()

    # --- Feature engineering (done first, Name is needed for Title) ---
    df["Title"] = df["Name"].apply(extract_title)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # --- Missing values ---
    # Age: fill with the median age of people with the same title (Master = boys, so younger)
    df["Age"] = df["Age"].fillna(df.groupby("Title")["Age"].transform("median"))
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    # Cabin is ~77% missing, so we drop it rather than guess.

    # --- Outliers ---
    df["Fare"] = cap_outliers(df["Fare"])

    keep = config.FEATURES + [config.TARGET] if config.TARGET in df else config.FEATURES
    return df[keep]


def main() -> None:
    raw = pd.read_csv(config.RAW_DATA)
    explore(raw)

    clean = clean_and_engineer(raw)
    train, test = train_test_split(
        clean, test_size=0.2, random_state=config.RANDOM_STATE, stratify=clean[config.TARGET]
    )

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(config.TRAIN_PATH, index=False)
    test.to_csv(config.TEST_PATH, index=False)
    print(f"\nSaved {len(train)} train rows and {len(test)} test rows to {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
