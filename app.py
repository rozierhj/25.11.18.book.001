import io
from contextlib import redirect_stdout
from textwrap import dedent

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Python Data Science Practice Lab",
    page_icon="🐍",
    layout="wide",
)

DATASETS = {
    "bike_rentals": {
        "name": "Bike Rentals",
        "path": "data/bike_rentals.csv",
        "description": "Daily bike rental counts with temperature readings.",
    },
    "customer_churn": {
        "name": "Customer Churn",
        "path": "data/customer_churn.csv",
        "description": "Subscription tenure, pricing, contracts, and churn labels.",
    },
    "movie_ratings": {
        "name": "Movie Ratings",
        "path": "data/movie_ratings.csv",
        "description": "User movie reviews with genre and viewing time.",
    },
}


@st.cache_data
def load_dataset(dataset_key: str) -> pd.DataFrame:
    dataset = DATASETS[dataset_key]
    return pd.read_csv(dataset["path"])


def beginner_exercises() -> list[dict]:
    return [
        {
            "title": "Inspect the rental data",
            "dataset": "bike_rentals",
            "prompt": "Load the bike rentals data into df and report the number of rows/columns.",
            "hints": ["Use df.shape to return (rows, columns).", "Remember to import pandas as pd."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/bike_rentals.csv")
rows, cols = df.shape
print(f"Rows: {rows}, Columns: {cols}")
""",
        },
        {
            "title": "Convert dates",
            "dataset": "bike_rentals",
            "prompt": "Parse the Day column as datetime and set it as the index sorted in ascending order.",
            "hints": ["Use pd.to_datetime to convert the column.", "df.set_index can assign the new index."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/bike_rentals.csv")
df["Day"] = pd.to_datetime(df["Day"])
df = df.set_index("Day").sort_index()
print(df.head())
""",
        },
        {
            "title": "Filter hot days",
            "dataset": "bike_rentals",
            "prompt": "Select rows where TemperatureC is above 29 degrees.",
            "hints": ["Use a boolean mask with df[df[""TemperatureC""] > 29]."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/bike_rentals.csv")
hot_days = df[df["TemperatureC"] > 29]
print(hot_days)
""",
        },
        {
            "title": "Add a heat flag",
            "dataset": "bike_rentals",
            "prompt": "Create a column is_very_hot that is True when TemperatureC >= 30.",
            "hints": ["Use vectorized comparisons to build the column."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/bike_rentals.csv")
df["is_very_hot"] = df["TemperatureC"] >= 30
print(df[["TemperatureC", "is_very_hot"]])
""",
        },
        {
            "title": "Average rentals",
            "dataset": "bike_rentals",
            "prompt": "Calculate the mean RentalCount for the dataset.",
            "hints": ["Use df[""RentalCount""].mean()."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/bike_rentals.csv")
print(df["RentalCount"].mean())
""",
        },
        {
            "title": "Summary stats",
            "dataset": "bike_rentals",
            "prompt": "Generate descriptive statistics for numeric columns.",
            "hints": ["df.describe() summarizes numeric columns."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/bike_rentals.csv")
print(df.describe())
""",
        },
        {
            "title": "Sort by popularity",
            "dataset": "bike_rentals",
            "prompt": "Sort rows by RentalCount in descending order.",
            "hints": ["df.sort_values accepts ascending=False."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/bike_rentals.csv")
sorted_df = df.sort_values("RentalCount", ascending=False)
print(sorted_df.head())
""",
        },
        {
            "title": "Temperature variance",
            "dataset": "bike_rentals",
            "prompt": "Compute the variance of TemperatureC.",
            "hints": ["df.var() gives variance; select the column first."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/bike_rentals.csv")
print(df["TemperatureC"].var())
""",
        },
        {
            "title": "Top three rental days",
            "dataset": "bike_rentals",
            "prompt": "Return the three days with the highest RentalCount.",
            "hints": ["Combine sort_values with head."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/bike_rentals.csv")
top_days = df.sort_values("RentalCount", ascending=False).head(3)
print(top_days)
""",
        },
        {
            "title": "Normalize rentals",
            "dataset": "bike_rentals",
            "prompt": "Create a column rentals_per_100 normalizing RentalCount per 100 rides.",
            "hints": ["Divide RentalCount by 100 for a quick normalization."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/bike_rentals.csv")
df["rentals_per_100"] = df["RentalCount"] / 100
print(df[["RentalCount", "rentals_per_100"]].head())
""",
        },
        {
            "title": "Column renaming",
            "dataset": "bike_rentals",
            "prompt": "Rename RentalCount to TotalRentals.",
            "hints": ["Use df.rename with columns mapping."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/bike_rentals.csv")
renamed = df.rename(columns={"RentalCount": "TotalRentals"})
print(renamed.head())
""",
        },
        {
            "title": "Check missing values",
            "dataset": "bike_rentals",
            "prompt": "Count missing values in each column.",
            "hints": ["df.isna().sum() tallies missing entries."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/bike_rentals.csv")
print(df.isna().sum())
""",
        },
        {
            "title": "Temperature bins",
            "dataset": "bike_rentals",
            "prompt": "Bin TemperatureC into cool (<27), warm (27-29), and hot (>=29).",
            "hints": ["Use pd.cut with appropriate bins and labels."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/bike_rentals.csv")
bins = [0, 27, 29, df["TemperatureC"].max() + 1]
labels = ["cool", "warm", "hot"]
df["temp_band"] = pd.cut(df["TemperatureC"], bins=bins, labels=labels, right=False)
print(df[["TemperatureC", "temp_band"]])
""",
        },
        {
            "title": "Daily rental delta",
            "dataset": "bike_rentals",
            "prompt": "Calculate the day-over-day difference in RentalCount.",
            "hints": ["Use df.diff on the RentalCount column."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/bike_rentals.csv")
df["rental_change"] = df["RentalCount"].diff()
print(df[["RentalCount", "rental_change"]])
""",
        },
        {
            "title": "Highest temperature day",
            "dataset": "bike_rentals",
            "prompt": "Find the row with the maximum TemperatureC.",
            "hints": ["idxmax returns the index of the max value."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/bike_rentals.csv")
highest = df.loc[df["TemperatureC"].idxmax()]
print(highest)
""",
        },
        {
            "title": "Median rentals",
            "dataset": "bike_rentals",
            "prompt": "Compute the median of RentalCount.",
            "hints": ["Use df.median() on the column."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/bike_rentals.csv")
print(df["RentalCount"].median())
""",
        },
        {
            "title": "Percent change",
            "dataset": "bike_rentals",
            "prompt": "Calculate percent change in RentalCount compared to the previous day.",
            "hints": ["pct_change computes percentage differences."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/bike_rentals.csv")
df["rental_pct_change"] = df["RentalCount"].pct_change() * 100
print(df[["RentalCount", "rental_pct_change"]])
""",
        },
        {
            "title": "Count hot days",
            "dataset": "bike_rentals",
            "prompt": "How many days have TemperatureC above 28 degrees?",
            "hints": ["Use a boolean mask then len() or sum()."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/bike_rentals.csv")
count_hot = (df["TemperatureC"] > 28).sum()
print(count_hot)
""",
        },
        {
            "title": "Reorder columns",
            "dataset": "bike_rentals",
            "prompt": "Display Day and RentalCount columns first, followed by the rest.",
            "hints": ["Rebuild the column list before subsetting."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/bike_rentals.csv")
columns = ["Day", "RentalCount"] + [col for col in df.columns if col not in {"Day", "RentalCount"}]
print(df[columns].head())
""",
        },
        {
            "title": "Save cleaned data",
            "dataset": "bike_rentals",
            "prompt": "Save a version with TemperatureC rounded to one decimal place.",
            "hints": ["Use round and df.to_csv."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/bike_rentals.csv")
df["TemperatureC"] = df["TemperatureC"].round(1)
df.to_csv("data/bike_rentals_clean.csv", index=False)
print(df.head())
""",
        },
        {
            "title": "Document your steps",
            "dataset": "bike_rentals",
            "prompt": "Add comments explaining each transformation you apply.",
            "hints": ["Use # to add concise comments above key lines."],
            "solution": """
import pandas as pd

# Load the dataset
# df = pd.read_csv("data/bike_rentals.csv")
# Add a rentals_per_100 column for an easy-to-read scale
# df["rentals_per_100"] = df["RentalCount"] / 100
# Preview the new feature
print(df.head())
""",
        },
    ]


def intermediate_exercises() -> list[dict]:
    return [
        {
            "title": "Churn rate",
            "dataset": "customer_churn",
            "prompt": "Compute the overall churn rate as a percentage.",
            "hints": ["Create a boolean mask for 'Yes' then take the mean and multiply by 100."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/customer_churn.csv")
churn_rate = (df["Churned"] == "Yes").mean() * 100
print(round(churn_rate, 2))
""",
        },
        {
            "title": "Tenure buckets",
            "dataset": "customer_churn",
            "prompt": "Bucket customers into 0-6, 6-18, and 18+ months of tenure.",
            "hints": ["Use pd.cut with bins and labels."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/customer_churn.csv")
bins = [0, 6, 18, df["TenureMonths"].max() + 1]
labels = ["0-6", "6-18", "18+"]
df["tenure_bucket"] = pd.cut(df["TenureMonths"], bins=bins, labels=labels, right=False)
print(df[["CustomerID", "tenure_bucket"]])
""",
        },
        {
            "title": "Average charge by contract",
            "dataset": "customer_churn",
            "prompt": "Find average MonthlyCharges grouped by ContractType.",
            "hints": ["Group by ContractType then aggregate mean."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/customer_churn.csv")
summary = df.groupby("ContractType")["MonthlyCharges"].mean()
print(summary)
""",
        },
        {
            "title": "Churn by contract",
            "dataset": "customer_churn",
            "prompt": "Compute churn rate per ContractType.",
            "hints": ["Groupby then apply mean on the churn boolean mask."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/customer_churn.csv")
df["churn_flag"] = df["Churned"] == "Yes"
churn_by_contract = df.groupby("ContractType")["churn_flag"].mean() * 100
print(churn_by_contract.round(2))
""",
        },
        {
            "title": "Monthly revenue",
            "dataset": "customer_churn",
            "prompt": "Estimate monthly revenue assuming active customers pay their monthly charges.",
            "hints": ["Filter to churned == 'No' before summing MonthlyCharges."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/customer_churn.csv")
active = df[df["Churned"] == "No"]
revenue = active["MonthlyCharges"].sum()
print(revenue)
""",
        },
        {
            "title": "Identify high spenders",
            "dataset": "customer_churn",
            "prompt": "Label customers paying more than $80 as premium_spender.",
            "hints": ["Create a boolean column using a comparison."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/customer_churn.csv")
df["premium_spender"] = df["MonthlyCharges"] > 80
print(df[["CustomerID", "MonthlyCharges", "premium_spender"]])
""",
        },
        {
            "title": "Charge z-scores",
            "dataset": "customer_churn",
            "prompt": "Compute z-scores for MonthlyCharges to spot outliers.",
            "hints": ["Subtract the mean and divide by the standard deviation."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/customer_churn.csv")
mean = df["MonthlyCharges"].mean()
std = df["MonthlyCharges"].std()
df["charge_zscore"] = (df["MonthlyCharges"] - mean) / std
print(df[["CustomerID", "charge_zscore"]])
""",
        },
        {
            "title": "Retention simulation",
            "dataset": "customer_churn",
            "prompt": "Assume a 10% discount lowers churn probability by 15%. Create a projected churn flag.",
            "hints": ["Start from the churn_flag boolean and multiply by 0.85."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/customer_churn.csv")
df["churn_flag"] = df["Churned"] == "Yes"
df["projected_churn_prob"] = df["churn_flag"].astype(float) * 0.85
print(df[["CustomerID", "projected_churn_prob"]])
""",
        },
        {
            "title": "Top churn risks",
            "dataset": "customer_churn",
            "prompt": "Return the three highest monthly charge customers who churned.",
            "hints": ["Filter churned == 'Yes' then sort descending."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/customer_churn.csv")
churned = df[df["Churned"] == "Yes"].sort_values("MonthlyCharges", ascending=False).head(3)
print(churned)
""",
        },
        {
            "title": "Contract pivot",
            "dataset": "customer_churn",
            "prompt": "Create a pivot table showing average MonthlyCharges by ContractType and Churned.",
            "hints": ["Use pd.pivot_table with values, index, and columns."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/customer_churn.csv")
pivot = pd.pivot_table(
    df,
    values="MonthlyCharges",
    index="ContractType",
    columns="Churned",
    aggfunc="mean",
)
print(pivot)
""",
        },
        {
            "title": "Tenure weighted charges",
            "dataset": "customer_churn",
            "prompt": "Calculate tenure-weighted average MonthlyCharges (weight by TenureMonths).",
            "hints": ["Use np.average with weights or (value*weight).sum()/weight.sum()."],
            "solution": """
import pandas as pd
import numpy as np

df = pd.read_csv("data/customer_churn.csv")
weighted_avg = np.average(df["MonthlyCharges"], weights=df["TenureMonths"])
print(weighted_avg)
""",
        },
        {
            "title": "Quick dashboard text",
            "dataset": "customer_churn",
            "prompt": "Generate a short text summary of total customers, churn count, and revenue.",
            "hints": ["Use len, sums, and f-strings to format the summary."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/customer_churn.csv")
customers = len(df)
churned = (df["Churned"] == "Yes").sum()
revenue = df[df["Churned"] == "No"]["MonthlyCharges"].sum()
print(f"Customers: {customers}, Churned: {churned}, Revenue: ${revenue:.2f}")
""",
        },
        {
            "title": "Genre popularity",
            "dataset": "movie_ratings",
            "prompt": "Count ratings per Genre sorted descending.",
            "hints": ["Use value_counts() on the Genre column."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/movie_ratings.csv")
print(df["Genre"].value_counts())
""",
        },
        {
            "title": "Average rating per movie",
            "dataset": "movie_ratings",
            "prompt": "Compute the mean Rating for each Movie.",
            "hints": ["Group by Movie and take mean of Rating."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/movie_ratings.csv")
movie_means = df.groupby("Movie")["Rating"].mean()
print(movie_means)
""",
        },
        {
            "title": "Longest watch time",
            "dataset": "movie_ratings",
            "prompt": "Find the movie with the highest total WatchMinutes.",
            "hints": ["Group by Movie, sum WatchMinutes, then sort."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/movie_ratings.csv")
watch_totals = df.groupby("Movie")["WatchMinutes"].sum().sort_values(ascending=False)
print(watch_totals.head(1))
""",
        },
        {
            "title": "Center ratings",
            "dataset": "movie_ratings",
            "prompt": "Subtract the overall average rating from each row to create centered_rating.",
            "hints": ["Compute df[""Rating""].mean() and subtract it."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/movie_ratings.csv")
avg = df["Rating"].mean()
df["centered_rating"] = df["Rating"] - avg
print(df[["Rating", "centered_rating"]].head())
""",
        },
        {
            "title": "User profile table",
            "dataset": "movie_ratings",
            "prompt": "Create a table with each UserID, their mean rating, and total watch minutes.",
            "hints": ["Group by UserID and aggregate mean Rating and sum WatchMinutes."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/movie_ratings.csv")
profiles = df.groupby("UserID").agg(mean_rating=("Rating", "mean"), total_minutes=("WatchMinutes", "sum"))
print(profiles)
""",
        },
        {
            "title": "Genre share",
            "dataset": "movie_ratings",
            "prompt": "Compute each Genre's share of total watch time as a percentage.",
            "hints": ["Divide each genre's WatchMinutes sum by the total and multiply by 100."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/movie_ratings.csv")
total_time = df["WatchMinutes"].sum()
genre_share = df.groupby("Genre")["WatchMinutes"].sum() / total_time * 100
print(genre_share.round(2))
""",
        },
        {
            "title": "Recommend top genre",
            "dataset": "movie_ratings",
            "prompt": "Identify the highest-rated genre based on mean Rating and print a recommendation sentence.",
            "hints": ["Group by Genre, mean on Rating, then idxmax."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/movie_ratings.csv")
genre_means = df.groupby("Genre")["Rating"].mean()
top_genre = genre_means.idxmax()
print(f"Viewers love {top_genre} films the most with an average rating of {genre_means[top_genre]:.2f}.")
""",
        },
    ]


def advanced_exercises() -> list[dict]:
    return [
        {
            "title": "Manual train/test split",
            "dataset": "customer_churn",
            "prompt": "Split churn data into 70/30 train-test sets using sklearn.model_selection.train_test_split.",
            "hints": ["Import train_test_split and pass test_size=0.3 with a random_state."],
            "solution": """
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/customer_churn.csv")
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["Churned"])
print(train_df.shape, test_df.shape)
""",
        },
        {
            "title": "One-hot encode contract",
            "dataset": "customer_churn",
            "prompt": "Create one-hot encoded columns for ContractType and show the head of the encoded frame.",
            "hints": ["Use pd.get_dummies with prefix."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/customer_churn.csv")
encoded = pd.get_dummies(df, columns=["ContractType"], prefix="contract")
print(encoded.head())
""",
        },
        {
            "title": "Simple churn model",
            "dataset": "customer_churn",
            "prompt": "Train a logistic regression to predict Churned from TenureMonths and MonthlyCharges.",
            "hints": ["Use sklearn.linear_model.LogisticRegression and fit on numeric columns."],
            "solution": """
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

df = pd.read_csv("data/customer_churn.csv")
X = df[["TenureMonths", "MonthlyCharges"]]
y = (df["Churned"] == "Yes").astype(int)
model = make_pipeline(StandardScaler(), LogisticRegression())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
model.fit(X_train, y_train)
print("Test accuracy:", model.score(X_test, y_test))
""",
        },
        {
            "title": "Precision and recall",
            "dataset": "customer_churn",
            "prompt": "Evaluate the logistic regression model with precision and recall scores.",
            "hints": ["Use sklearn.metrics.precision_score and recall_score after predictions."],
            "solution": """
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score

df = pd.read_csv("data/customer_churn.csv")
X = df[["TenureMonths", "MonthlyCharges"]]
y = (df["Churned"] == "Yes").astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
model = make_pipeline(StandardScaler(), LogisticRegression())
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
""",
        },
        {
            "title": "Cross-validation",
            "dataset": "customer_churn",
            "prompt": "Run 5-fold cross-validation accuracy for the churn model.",
            "hints": ["Use cross_val_score with cv=5."],
            "solution": """
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/customer_churn.csv")
X = df[["TenureMonths", "MonthlyCharges"]]
y = (df["Churned"] == "Yes").astype(int)
model = make_pipeline(StandardScaler(), LogisticRegression())
scores = cross_val_score(model, X, y, cv=5)
print("CV accuracy:", scores.mean())
""",
        },
        {
            "title": "Feature scaling comparison",
            "dataset": "customer_churn",
            "prompt": "Compare accuracy of logistic regression with and without StandardScaler.",
            "hints": ["Train two pipelines and report their test scores."],
            "solution": """
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/customer_churn.csv")
X = df[["TenureMonths", "MonthlyCharges"]]
y = (df["Churned"] == "Yes").astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

scaled = make_pipeline(StandardScaler(), LogisticRegression())
plain = LogisticRegression()
scaled.fit(X_train, y_train)
plain.fit(X_train, y_train)
print("Scaled accuracy:", scaled.score(X_test, y_test))
print("Unscaled accuracy:", plain.score(X_test, y_test))
""",
        },
        {
            "title": "SHAP-style intuition",
            "dataset": "customer_churn",
            "prompt": "Compute coefficient weights of the logistic regression to interpret influence of features.",
            "hints": ["Inspect the coef_ attribute after fitting."],
            "solution": """
import pandas as pd
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("data/customer_churn.csv")
X = df[["TenureMonths", "MonthlyCharges"]]
y = (df["Churned"] == "Yes").astype(int)
model = LogisticRegression()
model.fit(X, y)
for feature, weight in zip(X.columns, model.coef_[0]):
    print(feature, weight)
""",
        },
        {
            "title": "Movie similarity matrix",
            "dataset": "movie_ratings",
            "prompt": "Create a user-movie rating matrix and compute pairwise correlation between movies.",
            "hints": ["Pivot with users as rows, movies as columns, then corr()."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/movie_ratings.csv")
ratings_matrix = df.pivot_table(values="Rating", index="UserID", columns="Movie")
correlations = ratings_matrix.corr()
print(correlations)
""",
        },
        {
            "title": "Top correlated movies",
            "dataset": "movie_ratings",
            "prompt": "For 'Quantum Quest', list the top 2 correlated movies using the correlation matrix.",
            "hints": ["Use sort_values on the correlation column, dropping NaNs and itself."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/movie_ratings.csv")
ratings_matrix = df.pivot_table(values="Rating", index="UserID", columns="Movie")
correlations = ratings_matrix.corr()
top_matches = correlations["Quantum Quest"].drop(labels=["Quantum Quest"]).dropna().sort_values(ascending=False).head(2)
print(top_matches)
""",
        },
        {
            "title": "Content summary",
            "dataset": "movie_ratings",
            "prompt": "Build a descriptive summary per Genre including count, mean rating, and mean watch time.",
            "hints": ["Use groupby.agg with multiple aggregations."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/movie_ratings.csv")
summary = df.groupby("Genre").agg(
    rating_count=("Rating", "count"),
    mean_rating=("Rating", "mean"),
    mean_watch=("WatchMinutes", "mean"),
)
print(summary)
""",
        },
        {
            "title": "Cold-start fallback",
            "dataset": "movie_ratings",
            "prompt": "Create a function recommend_default that returns the globally highest-rated movie when user data is missing.",
            "hints": ["Sort mean ratings descending and return the top title."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/movie_ratings.csv")
movie_means = df.groupby("Movie")["Rating"].mean()

def recommend_default():
    return movie_means.sort_values(ascending=False).index[0]

print(recommend_default())
""",
        },
        {
            "title": "Watch-time weighted ratings",
            "dataset": "movie_ratings",
            "prompt": "Weight ratings by WatchMinutes to favor fully watched movies in the average per genre.",
            "hints": ["Compute (Rating*WatchMinutes).sum() / WatchMinutes.sum() for each genre."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/movie_ratings.csv")
weighted = (
    df.assign(weighted_rating=lambda d: d["Rating"] * d["WatchMinutes"])
    .groupby("Genre")
    .apply(lambda g: g["weighted_rating"].sum() / g["WatchMinutes"].sum())
)
print(weighted)
""",
        },
        {
            "title": "Outlier detection",
            "dataset": "bike_rentals",
            "prompt": "Flag rental counts more than 1.5 IQR above the third quartile.",
            "hints": ["Compute Q1, Q3, then IQR = Q3-Q1."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/bike_rentals.csv")
q1 = df["RentalCount"].quantile(0.25)
q3 = df["RentalCount"].quantile(0.75)
iqr = q3 - q1
upper = q3 + 1.5 * iqr
df["rental_outlier"] = df["RentalCount"] > upper
print(df[["RentalCount", "rental_outlier"]])
""",
        },
        {
            "title": "Rolling averages",
            "dataset": "bike_rentals",
            "prompt": "Calculate a 3-day rolling average of RentalCount after sorting by Day.",
            "hints": ["Sort by date then use rolling(window=3).mean()."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/bike_rentals.csv")
df["Day"] = pd.to_datetime(df["Day"])
df = df.sort_values("Day")
df["rolling_mean"] = df["RentalCount"].rolling(window=3).mean()
print(df[["Day", "RentalCount", "rolling_mean"]])
""",
        },
        {
            "title": "Seasonality baseline",
            "dataset": "bike_rentals",
            "prompt": "Build a simple baseline predicting rentals using the mean of the previous 3 days.",
            "hints": ["Shift the rolling mean by one day to avoid leakage."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/bike_rentals.csv")
df["Day"] = pd.to_datetime(df["Day"])
df = df.sort_values("Day")
df["rolling_mean"] = df["RentalCount"].rolling(3).mean()
df["baseline_pred"] = df["rolling_mean"].shift(1)
print(df[["Day", "RentalCount", "baseline_pred"]])
""",
        },
        {
            "title": "MAPE evaluation",
            "dataset": "bike_rentals",
            "prompt": "Compute mean absolute percentage error between baseline_pred and actual rentals (drop NaNs).",
            "hints": ["Use abs((y_true - y_pred)/y_true).mean() and drop missing predictions first."],
            "solution": """
import pandas as pd

df = pd.read_csv("data/bike_rentals.csv")
df["Day"] = pd.to_datetime(df["Day"])
df = df.sort_values("Day")
df["rolling_mean"] = df["RentalCount"].rolling(3).mean()
df["baseline_pred"] = df["rolling_mean"].shift(1)
valid = df.dropna(subset=["baseline_pred"])
mape = (valid["RentalCount"].sub(valid["baseline_pred"]).abs() / valid["RentalCount"]).mean() * 100
print(f"MAPE: {mape:.2f}%")
""",
        },
        {
            "title": "Custom evaluation helper",
            "dataset": "bike_rentals",
            "prompt": "Write a function evaluate_predictions(true, pred) returning MAE, MSE, and MAPE.",
            "hints": ["Leverage pandas Series operations for concise math."],
            "solution": """
import pandas as pd

def evaluate_predictions(true, pred):
    diff = true - pred
    mae = diff.abs().mean()
    mse = (diff ** 2).mean()
    mape = (diff.abs() / true).mean() * 100
    return {"mae": mae, "mse": mse, "mape": mape}

# Example using the bike rentals baseline
df = pd.read_csv("data/bike_rentals.csv")
df["Day"] = pd.to_datetime(df["Day"])
df = df.sort_values("Day")
df["rolling_mean"] = df["RentalCount"].rolling(3).mean()
df["baseline_pred"] = df["rolling_mean"].shift(1)
valid = df.dropna(subset=["baseline_pred"])
print(evaluate_predictions(valid["RentalCount"], valid["baseline_pred"]))
""",
        },
        {
            "title": "Confidence intervals",
            "dataset": "movie_ratings",
            "prompt": "Compute a 95% confidence interval for the mean movie rating assuming normality.",
            "hints": ["Use standard error = std/sqrt(n) and 1.96 multiplier."],
            "solution": """
import math
import pandas as pd

df = pd.read_csv("data/movie_ratings.csv")
mean = df["Rating"].mean()
std = df["Rating"].std()
n = len(df)
se = std / math.sqrt(n)
lower = mean - 1.96 * se
upper = mean + 1.96 * se
print((lower, upper))
""",
        },
        {
            "title": "Bootstrap mean",
            "dataset": "movie_ratings",
            "prompt": "Run 200 bootstrap samples of Rating mean and report the 2.5 and 97.5 percentiles.",
            "hints": ["Use np.random.choice with replace=True inside a loop."],
            "solution": """
import numpy as np
import pandas as pd

df = pd.read_csv("data/movie_ratings.csv")
means = []
for _ in range(200):
    sample = df["Rating"].sample(frac=1, replace=True, random_state=None)
    means.append(sample.mean())
low, high = np.percentile(means, [2.5, 97.5])
print(low, high)
""",
        },
    ]


def build_curriculum() -> dict:
    return {
        "Beginner": beginner_exercises(),
        "Intermediate": intermediate_exercises(),
        "Advanced": advanced_exercises(),
    }


CURRICULUM = build_curriculum()


def run_user_code(code: str, df: pd.DataFrame) -> str:
    local_vars: dict[str, object] = {"df": df.copy(), "pd": pd}
    stdout = io.StringIO()
    try:
        with redirect_stdout(stdout):
            exec(code, {}, local_vars)
    except Exception as error:  # noqa: BLE001
        return f"Execution error: {error}"
    return stdout.getvalue() or "Code executed without print output. Use print() to display results."


st.title("Python Data Science Practice Lab")
st.caption(
    "20 beginner, 20 intermediate, and 20 advanced exercises with datasets, hints, and solutions."
)

level = st.sidebar.selectbox("Choose your path", list(CURRICULUM.keys()))
if f"{level}_index" not in st.session_state:
    st.session_state[f"{level}_index"] = 0

exercises = CURRICULUM[level]
current_index = st.session_state[f"{level}_index"]
exercise = exercises[current_index]
df = load_dataset(exercise["dataset"])

st.sidebar.markdown("### Progress")
st.sidebar.progress((current_index + 1) / len(exercises))
st.sidebar.write(f"Exercise {current_index + 1} of {len(exercises)}")

if st.sidebar.button("Restart level"):
    st.session_state[f"{level}_index"] = 0
    st.experimental_rerun()

st.subheader(f"{level} Challenge {current_index + 1}: {exercise['title']}")
st.write(exercise["prompt"])

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### Dataset snapshot")
    st.write(DATASETS[exercise["dataset"]]["description"])
    st.dataframe(df.head())
    st.download_button(
        label="Download dataset (CSV)",
        data=df.to_csv(index=False),
        file_name=f"{exercise['dataset']}.csv",
        mime="text/csv",
    )

with col2:
    st.markdown("#### Try your code")
    default_code = dedent(
        f"""
        # df is already loaded for you
        # Add your solution below
        print(df.head())
        """
    ).strip()
    user_code = st.text_area("Code editor", value=default_code, height=260)
    if st.button("Run code"):
        output = run_user_code(user_code, df)
        st.text_area("Output", value=output, height=180)

st.markdown("#### Need help?")
with st.expander("Show hints"):
    for hint in exercise["hints"]:
        st.markdown(f"- {hint}")
with st.expander("Show solution"):
    st.code(exercise["solution"], language="python")

st.info("Pause here and reflect on what you learned. Ready for the next challenge?")
next_col1, next_col2 = st.columns(2)
with next_col1:
    if st.button("Yes, take me to the next exercise ➡️"):
        st.session_state[f"{level}_index"] = (current_index + 1) % len(exercises)
        st.experimental_rerun()
with next_col2:
    st.write("Use the hints or solution if you want more guidance before moving on.")

st.markdown("---")
st.markdown(
    "💡 Tip: You can copy any solution, run it, and modify parameters to see how the outcome changes."
)
