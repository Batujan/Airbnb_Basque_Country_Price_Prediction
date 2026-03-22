# Airbnb Price Prediction in Bilbao / Basque Region

This project explores whether Airbnb listing metadata can be used to predict nightly prices with a reasonably strong machine learning pipeline. The core goal is practical rather than purely academic: understand which listing characteristics matter most for price, then test whether those signals are strong enough to support a useful predictive model.

The repo is organized around two notebooks:

- [notebooks/01_data_exploration.ipynb](/Users/batuhanferhatoglu/Library/CloudStorage/GoogleDrive-batuhan.ferhatoglu@gmail.com/My%20Drive/Repositories/Personal%20Repositories/Airbnb_Bilbao/notebooks/01_data_exploration.ipynb) explores the raw Airbnb listing data, cleans key fields, studies price behavior, and identifies which variables are most promising.
- [notebooks/02_model_test.ipynb](/Users/batuhanferhatoglu/Library/CloudStorage/GoogleDrive-batuhan.ferhatoglu@gmail.com/My%20Drive/Repositories/Personal%20Repositories/Airbnb_Bilbao/notebooks/02_model_test.ipynb) turns those ideas into a full modeling pipeline and compares several regressors.

Although the repository name emphasizes Bilbao, the saved notebooks and the current CSV indicate a broader Basque-region dataset that includes listings from Vizcaya, Guipuzcoa, and Alava. In practice, this is better read as a Basque Country Airbnb pricing project with Bilbao as one of the main geographic anchors.

## Project Purpose

The project asks a straightforward question:

Can we estimate Airbnb listing prices from structured listing attributes such as capacity, location, host history, review signals, property type, and amenities?

To answer that, the notebooks:

- clean and simplify a large Inside Airbnb style listings table,
- inspect price skew and feature relationships,
- engineer more model-friendly predictors,
- compare linear and tree-based models,
- and evaluate the best model on a held-out test set.

## Explorations and Engineered Features

The exploration notebook focuses on building a sensible feature set before modeling.

Main exploration themes:

- **Target behavior:** `price` is strongly right-skewed, so the analysis shifts to `log_price` for a more stable learning target.
- **Numeric drivers:** accommodates, bedrooms, beds, and bathrooms show the clearest positive relationship with price.
- **Geography:** instead of relying only on raw latitude and longitude, the notebook engineers distance features to Bilbao, Donostia, Vitoria, and the coast.
- **Categoricals:** high-cardinality categories such as `property_type` and `neighbourhood_cleansed` are reduced to top groups plus `Other`.
- **Amenities:** the free-text amenities list is parsed into binary indicators for frequent amenities, along with broader luxury-style signals such as views, pool, hot tub, fireplace, gym, and EV charger.
- **Host signals:** host tenure is converted into `host_experience_days`.
- **Missing values:** review-related missingness is treated as meaningful absence, while sparse numeric gaps are handled with imputation.

The modeling notebook packages those ideas into a reusable scikit-learn workflow with:

- custom transformers for dropping columns, clipping outliers, applying `log1p`, parsing amenities, computing host experience, and generating distance features,
- model-specific preprocessing for linear vs. tree-based models,
- and hyperparameter tuning for Ridge Regression, Random Forest, and Histogram Gradient Boosting.

## Model Results

The saved notebook run compares three tuned models using 5-fold cross-validated RMSE on the log-transformed target:

- Ridge Regression: `0.423`
- Random Forest: `0.363`
- Histogram Gradient Boosting: `0.334`

Histogram Gradient Boosting is the clear winner among the tested models and is then evaluated on the held-out test set.

Saved test metrics:

- RMSE (log scale): `0.3256`
- RMSE (€): `95.45`
- MAE (€): `44.13`
- Median Absolute Error (€): `20.62`
- R²: `0.5945`

## Honest Interpretation of the Test Results

The results are encouraging, but they should be read with some care.

What looks strong:

- The best model clearly outperforms the Ridge baseline, which suggests the relationship between features and price is meaningfully non-linear.
- Cross-validation and test RMSE on the log scale are close (`0.334` vs `0.326`), which is a good sign for generalization.
- A median absolute error of about `€21` suggests the model is often reasonably close for more typical listings.

What still limits the model:

- An RMSE of about `€95` is still large in practical terms, especially for hosts who need fine-grained pricing guidance.
- The residual discussion in the notebook shows the model is noticeably less reliable on more extreme or unusual listings.
- An `R²` of `0.59` is respectable, but it also means a large share of price variation is still unexplained.

Two important caveats:

- The modeling notebook clips `log_price` at the 1st and 99th percentiles before training and evaluation. That is a defensible choice for robustness, but it means the reported metrics are slightly less punishing on extreme prices than a fully raw evaluation would be.
- The saved notebook outputs appear to come from an earlier dataset snapshot than the current [data/listings.csv](/Users/batuhanferhatoglu/Library/CloudStorage/GoogleDrive-batuhan.ferhatoglu@gmail.com/My%20Drive/Repositories/Personal%20Repositories/Airbnb_Bilbao/data/listings.csv). If you rerun the notebooks on the current file, exact counts and metrics may change.

Overall, the project demonstrates that listing structure, geography, host history, and amenities contain real predictive signal. At the same time, the current model should be understood as a solid pricing benchmark rather than a production-ready pricing engine.

## Repository Layout

- [data/listings.csv](/Users/batuhanferhatoglu/Library/CloudStorage/GoogleDrive-batuhan.ferhatoglu@gmail.com/My%20Drive/Repositories/Personal%20Repositories/Airbnb_Bilbao/data/listings.csv): Airbnb listings dataset
- [notebooks/01_data_exploration.ipynb](/Users/batuhanferhatoglu/Library/CloudStorage/GoogleDrive-batuhan.ferhatoglu@gmail.com/My%20Drive/Repositories/Personal%20Repositories/Airbnb_Bilbao/notebooks/01_data_exploration.ipynb): cleaning, exploratory analysis, and feature selection logic
- [notebooks/02_model_test.ipynb](/Users/batuhanferhatoglu/Library/CloudStorage/GoogleDrive-batuhan.ferhatoglu@gmail.com/My%20Drive/Repositories/Personal%20Repositories/Airbnb_Bilbao/notebooks/02_model_test.ipynb): pipeline-based model training, tuning, and evaluation

## Running the Notebooks

The repository currently has an empty [requirements.txt](/Users/batuhanferhatoglu/Library/CloudStorage/GoogleDrive-batuhan.ferhatoglu@gmail.com/My%20Drive/Repositories/Personal%20Repositories/Airbnb_Bilbao/requirements.txt), so the environment is defined implicitly by the notebook imports. At minimum, you will need:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `jupyter`

Then open the notebooks in order:

1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_model_test.ipynb`

If you rerun them on the current dataset, treat the newly generated outputs as the authoritative ones.
