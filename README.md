# Airbnb Basque Country Price Prediction

The goal of this project is to analyze and predict Airbnb listing prices in the Basque Country, where I am currently based. With its coastal and mountainous areas, the region attracts both nature- and culture-oriented tourism, which makes Airbnb price analysis a worthwhile task. The dataset was obtained from [Inside Airbnb](https://insideairbnb.com/get-the-data/)

Because the task involves predicting labeled outcomes, it is a supervised machine learning problem. Since the target is a continuous numerical value, it is specifically a regression problem. In this project, I used three different models and selected the best one based on cross-validation performance on the training set.

As a linear baseline, I used Ridge Regression because the dataset contains many features that are highly correlated with one another. I also used two tree-based ensemble models: Random Forest and Gradient Boosting. These models were better able to capture nonlinear relationships between the features and price than Ridge Regression, suggesting that much of the structure in the data is nonlinear.

Among the three, Gradient Boosting performed best. Its advantage comes from building trees sequentially, with each new tree learning to correct the errors of the previous ones. This allows it to reduce bias effectively while still modeling complex nonlinear patterns.

## Dataset

The dataset is relatively small, containing 6,339 rows and 79 columns. It includes Airbnb listings across the Basque Country, with a higher concentration around Bilbao, Donostia-San Sebastián, and Vitoria-Gasteiz.

The features can be grouped into the following categories:

- irrelevant identifiers such as IDs and URLs
- text features such as description and host_about
- features with clear information leakage, such as estimated_revenue and occupancy
- property-related features such as accommodates and room_type
- host and host-reputation features such as host_since and review-related variables
- location features such as neighbourhood

Irrelevant identifier columns were removed at the beginning of the project. Text features were also excluded, as text analysis is being reserved for a separate project.

## Project Structure

The schema of the project structure:

Airbnb_Basque_Country_Price_Prediction/
├── data/
│   └── listings.csv                 # Raw Airbnb dataset
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA and preprocessing decisions
│   └── 02_model_test.ipynb          # Pipelines, model training and evaluation
├── .gitignore
├── README.md
└── requirements.txt

Recommended reading order: first 01_data_exploration.ipynb, then 02_model_test.ipynb

## Exploratory Data Analysis

Structure:

1. **Import and Load Data**
2. **Initial Data Cleaning and Column Dropping**
3. **Train/Test Split**
4. **Exploratory Data Analysis**
   - **4.a. Analysis of Numerical Features**
   - **4.b. Analysis of Categorical Features**
   - **4.c. Host Tenure Analysis**
   - **4.d. Analysis of Amenities**
   - **4.e. Missing Values**

EDA begins with importing libraries, loading the data, and performing initial cleaning such as converting the strings that are essentially numbers to numeric values, including the removal of irrelevant columns and text features. These steps are followed immediately by a train-test split to prevent data leakage, so the entire EDA is conducted only on the training set.

I used a stratified train-test split based on price bins created from 5 quantiles, with an 80–20 train-test ratio. This helps preserve a similar distribution of lower- and higher-priced listings across both sets. 

In the fourth section, I analyzed the training set after the split and used the results to guide feature selection and preprocessing decisions.

First, I examined the **numerical features**, focusing on their distributions, skewness, and relationship with **log_price**. For highly skewed variables, I tested log and clipping transformations and kept the versions that appeared more useful based on correlation checks. I also analyzed the **geographical variables** and created distance-based features such as **distance_to_center** (this should be interpreted as **distance_to_bilbao**, since the feature was originally named when I thought the dataset only contained listings from Bilbao), **distance_to_donostia**, **distance_to_vitoria**, and **distance_to_coast**. These engineered features were intended to capture location effects more meaningfully than raw coordinates, especially for linear models. The reference locations were chosen based on the main concentration areas observed in the geographical plots. 

Next, I analyzed the **categorical features** by checking their cardinality, frequency distribution, and relationship with **log_price**. I dropped some less useful or redundant columns, grouped rare categories into **"Other"** for high-cardinality variables such as **property_type** and **neighbourhood_cleansed**, and kept the cleaner location variable that best balanced detail and consistency.

I then carried out a **host tenure analysis** by converting date columns into datetime format and creating **host_experience_days** and **host_experience_years**. This was done to test whether more experienced hosts tend to charge different prices.

After that, I analyzed **amenities** by parsing the text-based amenities lists into structured form. I identified common amenities, compared their price patterns, examined the amenities of the most expensive listings, and created simplified luxury-related indicators such as **has_luxury_amenity** and **luxury_amenity_count**.

Finally, I reviewed the **missing values** in the dataset. Based on their meaning and proportion, I decided that review-related variables would be imputed with **0** to reflect the absence of reviews, while a few numeric housing variables with very small missingness would be imputed with the median.

### Features Selected for Modeling

Based on the EDA, the following groups of features were carried forward into the modeling stage:

* List of numerical columns to keep:
    - log_price
    - host_listings_count
    - host_total_listings_count
    - log_accommodates
    - bathrooms_clipped
    - bedrooms_clipped
    - beds_clipped
    - log_minimum_nights_clipped
    - log_maximum_nights_clipped
    - availability_30
    - availability_90
    - availability_365
    - log_number_of_reviews
    - log_reviews_per_month
    - review_scores_rating
    - review_scores_cleanliness
    - review_scores_location
    - calculated_host_listings_count_entire_homes
    - calculated_host_listings_count_private_rooms
    - calculated_host_listings_count_shared_rooms
    - host_acceptance_rate_clipped
    - distance_to_coast
    - distance_to_vitoria
    - distance_to_donostia
    - distance_to_center
    - longitude
    - latitude

* List of categorical columns to keep
    - host_is_superhost
    - host_has_profile_pic
    - neighbourhood_cleansed_clean
    - property_type_clean
    - room_type
    - instant_bookable
    - host_experience_days
    - 99 most frequent amenities encoded
    - has_luxury_amenity
    - luxury_amenity_count

135 features in total has been chosen for the models.

## Creating the Pipelines, Training the Models and the Final Evaluation

This notebook implements the full **model development and evaluation pipeline** for the Airbnb Basque Country dataset.

It starts by loading the cleaned dataset, preparing the target variable, and applying the same **stratified train-test split** used in the EDA workflow. The target variable is transformed into `log_price`, and clipping is applied based only on the training set in order to reduce the effect of extreme values while avoiding leakage into the test set.

The notebook then defines a set of **helper functions** and **custom transformers** to make preprocessing reproducible inside scikit-learn pipelines. These transformers handle tasks such as dropping unnecessary columns, cleaning price and percentage fields, clipping and log-transforming skewed variables, encoding amenities, creating host-experience variables, engineering distance-based location features, mapping binary variables, and reducing high-cardinality categorical features by grouping rare categories into `"Other"`.

After that, I selected the input features and built separate preprocessing pipelines for different model families. The workflow distinguishes between the needs of **Ridge Regression** and the tree-based models, applying scaling where necessary and allowing each model to receive the transformations most appropriate to it.

I then trained and tuned three candidate models: **Ridge Regression**, **Random Forest**, and **HistGradientBoostingRegressor**. Ridge was tuned with **GridSearchCV**, while Random Forest and HistGradientBoosting were tuned with **RandomizedSearchCV**. Their performance was compared using **cross-validated RMSE on the log-transformed target**.

Among the tested models, **HistGradientBoostingRegressor** achieved the best cross-validation result, so it was selected as the final model. I then evaluated it on the held-out test set using both the **log scale** and the **original euro scale**, reporting RMSE, MAE, Median Absolute Error, and R².

Finally, I performed a set of **diagnostic checks**, including actual-versus-predicted comparisons, worst and best prediction examples, a residual plot, and the distribution of absolute errors. These diagnostics showed that the model captures the general pricing structure reasonably well, but has more difficulty with extreme or atypical listings.

### Final Results and Evaluations

Random sample of predictions:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>actual_price_eur</th>
      <th>predicted_price_eur</th>
      <th>absolute_error_eur</th>
      <th>percentage_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>783</th>
      <td>75.0</td>
      <td>127.886757</td>
      <td>52.886757</td>
      <td>70.515676</td>
    </tr>
    <tr>
      <th>898</th>
      <td>93.0</td>
      <td>82.551822</td>
      <td>10.448178</td>
      <td>11.234600</td>
    </tr>
    <tr>
      <th>413</th>
      <td>156.0</td>
      <td>232.491751</td>
      <td>76.491751</td>
      <td>49.033174</td>
    </tr>
    <tr>
      <th>467</th>
      <td>32.0</td>
      <td>40.569881</td>
      <td>8.569881</td>
      <td>26.780878</td>
    </tr>
    <tr>
      <th>745</th>
      <td>71.0</td>
      <td>64.859545</td>
      <td>6.140455</td>
      <td>8.648528</td>
    </tr>
    <tr>
      <th>109</th>
      <td>105.0</td>
      <td>107.067015</td>
      <td>2.067015</td>
      <td>1.968586</td>
    </tr>
    <tr>
      <th>522</th>
      <td>89.0</td>
      <td>137.618602</td>
      <td>48.618602</td>
      <td>54.627643</td>
    </tr>
    <tr>
      <th>56</th>
      <td>138.0</td>
      <td>128.865553</td>
      <td>9.134447</td>
      <td>6.619164</td>
    </tr>
    <tr>
      <th>1111</th>
      <td>193.0</td>
      <td>220.362550</td>
      <td>27.362550</td>
      <td>14.177487</td>
    </tr>
    <tr>
      <th>816</th>
      <td>203.0</td>
      <td>212.601840</td>
      <td>9.601840</td>
      <td>4.729971</td>
    </tr>
  </tbody>
</table>
</div>



    
