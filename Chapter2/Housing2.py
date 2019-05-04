from Chapter2.Housing import load_housing_data
from Chapter2.Housing import CombinedAttributesAdder
from Chapter2.Housing import strat_train_set
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


class LabelBinarizerPipelineFriendly(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return LabelBinarizer().fit(X).transform(X)

# full_pipeline = Pipeline([
#     ('num_pipeline', num_pipeline),
#     ('cat_pipeline', cat_pipeline),
# ])


def full_pipeline(housing):
    housing_num = housing.drop("ocean_proximity", axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', LabelBinarizerPipelineFriendly()),
    ])

    housing_num_tr = num_pipeline.fit_transform(housing)
    housing_cat_tr = cat_pipeline.fit_transform(housing)

    housing_prepared = np.ndarray(shape=(housing_num_tr.shape[0],
                                         housing_num_tr.shape[1]+ housing_cat_tr.shape[1]))
    for i in range(housing_prepared.shape[0]):
        for j in range(housing_num_tr.shape[1]):
            housing_prepared[i][j] = housing_num_tr[i][j]
        for n in range(housing_cat_tr.shape[1]):
            housing_prepared[i][j+n+1] = housing_cat_tr[i][n]
    return housing_prepared

housing = load_housing_data()
housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

housing_prepared = full_pipeline(housing)
# lin_reg = LinearRegression()
# lin_reg.fit(housing_prepared, housing_labels)

# some_data = housing.iloc[0:10000]
# some_labels = housing_labels.iloc[0:10000]
# some_data_prepared = full_pipeline(some_data)
# print("Prediction:\t", lin_reg.predict(some_data_prepared[:5]))
# print("Labels:\t\t", list(some_labels[:5]))

# housing_predictions = lin_reg.predict(housing_prepared)
# lin_mse = mean_squared_error(housing_labels, housing_predictions)
# lin_rmse = np.sqrt(lin_mse)
# print(lin_rmse)
#
# tree_reg = DecisionTreeRegressor()
# tree_reg.fit(housing_prepared, housing_labels)
# housing_predictions = tree_reg.predict(housing_prepared)
# tree_mse = mean_squared_error(housing_labels, housing_predictions)
# tree_rmse = np.sqrt(tree_mse)
# print(tree_rmse)

# scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
#                          scoring="neg_mean_squared_error", cv=10)
# rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())


# display_scores(rmse_scores)
# lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# lin_rmse_scores = np.sqrt(-lin_scores)
# display_scores(lin_rmse_scores)
#
# forest_reg = RandomForestRegressor()
# forest_reg.fit(housing_prepared, housing_labels)
# forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# forest_rmse_scores = np.sqrt(-forest_scores)
# display_scores(forest_rmse_scores)

# joblib.dump(lin_reg, "my_linear_model.pkl")
# joblib.dump(tree_reg, "my_tree_model.pkl")
# joblib.dump(forest_reg, "my_random_forest_model.pkl")

# lin_reg = joblib.load("my_linear_model.pkl")
# tree_reg = joblib.load("my_tree_model.pkl")
# forest_reg = joblib.load("my_random_forest_model.pkl")
# lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# lin_rmse_scores = np.sqrt(-lin_scores)
# display_scores(lin_rmse_scores)
# scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# rmse_scores = np.sqrt(-scores)
# display_scores(rmse_scores)
# forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
# forest_rmse_scores = np.sqrt(-forest_scores)
# display_scores(forest_rmse_scores)

# param_grid = [
#     {'n_estimators': [3, 10, 30], 'max_features': [2,4,6,8]},
#     {'bootstrap': [False], 'n_estimators': [3,10], 'max_features': [2,3,4]},
# ]
#
# forest_reg = RandomForestRegressor()
# grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
# grid_search.fit(housing_prepared, housing_labels)
# joblib.dump(grid_search, "my_grid_search_model.pkl")
grid_search = joblib.load("my_grid_search_model.pkl")
# print(grid_search.best_params_)
# print(grid_search.best_estimator_)
curves = grid_search.cv_results_
# for mean_score, params in zip(curves["mean_test_score"], curves["params"]):
#     print(np.sqrt(-mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_
# print(feature_importances)



# extra_attribs = ["rooms_per_hold", "pop_per_hhold", "bedrooms_per_room"]
# encoder = LabelBinarizer()
# housing_cat = housing["ocean_proximity"]
# encoder.fit(housing_cat)
# cat_one_hot_attribs = list(encoder.classes_)
# housing_num = housing.drop("ocean_proximity", axis=1)
# num_attribs = list(housing_num)
# attributes = num_attribs + extra_attribs + cat_one_hot_attribs
#
# for feature in sorted(zip(feature_importances, attributes), reverse=True):
#     print(feature)

final_model = grid_search.best_estimator_

X_test = strat_train_set.drop("median_house_value", axis=1)
y_test = strat_train_set["median_house_value"].copy()

X_test_prepared = full_pipeline(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)