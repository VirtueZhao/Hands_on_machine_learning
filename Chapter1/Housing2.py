from Chapter1.Housing import load_housing_data
from Chapter1.Housing import CombinedAttributesAdder
from Chapter1.Housing import strat_train_set
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

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
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# some_data = housing.iloc[0:10000]
# some_labels = housing_labels.iloc[0:10000]
# some_data_prepared = full_pipeline(some_data)
# print("Prediction:\t", lin_reg.predict(some_data_prepared[:5]))
# print("Labels:\t\t", list(some_labels[:5]))

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)


