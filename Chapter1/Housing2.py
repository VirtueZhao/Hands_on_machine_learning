from Chapter1.Housing import load_housing_data
from Chapter1.Housing import CombinedAttributesAdder
from Chapter1.Housing import strat_train_set
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion


housing = load_housing_data()
housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
housing = strat_train_set.drop("median_house_value", axis=1)

housing_num = housing.drop("ocean_proximity", axis=1)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]


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

# full_pipeline = Pipeline([
#     ('num_pipeline', num_pipeline),
#     ('cat_pipeline', cat_pipeline),
# ])

housing_num_tr = num_pipeline.fit_transform(housing)
housing_cat_tr = cat_pipeline.fit_transform(housing)
housing_prepared = np.ndarray(shape=(16512,16))

for i in range(housing_prepared.shape[0]):
    for j in range(housing_num_tr.shape[1]):
        housing_prepared[i][j] = housing_num_tr[i][j]
    for n in range(housing_cat_tr.shape[1]):
        housing_prepared[i][j+n+1] = housing_cat_tr[i][n]

print(housing_prepared.shape)

