from Chapter1.Housing import load_housing_data
from sklearn.pipeline import FeatureUnion

housing = load_housing_data()
housing_num = housing.drop("ocean_proximity", axis=1)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]