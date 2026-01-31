from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler , StandardScaler , OrdinalEncoder ,OneHotEncoder 
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn import set_config


class FeaturePipe:
    def __init__(self):
        self.cat_features = [
            "Gender", "Subscription Type", "Contract Length"
        ]
        self.num_features = [
            "Age", "Tenure", "Usage Frequency", "Total Spend"
        ]

        self.cat_pipe = Pipeline([
            ("ohe", OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore"
            ))
        ])

        self.num_pipe = Pipeline([
            ("scaler", RobustScaler())
        ])

        self.ct = ColumnTransformer(
            transformers=[
                ("cat", self.cat_pipe, self.cat_features),
                ("num", self.num_pipe, self.num_features)
            ],
            remainder="drop",
            verbose_feature_names_out=False
        )

    def fit(self, X):
        self.ct.fit(X)
        return self

    def transform(self, X):
        arr = self.ct.transform(X)
        return pd.DataFrame(
            arr,
            columns=self.ct.get_feature_names_out(),
            index=X.index
        )

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
