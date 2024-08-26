"""IHTBaggingClassifier."""

# Authors: Xiaoguang Wang <fridaymonday@hotmail.com>
# License: MIT


from sklearn.ensemble import BaggingClassifier
from imblearn.under_sampling import InstanceHardnessThreshold
from sklearn.base import clone
from sklearn.utils import resample
import numpy as np


class IHTBaggingClassifier(BaggingClassifier):
    def __init__(self, estimator=None, n_estimators=50, max_samples=1.0,
                 max_features=1.0, bootstrap=True, bootstrap_features=False,
                 random_state=None, n_jobs=None, verbose=0, **kwargs):
        super().__init__(estimator=estimator, n_estimators=n_estimators,
                         max_samples=max_samples, max_features=max_features,
                         bootstrap=bootstrap, bootstrap_features=bootstrap_features,
                         random_state=random_state, n_jobs=n_jobs, verbose=verbose,
                         **kwargs)

    def _fit_base_estimator(self, estimator, X, y, sample_weight=None):
        """
        Fit a single base estimator after applying InstanceHardnessThreshold.
        """
        # Bootstrap sample
        X_resampled, y_resampled = resample(X, y, replace=self.bootstrap,
                                            random_state=self.random_state)

        # Apply Instance Hardness Threshold with the default estimator
        iht = InstanceHardnessThreshold(random_state=self.random_state)
        X_resampled, y_resampled = iht.fit_resample(X_resampled, y_resampled)

        # Fit the estimator on the resampled data
        estimator.fit(X_resampled, y_resampled, sample_weight=sample_weight)
        return estimator

    def fit(self, X, y, sample_weight=None):
        """
        Fit the bagging ensemble.
        """
        # Initialize estimators_ and features_ lists
        self.estimators_ = []
        self.estimators_features_ = []

        # Random seed for reproducibility
        random_state = np.random.RandomState(self.random_state)

        # Determine the number of features to sample
        max_features = (int(self.max_features * X.shape[1])
                        if isinstance(self.max_features, float)
                        else self.max_features)

        for i in range(self.n_estimators):
            # Clone the base estimator
            estimator = clone(self.estimator)

            # Randomly select features if max_features is less than total features
            features = random_state.choice(range(X.shape[1]), max_features, replace=self.bootstrap_features)
            self.estimators_features_.append(features)

            # Train the estimator with the selected features and resampling
            estimator = self._fit_base_estimator(estimator, X[:, features], y, sample_weight)

            # Append the trained estimator to the ensemble
            self.estimators_.append(estimator)

        # Set attributes required for prediction
        self.n_classes_ = len(np.unique(y))
        self.classes_ = np.unique(y)

        return self

    def predict(self, X):
        """
        Predict class labels for the given samples.
        """
        # Aggregate predictions from all classifiers in the ensemble
        predictions = np.asarray([estimator.predict(X[:, features])
                                  for estimator, features in zip(self.estimators_, self.estimators_features_)]).T
        # Majority vote
        maj_vote = np.apply_along_axis(lambda x: np.bincount(x, minlength=self.n_classes_).argmax(), axis=1,
                                       arr=predictions)
        return self.classes_[maj_vote]

    def predict_proba(self, X):
        """
        Predict class probabilities for the given samples.
        """
        # Aggregate predicted probabilities from all classifiers in the ensemble
        proba = np.mean([estimator.predict_proba(X[:, features])
                         for estimator, features in zip(self.estimators_, self.estimators_features_)], axis=0)
        return proba
