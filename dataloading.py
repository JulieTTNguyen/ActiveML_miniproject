from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.under_sampling import RandomUnderSampler

def prepare_data(cov_type, n_init, n_points ,seed=None):
    """Extract classes and features and split data in training, pool, and test
    sets.

    PARAMETERS
    ----------
    cov_type
        The cov_type data object.
    n_init : int
        Number of initial data points in the training set.
    seed : int
        Seed for reproducibility.
    """
    undersampler=RandomUnderSampler()
    X,y=undersampler.fit_resample(cov_type['data'],cov_type['target'])
    # X = cov_type['data'][:n_points]
    # y = cov_type['target'][:n_points]

    n = len(X)
    assert n_init <= n

    # Split in train, pool, and test set
    # Use stratified split to make sure we sample all classes equally
    sss = StratifiedShuffleSplit(n_splits=1, train_size= n_init / n, random_state=seed)
    train, pool = next(sss.split(X, y))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    pool_, test = next(sss.split(X[pool], y[pool]))

    return dict(
        train = dict(
            X=X[train],
            y=y[train]
        ),
        pool = dict(
            X=X[pool[pool_]],
            y=y[pool[pool_]]
        ),
        test = dict(
            X=X[pool[test]],
            y=y[pool[test]]
        )
    )

print('yes')