import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA


def initial_stratified_with_coverage(X, y, n_init=20, seed=None):
    """
    Select an initial training set that:
    1) Contains at least one sample from each class
    2) Fills the remaining samples using stratified sampling

    PARAMETERS
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target labels
    n_init : int
        Number of initial training samples
    seed : int
        Random seed for reproducibility

    RETURNS
    -------
    np.ndarray
        Indices of selected initial training samples
    """
    rng = np.random.default_rng(seed)

    classes = np.unique(y)
    n_classes = len(classes)

    if n_init < n_classes:
        raise ValueError("n_init must be >= number of classes")

    # --- Step 1: ensure at least 1 sample per class ---
    initial_idx = []

    for c in classes:
        class_indices = np.where(y == c)[0]
        chosen = rng.choice(class_indices)
        initial_idx.append(chosen)

    initial_idx = np.array(initial_idx)

    # --- Step 2: fill remaining with stratified sampling ---
    remaining_needed = n_init - len(initial_idx)

    if remaining_needed > 0:
        remaining_pool = np.setdiff1d(np.arange(len(y)), initial_idx)

        X_remain = X[remaining_pool]
        y_remain = y[remaining_pool]

        sss = StratifiedShuffleSplit(
            n_splits=1,
            train_size=remaining_needed,
            random_state=seed
        )

        extra_idx_rel, _ = next(sss.split(X_remain, y_remain))
        extra_idx = remaining_pool[extra_idx_rel]

        initial_idx = np.concatenate([initial_idx, extra_idx])

    return initial_idx


def prepare_data(cov_type, n_init, n_points=None, seed=None,n_components=4, even_distribution = False):
    """
    Prepare dataset for active learning by creating:
    - Initial labeled training set (with class coverage)
    - Unlabeled pool set
    - Test set (held-out, stratified)

    PARAMETERS
    ----------
    cov_type : dict-like
        Dataset object with keys 'data' and 'target'
    n_init : int
        Number of initial labeled samples
    n_points : int or None
        Optional number of total samples to use (stratified subset)
    seed : int
        Random seed for reproducibility

    RETURNS
    -------
    dict
        Dictionary with keys:
        - 'train': initial labeled set
        - 'pool': unlabeled pool for active learning
        - 'test': held-out test set
    """

    #---PCA---
    X = cov_type['data']
    pca=PCA(n_components=n_components)
    X=(X-X.mean())/np.std(X)
    X=pca.fit_transform(X)
    explained_var=pca.explained_variance_ratio_
    y = cov_type['target']


    # --- Optional: reduce dataset size using stratified sampling ---
    if n_points is not None and n_points < len(X):
        if not even_distribution:
            sss = StratifiedShuffleSplit(
                n_splits=1,
                train_size=n_points,
                random_state=seed
            )
            idx, _ = next(sss.split(X, y))
            X, y = X[idx], y[idx]
        else:
            rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
            X, y = rus.fit_resample(X, y)




    # --- Step 1: split into train+pool and test (stratified) ---
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=seed
    )
    train_pool_idx, test_idx = next(sss.split(X, y))

    X_train_pool = X[train_pool_idx]
    y_train_pool = y[train_pool_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    # --- Step 2: select initial training set with class coverage ---
    init_rel_idx = initial_stratified_with_coverage(
        X_train_pool, y_train_pool, n_init=n_init, seed=seed
    )

    # Remaining samples become the pool
    pool_rel_idx = np.setdiff1d(
        np.arange(len(X_train_pool)),
        init_rel_idx
    )

    # --- Construct final splits ---
    return ({
        "train": {
            "X": X_train_pool[init_rel_idx],
            "y": y_train_pool[init_rel_idx],
        },
        "pool": {
            "X": X_train_pool[pool_rel_idx],
            "y": y_train_pool[pool_rel_idx],
        },
        "test": {
            "X": X_test,
            "y": y_test,
        },
    },explained_var)
