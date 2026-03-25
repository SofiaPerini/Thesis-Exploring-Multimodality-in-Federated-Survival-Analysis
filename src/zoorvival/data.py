import os
import warnings
import h5py
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from dataclasses import dataclass
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from typing import Optional

from src.zoorvival.const import BASE_DATA_DIR, AVAILABLE_DATASETS

__all__ = ["load_tcga_data", "TCGADataBag", "TCGADataSplit"]


def get_available_datasets() -> list[str]:
    """Get the list of available TCGA datasets."""
    return AVAILABLE_DATASETS.copy()


@dataclass
class TCGADataSplit:
    """Data class to hold the data for a single split of TCGA data."""
    df_clinical: pd.DataFrame
    df_cnv: pd.DataFrame
    df_dnam: pd.DataFrame
    df_mirna: pd.DataFrame
    df_mrna: pd.DataFrame
    wsi_embeddings: np.ndarray
    y: np.ndarray


@dataclass
class TCGADataBag:
    """Data class to hold the data for both train and test splits."""
    train: TCGADataSplit
    test: TCGADataSplit
    val: Optional[TCGADataSplit] = None


def get_labels(df: pd.DataFrame, dfs_survival: bool = False, verbose: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    """Get the labels for the survival task given a DataFrame.

    Args:
        df: DataFrame containing the labels.
        dfs_survival: Whether to use Overall Survival (OS) or Disease-Free Survival (DFS).
        verbose: Whether to print verbose output.

    Returns:
        A tuple containing the DataFrame without labels and the labels as a structured array.
    """
    # Check if the DataFrame contains the required columns
    if dfs_survival:
        event_col = "dfs_event"
        time_col = "dfs_time"
    else:
        event_col = "os_event"
        time_col = "os_time"
    if event_col not in df.columns or time_col not in df.columns:
        raise ValueError(f"Missing columns: {event_col}, {time_col}")

    # Filter out patients with missing labels and negative time
    orig_len = len(df)
    df = df.dropna(subset=[event_col, time_col])
    df = df[df[time_col] >= 0]
    if verbose:
        logger.info(
            f"Filtered {orig_len - len(df)}/{orig_len} patients. "
            f"Censoring percentage: {sum(df[event_col] == 0) / len(df) * 100:.2f}%"
        )

    # Create the labels array with scikit-survival format
    dtype = [("event", bool), ("time", float)]
    y = np.array(list(zip(df[event_col].values, df[time_col].values)), dtype=dtype)
    return df.drop(columns=["os_event", "os_time", "dfs_event", "dfs_time"], errors="ignore"), y

# Da cambiare per aggiungere validation 
def split_data(df: pd.DataFrame, y: np.ndarray, include_validation: float = False, test_size: float = 0.2, random_state: int = 42, verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Split the data into train and test sets with stratification."""
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, stratify=y["event"], test_size=test_size, random_state=random_state
    )
    if include_validation:
        X_train,X_val, y_train,y_val = train_test_split(X_train,y_train,stratify=y_train['event'],test_size=test_size/(1-test_size), random_state=random_state)
        if verbose:
            logger.info(f"Train set size: {len(X_train)}, Val set size: {len(X_val)}, Test set size: {len(X_test)}")
        return X_train,X_val,X_test,y_train,y_val,y_test
    if verbose:
        logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def get_preprocess_pipeline(X: pd.DataFrame) -> ColumnTransformer:
    """Create a preprocessing pipeline with imputation, scaling, and encoding."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("varthresh", VarianceThreshold(threshold=0.01)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
        ]
    )

    ct = ColumnTransformer(
        transformers=[("num", numeric_pipe, num_cols), ("cat", categorical_pipe, cat_cols)],
        verbose_feature_names_out=False,
    )
    return ct


def preprocess_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: Optional[pd.DataFrame] = None,
    nan_threshold: float = 0.1,
    min_freq: float = 0.01,
    max_cardinality: int = 50,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | \
     tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit the preprocessor on X_train, transform both splits, return DataFrames."""
    # Remove columns with too many missing values
    datasets = [df_train,df_test] if df_val is None else [df_train,df_test,df_val]

    if nan_threshold > 0:
        cols_to_drop = df_train.columns[df_train.isna().mean() > nan_threshold]
        for dataset in datasets:
            dataset = dataset.drop(columns=cols_to_drop)
        if verbose:
            logger.info(f"Dropped {len(cols_to_drop)} columns with >{nan_threshold:.0%} missing values")

    # Remove constant columns
    const_cols = [c for c in df_train.columns if df_train[c].nunique(dropna=False) <= 1]
    for dataset in datasets:
        dataset = dataset.drop(columns=const_cols)
    if verbose:
        logger.info(f"Dropped {len(const_cols)} constant columns")
    
    # Remove ID columns
    indices = []
    for dataset in datasets:
        ds_index = dataset["patient_id"].values.copy()
        indices.append(ds_index)
        dataset = dataset.drop(columns=["patient_id","study_id"],errors="ignore")

    # df_train_index = df_train["patient_id"].values
    # df_train = df_train.drop(columns=["patient_id", "study_id"], errors="ignore")
    # df_test_index = df_test["patient_id"].values
    # df_test = df_test.drop(columns=["patient_id", "study_id"], errors="ignore")
    if verbose:
        logger.info("Dropped patient_id and study_id columns")

    # Aggregate categorical columns with too many unique values
    cat_cols = df_train.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        if df_train[col].nunique(dropna=False) > max_cardinality:
            freq = df_train[col].value_counts(normalize=True)
            rare_categories = freq[freq < min_freq].index
            for dataset in datasets:
                dataset.loc[dataset[col].isin(rare_categories),col] = "__RARE__"
            # df_train.loc[df_train[col].isin(rare_categories), col] = "__RARE__"
            # df_test.loc[df_test[col].isin(rare_categories), col] = "__RARE__"

    # Create and fit the preprocessor
    preprocessor = get_preprocess_pipeline(df_train)
    preprocessor.fit(df_train)

    # Transform splits
    num_features = df_train.shape[1]
    array_datasets = []
    for dataset in datasets:
        X = preprocessor.transform(dataset).toarray()
        array_datasets.append(X)
    # X_train = preprocessor.transform(df_train).toarray()
    # X_test = preprocessor.transform(df_test).toarray()
    if verbose:
        logger.info(f"Transformed {num_features} features into {array_datasets[0].shape[1]} features")

    # Build DataFrames from the matrices
    feature_names = preprocessor.get_feature_names_out()
    df_train = pd.DataFrame(array_datasets[0], columns=feature_names, index=indices[0])
    df_test = pd.DataFrame(array_datasets[1], columns=feature_names, index=indices[1])
    
    if df_val is None:
        return df_train, df_test
    
    else:
        df_val = pd.DataFrame(array_datasets[2], columns=feature_names, index=indices[2])
        return df_train,df_val,df_test



def load_tcga_clinical_data(
    project: str,
    base_dir: str = None,
    include_validation: bool = False,
    dfs_survival: bool = False,
    test_size: float = 0.2,
    nan_threshold: float = 0.1,
    min_freq: float = 0.01,
    max_cardinality: int = 32,
    random_state: int = 42,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray] | \
tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,np.ndarray, np.ndarray, np.ndarray] :
    """Load TCGA clinical data split into train and test sets with preprocessing.
    
    Args:
        project: The TCGA project name, e.g., "BRCA".
        base_dir: The base directory where the data is stored.
        dfs_survival: Whether to use Overall Survival (OS) or Disease-Free Survival (DFS).
        test_size: The proportion of the dataset to include in the test split.
        nan_threshold: The threshold for missing values in a column to drop it.
        min_freq: The minimum frequency for a category to be kept.
        max_cardinality: The maximum number of unique values for a categorical column.
        random_state: Random seed for reproducibility.
        verbose: Whether to print verbose output.

    Returns:
        A tuple containing the preprocessed train and test DataFrames and their labels.
    """
    if project not in get_available_datasets():
        raise ValueError(f"Project {project} not found. Available projects: {get_available_datasets()}")

    if base_dir is None:
        base_dir = BASE_DATA_DIR
    df_clinical = pd.read_csv(os.path.join(base_dir, project, f"{project}_clinical_preprocessed.csv"))
    df_clinical, y = get_labels(df_clinical, dfs_survival=dfs_survival, verbose=verbose)

    if include_validation:
        df_train, df_val, df_test, y_train, y_val, y_test = split_data(df_clinical, y, test_size=test_size, include_validation=True,
                                                             random_state=random_state, verbose=verbose)
        df_train, df_val, df_test = preprocess_features(df_train, df_test,df_val, nan_threshold=nan_threshold, min_freq=min_freq, 
                                                    max_cardinality=max_cardinality, verbose=verbose)
        return df_train, df_val, df_test, y_train, y_val, y_test
    
    df_train, df_test, y_train, y_test = split_data(df_clinical, y, test_size=test_size, include_validation=False,
                                                     random_state=random_state, verbose=verbose)
    df_train, df_test = preprocess_features(df_train, df_test, nan_threshold=nan_threshold, min_freq=min_freq,
                                             max_cardinality=max_cardinality, verbose=verbose)
    return df_train, df_test, y_train, y_test


def format_tcga_omics_data(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Format TCGA omics data by imputing missing values, and scaling features."""
    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(df_train)
    df_train = pd.DataFrame(imputer.transform(df_train), index=df_train.index, columns=df_train.columns)
    df_test = pd.DataFrame(imputer.transform(df_test), index=df_test.index, columns=df_test.columns)
    # Scale features
    scaler = StandardScaler()
    scaler.fit(df_train)
    df_train = pd.DataFrame(scaler.transform(df_train), index=df_train.index, columns=df_train.columns)
    df_test = pd.DataFrame(scaler.transform(df_test), index=df_test.index, columns=df_test.columns)
    # Convert to float32
    df_train = df_train.astype(np.float32)
    df_test = df_test.astype(np.float32)
    return df_train, df_test

def load_tcga_data(
    project: str,
    base_dir: str = None,
    dfs_survival: bool = False,
    test_size: float = 0.2,
    nan_threshold: float = 0.1,
    min_freq: float = 0.01,
    max_cardinality: int = 32,
    use_gigapath_embeddings: bool = True,
    include_validation: bool = False,
    random_state: int = 42,
    verbose: bool = False,
) -> TCGADataBag:
    """Load TCGA data for a given project, including clinical, omics, and WSI data.
    
    Args:
        project: The TCGA project name, e.g., "BRCA".
        base_dir: The base directory where the data is stored.
        dfs_survival: Whether to use Overall Survival (OS) or Disease-Free Survival (DFS).
        test_size: The proportion of the dataset to include in the test split.
        nan_threshold: The threshold for missing values in a column to drop it.
        min_freq: The minimum frequency for a category to be kept.
        max_cardinality: The maximum number of unique values for a categorical column.
        use_gigapath_embeddings: Whether to use Gigapath or ResNet50 embeddings.
        random_state: Random seed for reproducibility.
        verbose: Whether to print verbose output.
    
    Returns:
        A TCGADataBag object containing the train and test splits.
    """
    if project not in get_available_datasets():
        raise ValueError(f"Project {project} not found. Available projects: {get_available_datasets()}")

    if base_dir is None:
        base_dir = BASE_DATA_DIR

    if verbose:
        logger.info(f"Loading TCGA clinical data for {project}...")

    # Load clinical data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clinical_data = load_tcga_clinical_data(
            project=project, 
            base_dir=base_dir,
            dfs_survival=dfs_survival,
            test_size=test_size,
            include_validation=include_validation,
            nan_threshold=nan_threshold,
            min_freq=min_freq,
            max_cardinality=max_cardinality,
            random_state=random_state,
            verbose=verbose,
        )

        if include_validation:
            df_clinical_train, df_clinical_val, df_clinical_test, y_train, y_val, y_test = clinical_data
        else:
            df_clinical_train,df_clinical_test,y_train,y_test = clinical_data

    # Load omics data
    if verbose:
        logger.info(f"Loading TCGA omics data for {project}...")
    df_miRNA = pd.read_csv(os.path.join(base_dir, project, f"{project}_miRNA_filtered.csv"), index_col=0)
    df_DNAm = pd.read_csv(os.path.join(base_dir, project, f"{project}_DNAm_filtered.csv"), index_col=0)
    df_CNV = pd.read_csv(os.path.join(base_dir, project, f"{project}_CNV_filtered.csv"), index_col=0)
    df_mRNA = pd.read_csv(os.path.join(base_dir, project, f"{project}_mRNA_filtered.csv"), index_col=0)
    
    # Check if the omics data has the same patients as the clinical data
    patient_ids_with_all_data = set(df_clinical_train.index).union(set(df_clinical_test.index))
    if include_validation:
        patient_ids_with_all_data = patient_ids_with_all_data.union(set(df_clinical_val.index))

    for i, _df in enumerate([df_miRNA, df_CNV, df_mRNA, df_DNAm]):
        patient_ids_with_all_data.intersection_update(_df.index)
        if len(patient_ids_with_all_data) == 0 and verbose:
            logger.warning(f"No patients found in {project} for the omics data #{i}")

    # Check existence of the embedding files
    model = "gigapath" if use_gigapath_embeddings else "resnet"
    embeddings_path = os.path.join(base_dir, project, f"{project}_{model}_compressed_embeddings.h5")
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embedding file {embeddings_path} not found. Please run the embedding generation script.")
    with h5py.File(embeddings_path, "r") as f:
        embedding_patient_ids = list(f.keys())
        patient_ids_with_all_data.intersection_update(embedding_patient_ids)

    train_mask = [i in patient_ids_with_all_data for i in df_clinical_train.index]
    test_mask = [i in patient_ids_with_all_data for i in df_clinical_test.index]


    train_ids_filtered = df_clinical_train.index[train_mask]
    test_ids_filtered = df_clinical_test.index[test_mask]

    y_train = y_train[train_mask]
    y_test = y_test[test_mask]

    df_clinical_train = df_clinical_train.loc[train_ids_filtered]
    df_clinical_test = df_clinical_test.loc[test_ids_filtered]
    df_miRNA_train = df_miRNA.loc[train_ids_filtered]
    df_miRNA_test = df_miRNA.loc[test_ids_filtered]
    df_DNAm_train = df_DNAm.loc[train_ids_filtered]
    df_DNAm_test = df_DNAm.loc[test_ids_filtered]
    df_CNV_train = df_CNV.loc[train_ids_filtered]
    df_CNV_test = df_CNV.loc[test_ids_filtered]
    df_mRNA_train = df_mRNA.loc[train_ids_filtered]
    df_mRNA_test = df_mRNA.loc[test_ids_filtered]

    if include_validation:
        val_mask = [i in patient_ids_with_all_data for i in df_clinical_val.index]
        val_ids_filtered = df_clinical_val.index[val_mask]
        y_val = y_val[val_mask]
        df_clinical_val = df_clinical_val.loc[val_ids_filtered]
        df_miRNA_val = df_miRNA.loc[val_ids_filtered]
        df_DNAm_val = df_DNAm.loc[val_ids_filtered]
        df_CNV_val = df_CNV.loc[val_ids_filtered]
        df_mRNA_val = df_mRNA.loc[val_ids_filtered]
        df_miRNA_val,_ = format_tcga_omics_data(df_miRNA_val,df_miRNA_train)
        df_DNAm_val,_ = format_tcga_omics_data(df_DNAm_val,df_DNAm_train)
        df_CNV_val,_ = format_tcga_omics_data(df_CNV_val,df_CNV_train)
        df_mRNA_val,_ = format_tcga_omics_data(df_mRNA_val,df_mRNA_train)

    # Format OMICS data
    df_miRNA_train, df_miRNA_test = format_tcga_omics_data(df_miRNA_train, df_miRNA_test)
    df_DNAm_train, df_DNAm_test = format_tcga_omics_data(df_DNAm_train, df_DNAm_test)
    df_CNV_train, df_CNV_test = format_tcga_omics_data(df_CNV_train, df_CNV_test)
    df_mRNA_train, df_mRNA_test = format_tcga_omics_data(df_mRNA_train, df_mRNA_test)

    # Load WSI embeddings
    if verbose:
        logger.info(f"Loading WSI embeddings for {project}...")
    with h5py.File(embeddings_path, "r") as f:
        train_embeddings = np.array([f[patient_id]["embeddings"][()] for patient_id in train_ids_filtered])
        test_embeddings = np.array([f[patient_id]["embeddings"][()] for patient_id in test_ids_filtered])
        if include_validation:
            val_embeddings = np.array([f[patient_id]["embeddings"][()] for patient_id in val_ids_filtered])


    # Load WSI embeddings
    if verbose:
        logger.info(f"Loading WSI magnification info for {project}...")
    with h5py.File(embeddings_path, "r") as f:
        print(f[train_ids_filtered[0]].keys())
        #train_mag = np.array([f[patient_id]["tiles_info"][()] for patient_id in train_ids_filtered])
        #test_mag = np.array([f[patient_id]["tiles_info"][()] for patient_id in test_ids_filtered])
        print(f[train_ids_filtered[0]]["tiles_info"][()])
        #if include_validation:
            #val_mag = np.array([f[patient_id]["tiles_info"][()] for patient_id in val_ids_filtered])



    if verbose:
        logger.info(f"Loaded {project} train embeddings with shape {train_embeddings.shape}")
        logger.info(f"Loaded {project} test embeddings with shape {test_embeddings.shape}")
        if include_validation:
            logger.info(f"Loaded {project} val embeddings with shape {val_embeddings.shape}")


    # Build the dataclasses
    train_data = TCGADataSplit(
        df_clinical=df_clinical_train,
        df_cnv=df_CNV_train,
        df_dnam=df_DNAm_train,
        df_mirna=df_miRNA_train,
        df_mrna=df_mRNA_train,
        wsi_embeddings=train_embeddings,
        #wsi_magnification=train_mag,
        y=y_train,
    )
    test_data = TCGADataSplit(
        df_clinical=df_clinical_test,
        df_cnv=df_CNV_test,
        df_dnam=df_DNAm_test,
        df_mirna=df_miRNA_test,
        df_mrna=df_mRNA_test,
        wsi_embeddings=test_embeddings,
        #wsi_magnification=test_mag,
        y=y_test,
    )

    if include_validation:
        val_data = TCGADataSplit(
        df_clinical=df_clinical_val,
        df_cnv=df_CNV_val,
        df_dnam=df_DNAm_val,
        df_mirna=df_miRNA_val,
        df_mrna=df_mRNA_val,
        wsi_embeddings=val_embeddings,
        #wsi_magnification=val_mag,
        y=y_val,
        )
    if include_validation: 
        data_bag= TCGADataBag(train = train_data, test=test_data, val = val_data)
    else: 
        data_bag = TCGADataBag(train=train_data, test=test_data)

    if verbose:
        logger.info(f"Loaded TCGA data for {project} with {len(train_ids_filtered)} train and {len(test_ids_filtered)} test patients")
        if include_validation:
            logger.info(f"Loaded TCGA data for {project} with {len(train_ids_filtered)} train, {len(val_ids_filtered)} validation patients \
                         and {len(test_ids_filtered)} test patients")

    return data_bag
