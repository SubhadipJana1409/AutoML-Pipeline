"""Tests for src/data/simulator.py"""
import numpy as np, pandas as pd, pytest
from src.data.simulator import simulate_dataset, get_all_datasets, OTU_NAMES, N_OTUS

class TestSimulateDataset:
    def test_shape(self):
        X,y = simulate_dataset(n_samples=50,seed=0)
        assert X.shape==(50,N_OTUS) and len(y)==50

    def test_binary_labels(self):
        _,y = simulate_dataset(n_samples=40,seed=0)
        assert set(y.unique()).issubset({0,1})

    def test_balance(self):
        _,y = simulate_dataset(n_samples=100,ibd_fraction=0.5,seed=0)
        assert y.sum()==50

    def test_reproducible(self):
        X1,y1=simulate_dataset(n_samples=30,seed=5)
        X2,y2=simulate_dataset(n_samples=30,seed=5)
        pd.testing.assert_frame_equal(X1,X2)

    def test_no_nan(self):
        X,_ = simulate_dataset(n_samples=50,seed=0)
        assert not X.isnull().any().any()

    def test_label_noise_changes_labels(self):
        _,y0=simulate_dataset(n_samples=100,label_noise=0.0,seed=0)
        _,yn=simulate_dataset(n_samples=100,label_noise=0.3,seed=0)
        assert not (y0==yn).all()

    def test_otu_names_length(self):
        assert len(OTU_NAMES)==N_OTUS==150

    def test_imbalanced(self):
        _,y=simulate_dataset(n_samples=200,ibd_fraction=0.25,seed=0)
        assert abs(y.sum()-50)<=2

class TestGetAllDatasets:
    def test_returns_three_datasets(self):
        ds=get_all_datasets(seed=0)
        assert set(ds.keys())=={"balanced","imbalanced","noisy"}

    def test_each_has_correct_type(self):
        ds=get_all_datasets(seed=0)
        for _,(X,y) in ds.items():
            assert isinstance(X,pd.DataFrame) and isinstance(y,pd.Series)
