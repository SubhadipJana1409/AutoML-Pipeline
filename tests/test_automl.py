"""Tests for AutoMLPipeline and visualization."""
import numpy as np, pandas as pd, pytest
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.data.simulator  import simulate_dataset, OTU_NAMES
from src.models.automl   import AutoMLPipeline

@pytest.fixture(scope="module")
def fitted():
    X,y=simulate_dataset(n_samples=100,seed=0)
    X_tr,X_te,y_tr,y_te=train_test_split(X.values,y.values,test_size=0.25,stratify=y.values,random_state=0)
    a=AutoMLPipeline(seed=0,cv_folds=3,flaml_budget=10)
    a.run_benchmark(X_tr,y_tr,OTU_NAMES)
    a.run_automl(X_tr,y_tr,X_te,y_te)
    return a,X_tr,X_te,y_tr,y_te,X.values,y.values

class TestBenchmark:
    def test_results_nonempty(self,fitted):
        a=fitted[0]; assert len(a.benchmark_results_)>0

    def test_auc_in_range(self,fitted):
        a=fitted[0]
        for _,row in a.benchmark_results_.iterrows():
            assert 0.0<=row["auc_roc"]<=1.0

    def test_feature_importance_set(self,fitted):
        a=fitted[0]; assert a.feature_importance_ is not None

    def test_cv_details_populated(self,fitted):
        a=fitted[0]; assert len(a.cv_details_)>0
        for m,folds in a.cv_details_.items():
            assert len(folds)==3

class TestAutoML:
    def test_automl_results_has_auc(self,fitted):
        a=fitted[0]; assert 0.0<=a.automl_results_["auc_roc"]<=1.0

    def test_automl_confusion_matrix(self,fitted):
        a=fitted[0]; assert a.automl_results_["confusion"].shape==(2,2)

    def test_automl_best_estimator_string(self,fitted):
        a=fitted[0]; assert isinstance(a.automl_results_["best_estimator"],str)

class TestEvaluateBestManual:
    def test_returns_dict(self,fitted):
        a,X_tr,X_te,y_tr,y_te,_,_=fitted
        r=a.evaluate_best_manual(X_tr,y_tr,X_te,y_te)
        assert isinstance(r,dict) and "auc_roc" in r

class TestBenchmarkDatasets:
    def test_three_datasets(self,fitted):
        a,_,_,_,_,X,y=fitted
        from src.data.simulator import get_all_datasets
        ds={k:(df.values,s.values) for k,(df,s) in get_all_datasets(seed=0).items()}
        rob=a.benchmark_datasets(ds,model_name="RandomForest")
        assert len(rob)==3 and "balanced" in rob["dataset"].values

class TestSaveLoad:
    def test_save_load(self,fitted,tmp_path):
        a,_,X_te,_,y_te,_,_=fitted
        p=tmp_path/"pipe.joblib"; a.save(p)
        a2=AutoMLPipeline.load(p)
        assert a2.benchmark_results_.equals(a.benchmark_results_)

class TestPlots:
    def test_fig1(self,fitted,tmp_path):
        from src.visualization.plots import fig1_dataset_overview
        _,_,_,_,_,X,y=fitted
        from src.data.simulator import OTU_NAMES
        fig1_dataset_overview(X,y,OTU_NAMES,Path(tmp_path))
        assert (tmp_path/"fig1_dataset_overview.png").exists()

    def test_fig3(self,fitted,tmp_path):
        from src.visualization.plots import fig3_performance_grid
        a=fitted[0]; fig3_performance_grid(a.benchmark_results_,Path(tmp_path))
        assert (tmp_path/"fig3_performance_grid.png").exists()

    def test_fig6(self,fitted,tmp_path):
        from src.visualization.plots import fig6_cv_boxplot
        a=fitted[0]; fig6_cv_boxplot(a.cv_details_,Path(tmp_path))
        assert (tmp_path/"fig6_cv_boxplot.png").exists()

    def test_fig7(self,fitted,tmp_path):
        from src.visualization.plots import fig7_feature_importance
        a=fitted[0]; fig7_feature_importance(a.feature_importance_,Path(tmp_path))
        assert (tmp_path/"fig7_feature_importance.png").exists()
