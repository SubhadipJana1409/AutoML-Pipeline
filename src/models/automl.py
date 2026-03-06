"""
AutoML pipeline: manual 13-model benchmark + FLAML automated search.

FLAML (Fast and Lightweight AutoML, Microsoft Research) is a production-grade
AutoML framework that uses cost-frugal optimisation, searching RF, XGBoost,
LightGBM, ExtraTrees, LR, KNN and more — compatible with sklearn >= 1.3.
"""
from __future__ import annotations
import logging, time
from pathlib import Path
import numpy as np, pandas as pd, joblib

from flaml import AutoML as FLAMLAutoML

from sklearn.linear_model  import LogisticRegression, SGDClassifier
from sklearn.svm           import SVC, LinearSVC
from sklearn.neighbors     import KNeighborsClassifier
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import (RandomForestClassifier, ExtraTreesClassifier,
                                   GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.naive_bayes   import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
    matthews_corrcoef, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix)

logger = logging.getLogger(__name__)

def _has_proba(pipe):
    clf = pipe.named_steps.get("clf", pipe)
    return hasattr(clf, "predict_proba")

def _make_manual_models(seed):
    sc = ("scaler", StandardScaler())
    return {
        "LogisticL1":       Pipeline([sc, ("clf", LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=2000, random_state=seed))]),
        "LogisticL2":       Pipeline([sc, ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=seed))]),
        "SGD":              Pipeline([sc, ("clf", SGDClassifier(loss="modified_huber", random_state=seed, max_iter=1000))]),
        "LDA":              Pipeline([sc, ("clf", LinearDiscriminantAnalysis())]),
        "SVM_linear":       Pipeline([sc, ("clf", LinearSVC(C=1.0, max_iter=2000, random_state=seed))]),
        "SVM_RBF":          Pipeline([sc, ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=seed))]),
        "KNN":              Pipeline([sc, ("clf", KNeighborsClassifier(n_neighbors=7))]),
        "DecisionTree":     Pipeline([("clf", DecisionTreeClassifier(max_depth=6, random_state=seed))]),
        "RandomForest":     Pipeline([("clf", RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=1))]),
        "ExtraTrees":       Pipeline([("clf", ExtraTreesClassifier(n_estimators=150, random_state=seed, n_jobs=1))]),
        "GradientBoosting": Pipeline([("clf", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=seed))]),
        "AdaBoost":         Pipeline([("clf", AdaBoostClassifier(n_estimators=100, random_state=seed))]),
        "NaiveBayes":       Pipeline([sc, ("clf", GaussianNB())]),
    }


class AutoMLPipeline:
    """AutoML pipeline: 13-model manual benchmark + FLAML evolutionary search."""

    def __init__(self, seed=42, cv_folds=5, flaml_budget=60):
        self.seed=seed; self.cv_folds=cv_folds; self.flaml_budget=flaml_budget
        self._manual_models={}; self._flaml=None
        self.benchmark_results_=pd.DataFrame(); self.automl_results_={}
        self.cv_details_={}; self.feature_importance_=None

    def run_benchmark(self, X, y, feature_names):
        skf=StratifiedKFold(n_splits=self.cv_folds,shuffle=True,random_state=self.seed)
        models=_make_manual_models(self.seed); rows=[]; per_fold={}
        for name, pipe in models.items():
            t0=time.time()
            try:
                if _has_proba(pipe):
                    proba=cross_val_predict(pipe,X,y,cv=skf,method="predict_proba",n_jobs=1)[:,1]
                else:
                    dec=cross_val_predict(pipe,X,y,cv=skf,method="decision_function",n_jobs=1)
                    proba=(dec-dec.min())/(dec.max()-dec.min()+1e-10)
                pred=(proba>=0.5).astype(int)
                per_fold[name]=[]
                for tr,te in skf.split(X,y):
                    p2=pipe.fit(X[tr],y[tr])
                    p=pipe.predict_proba(X[te])[:,1] if _has_proba(pipe) else pipe.decision_function(X[te])
                    if not _has_proba(pipe): p=(p-p.min())/(p.max()-p.min()+1e-10)
                    per_fold[name].append(roc_auc_score(y[te],p))
                elapsed=time.time()-t0
                rows.append({"model":name,"auc_roc":round(roc_auc_score(y,proba),4),
                    "auc_pr":round(average_precision_score(y,proba),4),
                    "f1":round(f1_score(y,pred,zero_division=0),4),
                    "accuracy":round(accuracy_score(y,pred),4),
                    "mcc":round(matthews_corrcoef(y,pred),4),"time_s":round(elapsed,2)})
                self._manual_models[name]=pipe
                logger.info("  %-22s AUC=%.3f  F1=%.3f  [%.1fs]",name,rows[-1]["auc_roc"],rows[-1]["f1"],elapsed)
            except Exception as e:
                logger.warning("  %-22s FAILED: %s",name,e)
        self.cv_details_=per_fold
        self.benchmark_results_=pd.DataFrame(rows).sort_values("auc_roc",ascending=False).reset_index(drop=True)
        for name in ["RandomForest","ExtraTrees","GradientBoosting"]:
            if name in self._manual_models:
                self._manual_models[name].fit(X,y)
                clf=self._manual_models[name].named_steps["clf"]
                if hasattr(clf,"feature_importances_"):
                    self.feature_importance_=pd.Series(clf.feature_importances_,index=feature_names).sort_values(ascending=False)
                    logger.info("Feature importance from %s",name); break
        return self.benchmark_results_

    def run_automl(self, X_train, y_train, X_test, y_test):
        """FLAML automated model + hyperparameter search."""
        logger.info("Starting FLAML (budget=%ds)…",self.flaml_budget)
        self._flaml=FLAMLAutoML()
        self._flaml.fit(X_train, y_train, task="classification",
                        metric="roc_auc", time_budget=self.flaml_budget,
                        seed=self.seed, verbose=0, n_jobs=1)
        best_model=self._flaml.best_estimator
        logger.info("FLAML best estimator: %s",best_model)
        proba=self._flaml.predict_proba(X_test)[:,1]
        pred=(proba>=0.5).astype(int)
        fpr,tpr,_=roc_curve(y_test,proba); prec,rec,_=precision_recall_curve(y_test,proba)
        self.automl_results_={"auc_roc":round(roc_auc_score(y_test,proba),4),
            "auc_pr":round(average_precision_score(y_test,proba),4),
            "f1":round(f1_score(y_test,pred,zero_division=0),4),
            "accuracy":round(accuracy_score(y_test,pred),4),
            "mcc":round(matthews_corrcoef(y_test,pred),4),
            "confusion":confusion_matrix(y_test,pred),
            "fpr":fpr,"tpr":tpr,"prec_curve":prec,"rec_curve":rec,"proba":proba,"pred":pred,
            "best_estimator":best_model,
            "pipeline_str":f"FLAML → {best_model}"}
        logger.info("FLAML → AUC=%.3f  F1=%.3f",self.automl_results_["auc_roc"],self.automl_results_["f1"])
        return self.automl_results_

    def evaluate_best_manual(self, X_train, y_train, X_test, y_test):
        name=self.benchmark_results_.iloc[0]["model"]; pipe=self._manual_models[name]
        pipe.fit(X_train,y_train)
        proba=pipe.predict_proba(X_test)[:,1] if _has_proba(pipe) else pipe.decision_function(X_test)
        if not _has_proba(pipe): proba=(proba-proba.min())/(proba.max()-proba.min()+1e-10)
        pred=(proba>=0.5).astype(int)
        fpr,tpr,_=roc_curve(y_test,proba); prec,rec,_=precision_recall_curve(y_test,proba)
        return {"model":name,"auc_roc":round(roc_auc_score(y_test,proba),4),
            "auc_pr":round(average_precision_score(y_test,proba),4),
            "f1":round(f1_score(y_test,pred,zero_division=0),4),
            "accuracy":round(accuracy_score(y_test,pred),4),
            "mcc":round(matthews_corrcoef(y_test,pred),4),
            "confusion":confusion_matrix(y_test,pred),
            "fpr":fpr,"tpr":tpr,"prec_curve":prec,"rec_curve":rec}

    def benchmark_datasets(self, datasets, model_name="RandomForest"):
        skf=StratifiedKFold(n_splits=self.cv_folds,shuffle=True,random_state=self.seed)
        rows=[]
        for ds_name,(X,y) in datasets.items():
            pipe=_make_manual_models(self.seed)[model_name]
            proba=cross_val_predict(pipe,X,y,cv=skf,method="predict_proba",n_jobs=1)[:,1]
            rows.append({"dataset":ds_name,"auc_roc":round(roc_auc_score(y,proba),4),
                "auc_pr":round(average_precision_score(y,proba),4),
                "f1":round(f1_score(y,(proba>=0.5).astype(int),zero_division=0),4)})
            logger.info("  Dataset %-12s AUC=%.3f",ds_name,rows[-1]["auc_roc"])
        return pd.DataFrame(rows)

    def save(self, path):
        Path(path).parent.mkdir(parents=True,exist_ok=True); joblib.dump(self,path)

    @staticmethod
    def load(path): return joblib.load(path)
