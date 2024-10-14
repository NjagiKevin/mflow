from inspect import signature
import os
from venv import logger
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
from mlflow.models import infer_signature

import logging
logging.basicConfig(level=logging.WARN)
logger=logging.getLogger(__name__)

## Function to Evaluate Model Performance
def eval_metrics(actual,pred):
    rmse=np.sqrt(mean_squared_error(actual,pred))
    mae=mean_absolute_error(actual,pred)
    r2=r2_score(actual, pred)
    return rmse,mae,r2


# Filtering out warnings for a cleaner output
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40) # random seed for reproducibility of the model's behavior

    # Read the wine quality csv from the url
    csv_url=(
             "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"  
            )
    # If the download fails, the error is caught, and a message is logged.
    try:
        data=pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets (0.75,0.25) split
    train, test =train_test_split(data)

    # The predicted column is "quality" which is a scalar form [3,9]
    train_x=train.drop(["quality"], axis=1)
    test_x=test.drop(["quality"], axis=1)
    train_y=train[["quality"]]
    test_y=test[["quality"]]

    # The script accepts two command-line arguments: alpha and l1_ratio for the ElasticNet model. 
    # If no arguments are passed, it defaults to 0.5 for both
    alpha=float(sys.argv[1]) if len (sys.argv) >1 else 0.5
    l1_ratio=float(sys.argv[2] if len (sys.argv) >2 else 0.5)

    with mlflow.start_run():
        # ElasticNet is initialized with the provided alpha and l1_ratio. 
        # This is a combination of Ridge and Lasso regression
        lr=ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        # alpha controls the strength of regularization.
        # l1_ratio controls the balance between L1 (Lasso) and L2 (Ridge) penalties.
        lr.fit(train_x, train_y) # model is trained on the training data

        # The model predicts wine quality on the test data (test_x)
        # and the predicted values are stored in predicted_qualities
        predicted_qualities=lr.predict(test_x)

        (rmse,mae,r2)=eval_metrics(test_y, predicted_qualities)

        # The model's configuration (alpha, l1_ratio) and metrics (RMSE, MAE, R²) are printed.
        print("Elasticnet model  (alpha=%f, l1_ratio={:f}):".format (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print (" MAE: %s" % mae)
        print (" R2: %s" % r2)
        
        # MLflow logs the model's hyperparameters and metrics so they can be tracked and viewed later
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)


        predictions=lr.predict(train_x)
        signature=infer_signature(train_x,predictions) # infers the input and output schema of the model (e.g., feature columns and prediction shape) for model versioning and tracking.

        tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme # captures the storage location where MLflow is tracking the run.

        # Model registry does not work with file here
        # If MLflow is not tracking runs locally (i.e., not using a file system), the model is registered with a name ("ElasticWineModel").
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case
            # See: https://mlflow.org/docs/latest/model-registry.html
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="ElasticWineModel", signature=signature
            )
        # Otherwise, the model is simply logged without registering it in a model registry
        else:
            mlflow.sklearn.log_model(lr, "model", signature=signature)

