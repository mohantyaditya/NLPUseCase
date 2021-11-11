import argparse
import os
import random
import shutil
from src.utils.common import read_yaml, create_directories
from tqdm import tqdm
import logging
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
STAGE = "Stage 3"

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def main(config_path,param_path):
    config = read_yaml(config_path)
    params = read_yaml(param_path)
    artifacts = config["artifacts"]

    featurized_data_dir_path = os.path.join(artifacts["ARTIFACT_DIR"], artifacts["FEATURIZED_DATA"])
    featurized_train_data_path = os.path.join(featurized_data_dir_path, artifacts["FEATURIZED_OUT_TRAIN"])

    model_dir_path = os.path.join(artifacts["ARTIFACT_DIR"], artifacts["MODEL_DIR"])
    create_directories([model_dir_path])
    model_path = os.path.join(model_dir_path, artifacts["MODEL_NAME"])

    #model_dir_path = os.path.join(artifacts["ARTIFACTS_DIR"], artifacts["MODEL_DIR"])

    matrix = joblib.load(featurized_train_data_path)

    labels = np.squeeze(matrix[:,1].toarray())

    x = matrix[:,2:]

    logging.info(f"Input matrix size is {matrix.shape}")
    logging.info(f"Matrix size is {x.shape}")
    logging.info(f"Y matrix size or label size is {labels.shape}")

    seed = params["train"]["seed"]
    n_est = params["train"]["n_est"]
    min_split = params["train"]["min_split"]

    model = RandomForestClassifier(
        n_estimators = n_est,min_samples_split = min_split,n_jobs = 2,random_state = seed

    )
    model.fit(x,labels)
    logging.info(f"Model training completed")

    joblib.dump(model,model_path)

    logging.info(f"Model stored at {model_path}")





if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config,param_path = parsed_args.params)
        logging.info(f">>>>> stage {STAGE} completed! all the data are saved in local <<<<<n")
    except Exception as e:
        logging.exception(e)
        raise e