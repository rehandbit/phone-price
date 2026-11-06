from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging


def run_training():
    train_path, test_path = DataIngestion().initiate_data_ingestion()
    train_array, test_array, train_groups = DataTransformation().initiate_data_transformation(train_path, test_path)
    info = ModelTrainer().initiate_model_trainer(train_array, test_array, train_groups)
    logging.info("Training pipeline finished")
    return info


if __name__ == "__main__":
    print(run_training())
