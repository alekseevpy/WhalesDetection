import os
import random
import shutil
from tqdm import tqdm
import data_preparation
import train_modules
import validation
import yaml

def run_pipeline():
    print('hello hello, hope u feeling good')
    if os.path.exists('../whales_processed'):
        print(f"Папка с данными уже существует, пропускаем подготовку данных")
    else:
        print(f"Приступаю к подготовке данных")
        data_preparation.prepare_data('../config/data_preparation_config.yaml')

    train_loader, model, shape = train_modules.prepare_training_modules('../config/train_modules_config.yaml')

    with open('../config/train_modules_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config = config['trainer_config']

    model = train_modules.train_model(model, train_loader, shape, epochs = config['num_epochs'], lr = config['lr'], eval_step = config['eval_step'])

    validation.evaluate_final_model(model, shape)

if __name__ == "__main__":
  run_pipeline()