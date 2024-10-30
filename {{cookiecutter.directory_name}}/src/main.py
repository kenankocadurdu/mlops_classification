
import cv2
import glob
import logging
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from utils import model_generator
import torch
import shutil

@hydra.main(config_path="../config", config_name="preprocessing", version_base=None)
def preprocess_images(config: DictConfig):
    logging.info("-" * 50)
    logging.info("Preprocessing started...")

    pre_processor = instantiate(config.preprocessing)
    dir_root = os.getcwd() 
    total_saved_images = 0
    
    for data_dir in config.data["dirs"]:
        logging.info(f"Processing directory: {data_dir}...")
        
        for class_label in config.data["classes"]:
            image_class_path = os.path.join(data_dir, class_label)
            logging.info(f"Processing class: {image_class_path}")
            image_paths = glob.glob(os.path.join(dir_root, config.data["raw_path"], image_class_path, f"*.{config.data['file']}"))

            save_class_path = os.path.join(dir_root, config.data["processed_path"], image_class_path)
            if os.path.exists(save_class_path):
                shutil.rmtree(save_class_path) 
            os.makedirs(save_class_path)

            for idx, image_path in enumerate(tqdm(image_paths, desc=f"Processing images for {class_label}")):
                save_path = os.path.join(save_class_path, os.path.basename(image_path))

                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                processed_image = pre_processor.do_process(image)

                cv2.imwrite(save_path, processed_image)
                total_saved_images += 1
    
    logging.info(f"Total {total_saved_images} images saved.")
    logging.info("Image preprocessing finished.")
    logging.info("-" * 50)


@hydra.main(config_path="../config", config_name="training", version_base=None)
def do_training(config: DictConfig):
    logging.info("-"*50)
    executer_training = instantiate(config.training)
    model_names = OmegaConf.to_container(config.model, resolve=True)
    model_params = OmegaConf.to_container(config.model_params, resolve=True)
    for model_name in model_names:
        model_gen = model_generator.generator(model_name, model_params['num_class'], model_params['image_size'])
        logging.info("Process training with "+ model_gen.name +" started...")
        executer_training.execute(model_gen)
        logging.info("Process training with "+ model_gen.name +" finished...")

@hydra.main(config_path="../config", config_name="training", version_base=None)
def do_evaluation(config: DictConfig):
    executer_training = instantiate(config.training)
    executer_prediction = instantiate(config.evaluation)
    model_names = OmegaConf.to_container(config.model, resolve=True)
    model_hyperparams = OmegaConf.to_container(config.model_params, resolve=True)
    for model_name in model_names:
        model_gen = model_generator.generator(model_name,  model_hyperparams['num_class'], model_hyperparams['image_size'])
        logging.info("Process prediction with "+ model_gen.name +" started...")
        executer_prediction.execute(model_gen, executer_training.batch_size)
        logging.info("Process prediction with "+ model_gen.name +" finished...")

if __name__ == "__main__":
    torch.manual_seed(43)
    preprocess_images()
    do_training()
    do_evaluation()