import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from utils import data_generator
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Executor:
    def __init__(self, path_testset, save_path, path_pth:str, normalize: bool, path_trainset: str, product: bool) -> None:
        self.path_testset = path_testset
        self.save_path = save_path
        self.path_pth = path_pth
        self.normalize = normalize
        self.path_trainset = path_trainset
        self.product = product

        self._log_initial_parameters()

    def _log_initial_parameters(self):
        logging.info("Executor Parameters:")
        params = vars(self)  # Fetches all class attributes
        for key, value in params.items():
            logging.info(f"{key}: {value}")

    def execute(self, model_generator, batch_size):
        tt_transforms = transforms.Compose(
            [
                transforms.Resize((model_generator.image_size, model_generator.image_size)),
                transforms.ToTensor()
            ]
        )
        if self.normalize:
            ds_train_1 = data_generator.Dataset(self.path_trainset, transform=tt_transforms)
            dl_train_1 = DataLoader(ds_train_1, batch_size, shuffle=True, num_workers=0, pin_memory=True)
            mean, std = batch_mean_and_sd(dl_train_1)
            tt_transforms = transforms.Compose(
                [
                    transforms.Resize((model_generator.image_size, model_generator.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean.tolist(), std.tolist()),
                ]
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_generator.model
        model.load_state_dict(torch.load(os.path.join(self.path_pth, model_generator.name) + '.pth'))
        model.eval()
        model = model.to(device)

        cls_names = [str(i) for i in range(int(model_generator.num_class))]
        idx_to_class = {i: cls_name for i, cls_name in enumerate(cls_names)}

        ds_val = data_generator.Dataset(self.path_testset, transform=tt_transforms)
        dl_val = DataLoader(ds_val, batch_size, num_workers=0, pin_memory=True, drop_last=True, shuffle=False)
        
        y_preds, y_trues, y_scores = [], [], []
        with torch.no_grad():
            for i, (image, label) in enumerate(tqdm(dl_val, 0)):
                image = image.to(device)
                output = model(image)
                y_scores.extend(output.cpu().numpy())
                y_preds.extend(torch.argmax(output, axis=1).cpu().numpy())
                y_trues.extend(label.cpu().numpy())

        y_preds_ = list(map(lambda x: idx_to_class[x], y_preds))
        y_trues_ = list(map(lambda x: idx_to_class[x], y_trues))

        cm = confusion_matrix(y_trues_, y_preds_)
        fig, ax = plt.subplots(figsize=(12, 8))
        mask = np.eye(len(cls_names), dtype=bool)

        sns.heatmap(cm, annot=True, fmt='g', annot_kws={'size': 20},
                    cmap=sns.color_palette(['green'], as_cmap=True), cbar=False,
                    mask=~mask, vmin=0, vmax=cm.max(),
                    yticklabels=cls_names, xticklabels=cls_names, ax=ax)

        sns.heatmap(cm, annot=True, fmt='g', annot_kws={'size': 20},
                    cmap=sns.color_palette(['red'], as_cmap=True), cbar=False,
                    mask=mask, vmin=0, vmax=cm.max(),
                    yticklabels=cls_names, xticklabels=cls_names, ax=ax)

        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.tick_params(labelsize=15, length=0)
        ax.set_title('Confusion Matrix ' + str(model_generator.name), size=20, pad=20)
        ax.set_xlabel('Predicted Values', size=15)
        ax.set_ylabel('Actual Values', size=15)
        
        plt.tight_layout()
        plt.savefig(self.save_path + '/' + model_generator.name + '_conf_mtrx.png', dpi=600)
        plt.clf()

        report = classification_report(y_trues_, y_preds_, target_names=cls_names, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        fig, ax = plt.subplots(figsize=(10, len(report_df) * 0.6))
        sns.heatmap(report_df.iloc[:-1, :].T, annot=True, fmt=".2f", cmap="YlGn", cbar=False, ax=ax)
        plt.title(f"Classification Report for {model_generator.name}", size=15)
        plt.xlabel("Metrics")
        plt.ylabel("Classes")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.save_path + '/' + model_generator.name + '_classification_report.png', dpi=600)
        plt.clf()

        y_trues_bin = np.eye(len(cls_names))[y_trues]
        roc_auc_scores = {}
        for i, cls_name in enumerate(cls_names):
            roc_auc_scores[cls_name] = roc_auc_score(y_trues_bin[:, i], np.array(y_scores)[:, i])

        roc_auc_df = pd.DataFrame(list(roc_auc_scores.items()), columns=['Class', 'ROC AUC Score'])
        
        fig, ax = plt.subplots(figsize=(8, len(roc_auc_df) * 0.5))
        sns.heatmap(roc_auc_df.set_index('Class').T, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=ax)
        plt.title(f"ROC AUC Scores for {model_generator.name}", size=15)
        plt.tight_layout()
        plt.savefig(self.save_path + '/' + model_generator.name + '_roc_auc_scores.png', dpi=600)
        plt.clf()

def batch_mean_and_sd(dl_val):
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)
    logging.info("-"*50)
    logging.info("Normalization started")
    for i, (images, label) in enumerate(tqdm(dl_val, 0)):
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels
    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2) 
    logging.info("mean: {} - std: {} ".format(mean, std))
    logging.info("Normalization finished")       
    return mean, std
