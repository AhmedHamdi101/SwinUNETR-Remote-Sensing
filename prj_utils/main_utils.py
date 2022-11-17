from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from models.ssl_ptl_module import SSL_PTLModule
from dataloader import SentinelDataloader

import pytorch_lightning as pl
from prj_utils.model_utils import load_from_pretrained_weights

def main_validate(args):
    dataloaders = SentinelDataloader(root_img_dir="/home/amhamdi/Desktop/swinunetr_selfsupervised/cropped_data" , train_df="/home/amhamdi/Desktop/swinunetr_selfsupervised/train_samples.csv",
                                        valid_df="/home/amhamdi/Desktop/swinunetr_selfsupervised/valid_samples.csv" , test_df="/home/amhamdi/Desktop/swinunetr_selfsupervised/valid_samples.csv",
                                        batch_size=1 , num_workers= 12)
    dataloaders.setup(stage="fit")
    validation_dataloader = dataloaders.val_dataloader()

    model = SSL_PTLModule(0.0001)
    model = load_from_pretrained_weights(model , "/home/amhamdi/Desktop/swinunetr_selfsupervised/checkpoints/checkpoint-epoch=09-step=23500--v1.ckpt")

    trainer = pl.Trainer(accelerator="gpu",
                            devices=1,
                            num_nodes=1,)

    trainer.validate(model = model , dataloaders=validation_dataloader)


def main_train(args):
    logger = CometLogger(
            api_key="oZ01tNc0QwfzPcXODfwkCrDdL",
            workspace="amhamdi",
            # project_name=,  # Optional
            experiment_name="tmp",  # Optional
            # experiment_key="f5d5b1e9139d4bbca9e7621973fae8bb"
        )


    checkpoint_callback = ModelCheckpoint(
            dirpath="/home/amhamdi/Desktop/swinunetr_selfsupervised/checkpoints/",
            filename="checkpoint" + "-{epoch:02d}-{step:02d}-",
            save_on_train_epoch_end=True,
            save_top_k=-1,
            every_n_train_steps=500)

    dataloaders = SentinelDataloader(root_img_dir="/home/amhamdi/Desktop/swinunetr_selfsupervised/cropped_data" , train_df="/home/amhamdi/Desktop/swinunetr_selfsupervised/train_samples.csv",
                                        valid_df="/home/amhamdi/Desktop/swinunetr_selfsupervised/valid_samples.csv" , test_df="/home/amhamdi/Desktop/swinunetr_selfsupervised/valid_samples.csv",
                                        batch_size=6 , num_workers= 6)
    dataloaders.setup(stage="fit")

    train_dataloader = dataloaders.train_dataloader()
    validation_dataloader = dataloaders.val_dataloader()
    model = SSL_PTLModule(0.00005)
    trainer = pl.Trainer(   accelerator="gpu",
                            devices=1,
                            num_nodes=1,
                            max_epochs=100,
                            logger=logger,
                            callbacks=checkpoint_callback,
                            auto_lr_find=True,
                            # resume_from_checkpoint="/home/amhamdi/Desktop/swinunetr_selfsupervised/checkpoints/checkpoint-epoch=24-step=58000-.ckpt",
                            log_every_n_steps=5)
    trainer.fit(model=model,train_dataloaders = train_dataloader , val_dataloaders=validation_dataloader) 