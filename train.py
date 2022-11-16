import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data import DuelViewDataModule
from src.model import CANModel
from src.pl_utils import MyLightningArgumentParser, init_logger

model_class = CANModel
dm_class = DuelViewDataModule

# Parse arguments
parser = MyLightningArgumentParser()
parser.add_lightning_class_args(pl.Trainer, None)  # type:ignore
parser.add_lightning_class_args(dm_class, "data")
parser.add_lightning_class_args(model_class, "model")
parser.link_arguments("data.crop_size", "model.img_size")
args = parser.parse_args()

# Setup trainer
logger = init_logger(args)
checkpoint_callback = ModelCheckpoint(
    filename="best-{epoch}-{val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_last=True,
)
dm = dm_class(**args["data"])
model = model_class(**args["model"])

trainer = pl.Trainer.from_argparse_args(
    args, logger=logger, callbacks=[checkpoint_callback]
)

# Train
trainer.tune(model, dm)
trainer.fit(model, dm)
