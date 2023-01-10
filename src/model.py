import os
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange
from timm.optim.optim_factory import param_groups_weight_decay
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid, save_image
from transformers.optimization import get_cosine_schedule_with_warmup

from src.loss import info_nce_loss, masked_mse_loss
from src.network.decoder import VitDecoder
from src.network.encoder import build_encoder
from src.network.pos_embed import get_1d_sincos_pos_embed


class CANModel(pl.LightningModule):
    def __init__(
        self,
        img_size: int = 224,
        encoder_name: str = "vit_base_patch16",
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        decoder_embed_unmasked_tokens: bool = True,
        projector_hidden_dim: int = 4096,
        projector_out_dim: int = 128,
        noise_embed_in_dim: int = 768,
        noise_embed_hidden_dim: int = 768,
        mask_ratio: float = 0.5,
        norm_pixel_loss: bool = True,
        temperature: float = 0.1,
        noise_std_max: float = 0.05,
        weight_contrast: float = 0.03,
        weight_recon: float = 0.67,
        weight_denoise: float = 0.3,
        lr: float = 2.5e-4,
        optimizer: str = "adamw",
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.05,
        momentum: float = 0.9,
        scheduler: str = "cosine",
        warmup_epochs: int = 0,
        channel_last: bool = False,
    ):
        """Contrastive Masked Autoencoder and Noise Prediction Pretraining Model

        Args:
            img_size: Size of input image
            encoder_name: Name of encoder network
            decoder_embed_dim: Embed dim of decoder
            decoder_depth: Number of transformer blocks in the decoder
            decoder_num_heads: Number of attention heads in the decoder
            decoder_embed_unmasked_tokens: Apply decoder embedding layer on both masked and unmasked tokens.
                Else only apply to masked tokens
            projector_hidden_dim: Hidden dim of projector
            projector_out_dim: Output dim of projector
            noise_embed_in_dim: Dim of noise level sinusoidal embedding
            noise_embed_hidden_dim: Hidden dim of noising embed MLP
            mask_ratio: Ratio of input image patches to mask
            norm_pixel_loss: Calculate loss using normalized pixel value targets
            temperature: Temperature for contrastive loss
            noise_std_max: Maximum noise standard deviation
            weight_contrast: Weight for contrastive loss
            weight_recon: Weight for patch reconstruction loss
            weight_denoise: Weight for denoising loss
            lr: Learning rate (should be scaled with batch size. i.e. lr = base_lr*batch_size/256)
            optimizer: Name of optimizer (adam | adamw | sgd)
            betas: Adam beta parameters
            weight_decay: Optimizer weight decay
            momentum: SGD momentum parameter
            scheduler: Name of learning rate scheduler [cosine, none]
            warmup_epochs: Number of warmup epochs
            channel_last: Change to channel last memory format for possible training speed up
        """
        super().__init__()
        self.save_hyperparameters()
        self.img_size = img_size
        self.encoder_name = encoder_name
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.decoder_embed_unmasked_tokens = decoder_embed_unmasked_tokens
        self.projector_hidden_dim = projector_hidden_dim
        self.projector_out_dim = projector_out_dim
        self.noise_embed_in_dim = noise_embed_in_dim
        self.noise_embed_hidden_dim = noise_embed_hidden_dim
        self.mask_ratio = mask_ratio
        self.norm_pixel_loss = norm_pixel_loss
        self.temperature = temperature
        self.noise_std_max = noise_std_max
        self.weight_contrast = weight_contrast
        self.weight_recon = weight_recon
        self.weight_denoise = weight_denoise
        self.lr = lr
        self.optimizer = optimizer
        self.betas = betas
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.channel_last = channel_last

        # Initialize networks
        self.encoder, self.patch_size = build_encoder(
            encoder_name, img_size=self.img_size
        )
        self.decoder = VitDecoder(
            patch_size=self.patch_size,
            num_patches=self.encoder.patch_embed.num_patches,
            in_dim=self.encoder.embed_dim,
            embed_dim=self.decoder_embed_dim,
            depth=self.decoder_depth,
            num_heads=self.decoder_num_heads,
            embed_unmasked_tokens=self.decoder_embed_unmasked_tokens,
        )
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.embed_dim, self.projector_hidden_dim),
            nn.BatchNorm1d(self.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.projector_hidden_dim, self.projector_hidden_dim),
            nn.BatchNorm1d(self.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.projector_hidden_dim, self.projector_out_dim),
        )
        # Based on updated openreview version (as of Nov 17, 2022), the MLP is two layers
        # without BN (maybe?) and input and hidden dims the same as the encoder embedding
        self.noise_embed = nn.Sequential(
            nn.Linear(self.noise_embed_in_dim, self.noise_embed_hidden_dim),
            nn.ReLU(),
            nn.Linear(
                self.noise_embed_hidden_dim,
                self.encoder.embed_dim
                if self.decoder_embed_unmasked_tokens
                else self.decoder_embed_dim,
            ),
        )

        # Change to channel last memory format
        # https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html
        if self.channel_last:
            self = self.to(memory_format=torch.channels_last)

    def patchify(self, x: torch.Tensor):
        """Rearrange image into patches

        Args:
            x: Tensor of size (b, 3, h, w)

        Return:
            x: Tensor of size (b, h*w, patch_size^2 * 3)
        """
        assert x.shape[2] == x.shape[3] and x.shape[2] % self.patch_size == 0

        return rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size,
        )

    def unpatchify(self, x: torch.Tensor):
        """Rearrange patches back to an image

        Args:
            x: Tensor of size (b, h*w, patch_size^2 * 3)

        Return:
            x: Tensor of size (b, 3, h, w)
        """
        h = w = int(x.shape[1] ** 0.5)
        return rearrange(
            x,
            " b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            p1=self.patch_size,
            p2=self.patch_size,
            h=h,
            w=w,
        )

    def log_samples(self, inp: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor):
        """Log sample images"""
        # Only log up to 16 images
        inp, pred, mask = inp[:16], pred[:16], mask[:16]

        # Patchify the input image
        inp = self.patchify(inp)

        # Merge original and predicted patches
        pred = pred * mask[:, :, None]
        inp = inp * (1 - mask[:, :, None])
        res = self.unpatchify(inp) + self.unpatchify(pred)

        # Log result
        if "CSVLogger" in str(self.logger.__class__):
            path = os.path.join(
                self.logger.log_dir,  # type:ignore
                "samples",
            )
            if not os.path.exists(path):
                os.makedirs(path)
            filename = os.path.join(path, str(self.current_epoch) + "ep.png")
            save_image(res, filename, nrow=4, normalize=True)
        elif "WandbLogger" in str(self.logger.__class__):
            grid = make_grid(res, nrow=4, normalize=True)
            self.logger.log_image(key="sample", images=[grid])  # type:ignore

    @torch.no_grad()
    def add_noise(self, x: torch.Tensor):
        """Add noise to input image

        Args:
            x: Tensor of size (b, c, h, w)

        Return:
            x_noise: x tensor with added Gaussian noise of size (b, c, h, w)
            noise: Noise tensor of size (b, c, h, w)
            std: Noise standard deviation (noise level) tensor of size (b,)
        """
        # Sample std uniformly from [0, self.noise_std_max]
        std = torch.rand(x.size(0), device=x.device) * self.noise_std_max

        # Sample noise
        noise = torch.randn_like(x) * std[:, None, None, None]

        # Add noise to x
        x_noise = x + noise

        return x_noise, noise, std

    def shared_step(
        self,
        x: Tuple[torch.Tensor, torch.Tensor],
        mode: str = "train",
        batch_idx: Optional[int] = None,
    ):
        x1, x2 = x

        if self.channel_last:
            x1 = x1.to(memory_format=torch.channels_last)  # type:ignore
            x2 = x2.to(memory_format=torch.channels_last)  # type:ignore

        # Add noise to views
        x1_noise, noise1, std1 = self.add_noise(x1)
        x2_noise, noise2, std2 = self.add_noise(x2)

        # Mask and extract features
        z1, mask1, idx_unshuffle1 = self.encoder(x1_noise, self.mask_ratio)
        z2, mask2, idx_unshuffle2 = self.encoder(x2_noise, self.mask_ratio)

        # Pass mean encoder features through projector
        u1 = self.projector(torch.mean(z1[:, 1:, :], dim=1))  # Skip cls token
        u2 = self.projector(torch.mean(z2[:, 1:, :], dim=1))

        # Generate noise level embedding
        p1 = self.noise_embed(
            get_1d_sincos_pos_embed(std1, dim=self.noise_embed_in_dim)
        )
        p2 = self.noise_embed(
            get_1d_sincos_pos_embed(std2, dim=self.noise_embed_in_dim)
        )

        # Predict masked patches and noise
        x1_pred = self.decoder(z1, idx_unshuffle1, p1)
        x2_pred = self.decoder(z2, idx_unshuffle2, p2)

        # Contrastive loss
        loss_contrast = info_nce_loss(torch.cat([u1, u2]), temperature=self.temperature)

        # Patch reconstruction loss
        loss_recon = (
            masked_mse_loss(x1_pred, self.patchify(x1), mask1, self.norm_pixel_loss)
            + masked_mse_loss(x2_pred, self.patchify(x2), mask2, self.norm_pixel_loss)
        ) / 2

        # Denoising loss
        loss_denoise = (
            masked_mse_loss(
                x1_pred, self.patchify(noise1), 1 - mask1, self.norm_pixel_loss
            )
            + masked_mse_loss(
                x2_pred, self.patchify(noise2), 1 - mask2, self.norm_pixel_loss
            )
        ) / 2

        # Combined loss
        loss = (
            self.weight_contrast * loss_contrast
            + self.weight_recon * loss_recon
            + self.weight_denoise * loss_denoise
        )

        # Log
        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_loss_contrast", loss_contrast)
        self.log(f"{mode}_loss_recon", loss_recon)
        self.log(f"{mode}_loss_denoise", loss_denoise)
        if mode == "val" and batch_idx == 0:
            self.log_samples(x1, x1_pred, mask1)

        return {"loss": loss}

    def training_step(self, x, _):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return self.shared_step(x, mode="train")

    def validation_step(self, x, batch_idx):
        return self.shared_step(x, mode="val", batch_idx=batch_idx)

    def configure_optimizers(self):
        """Initialize optimizer and learning rate schedule"""
        # Set weight decay to 0 for bias and norm layers (following MAE)
        params = param_groups_weight_decay(
            self.encoder, self.weight_decay
        ) + param_groups_weight_decay(self.decoder, self.weight_decay)

        # Optimizer
        if self.optimizer == "adam":
            optimizer = Adam(
                params,
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adamw":
            optimizer = AdamW(
                params,
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "sgd":
            optimizer = SGD(
                params,
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(
                f"{self.optimizer} is not an available optimizer. Should be one of ['adam', 'adamw', 'sgd']"
            )

        # Learning rate schedule
        if self.scheduler == "cosine":
            epoch_steps = (
                self.trainer.estimated_stepping_batches
                // self.trainer.max_epochs  # type:ignore
            )
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_training_steps=self.trainer.estimated_stepping_batches,  # type:ignore
                num_warmup_steps=epoch_steps * self.warmup_epochs,
            )
        elif self.scheduler == "none":
            scheduler = LambdaLR(optimizer, lambda _: 1)
        else:
            raise ValueError(
                f"{self.scheduler} is not an available optimizer. Should be one of ['cosine', 'none']"
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
