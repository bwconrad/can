# CAN: Contrastive Masked Autoencoders and Noise Prediction Pretraining

PyTorch reimplementation of ["A simple, efficient and scalable contrastive masked autoencoder for learning visual representations"](https://arxiv.org/abs/2210.16870).


<p align="center">
<img src="assets/can.png" width="80%" style={text-align: center;}/>
</p>

### Requirements
- Python 3.8+
- `pip install -r requirements`

### Usage
To pretrain a ViT-b/16 network run:
```
python train.py --accelerator gpu --devices 1 --precision 16  --data.root path/to/data/
--max_epochs 1000 --data.batch_size 256 --model.encoder_name vit_base_patch16
--model.mask_ratio 0.5 --mode.weight_contrast 0.03 --model.weight_recon 0.67 
--model.weight_denoise 0.3
```
- Run `python train.py --help` for descriptions of all options.
- `--model.encoder_name` can be one of `vit_tiny_patch16, vit_small_patch16, vit_base_patch16, vit_large_patch16, vit_huge_patch14`.

