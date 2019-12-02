## Wasserstein GAN

Tensorflow implementation of Wasserstein GAN.
The output path is log/.

```
python wgan.py --data mnist --model dcgan --gpus 0 --output output_path --shape [28,28,3] --de_path database
or
python wgan.py --data kuzushi --model dcgan --gpus 0 --output output_path --shape [512,512,3] --de_path database
```
