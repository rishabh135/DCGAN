# DCGAN
Implementation of DCGAN paper with bernoulli distribution used as input noise vector and slight modification to train it better and skip modal collapse. The results are taken on anime dataset generated from danbooru.donmai.us website using 21k images as test samples and training on 50 epochs with learning-rate = 0.002 . Lots of slight modifications have been done to improve GANs training.

### Things I've learned
1. GANs are really hard to train and try at your own luck.
2. DCGAN generally works well, simply adding fully-connected layers causes problems and more layers for Generator yields better images, in the sense that Generator should be more powerful than Discriminator.
4. Add noise to Discriminator's inputs and labels helps stablize training.
5. Usage of different input and output resolution (64x64 vs 96x96), there seems no obvious difference during training, the generated images are also very similar.
6. Binray Noise as G's input amazingly works, but the images are not as good as those with Gussian Noise.
7. For some additional GAN tips, see @soumith's [ganhacks](https://github.com/soumith/ganhacks)
8. Credits to [jayleicn](https://github.com/jayleicn) and [carpedm20](https://github.com/carpedm20/DCGAN-tensorflow)
