<p align="center">
  <a href="https://www.python.org"><img alt="Python Version" src="https://img.shields.io/badge/Python-3.7.x-brightgreen.svg" /></a>
  <a href="https://www.tensorflow.org/install"><img alt="TensorFlow Version" src="https://img.shields.io/badge/TensorFlow-2.8.x-red.svg" /></a>
  <a href="https://github.com/rob-med-usp/seizure-prediction/blob/main/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-yellow.svg" /></a>
</p>

--------------------------------------------------------------------------------

# Temporal Multi-Channel Vision Transformer (TMC-ViT)
This project implements a Transformer-based model called Temporal Multi-Channel Vision Transformer (TMC-ViT). The TMC-ViT was developed to adapt the Vision Transformer model proposed by Dosovitskiy et al.[[1]](#1) for processing multi-channel temporal signals as input. In this example, we will predict 18 gestures from the Ninapro DB05 Database. This has also been implemented in a Google Colab project<https://colab.research.google.com/drive/1ZWhzv8EOtwCHfuytcvOSKZQ76hSCQdFJ?hl=pt-BR#scrollTo=5K2na9pj0KJn>.

## Loading the data
The input data must already be divided into training and test sets, with 200 ms samples. Use one separate repetition for testing. More information on the data preprocessing can be found in [[2]](#2).

## References
<a id="1">[1]</a> 
Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. 1–21. http://arxiv.org/abs/2010.11929
<a id="2">[2]</a> 
R. V. Godoy et al., "Electromyography-Based, Robust Hand Motion Classification Employing Temporal Multi-Channel Vision Transformers," in IEEE Robotics and Automation Letters, vol. 7, no. 4, pp. 10200-10207, Oct. 2022, doi: 10.1109/LRA.2022.3192623.
