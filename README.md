# APD-Nets
Image denoising methods using deep neural networks have achieved a great progress in the image restoration. However, the recovered images restored by these deep denoising methods usually suffer from severe over-smoothness, artifacts, and detail loss. To improve the quality of restored images, we first propose Supplemental Priors (SP) method to adaptively predict depth-directed and sample-directed prior information for the reconstruction (decoder) networks. Furthermore, the over-parameterized deep neural networks and too precise supplemental prior information may cause an over-fitting, restricting the performance promotion. To improve the generalization of denoising networks, we further propose Regularization Priors (RP) method to flexibly learn depth-directed and dataset-directed regularization noise for the retrieving (encoder) networks. By respectively integrating the encoder and decoder with these plug-and-play RP block and SP block, we propose the final Adaptive Prior Denoising Networks, called APD-Nets. APD-Nets is the first attempt to simultaneously regularize and supplement denoising networks from the adaptive priors’ view with drawing learning-based mechanism into producing adaptive regularization noise and supplemental information. Extensive experiment results demonstrate our method significantly improves the generalization of denoising networks and the quality of restored images with greatly outperforming the traditional deep denoising methods both quantitatively and visually.

### The code will be released as soon as possible.

## Network Architecture
<img src="https://github.com/JiangBoCS/APD-Nets/blob/main/The%20framework%20of%20APD-Nets.png"
     alt="Picture displays an error."
     style="zoom:30%"/>
<center><p>The framework of APD-Nets. APD-Nets includes two main components, i.e., encoder with RP block and decoder with SP block.</p></center>

## Structure of Supplemental Prior (SP).
<img src="https://github.com/JiangBoCS/APD-Nets/blob/main/The%20structure%20of%20Supplemental%20Prior%20(SP)%20block.png"
     alt="Picture displays an error."
     style="zoom:30%"/>
<center><p>The structure of Supplemental Prior (SP) block. The SP block includes External Reconstruction Extent (ERE), Semantic Channel Attention (SCA) branch (I), and Semantic Spatial Attention (SSA) branch (II). ``CNR'' stands for the convolutional, Normalization and ReLU layers in series. ``DNR'' represents the deconvolutional, Normalization and ReLU layers in series. </p></center>

## Framework of Regularization Priors (RP).
<img src="https://github.com/JiangBoCS/APD-Nets/blob/main/The%20framework%20of%20Regularization%20priors%20(RP)%20block.png"
     alt="Picture displays an error."
     style="zoom:30%"/>
<center><p>The framework of Regularization priors (RP) block. RP block includes External Captured Extent (ECE) and learnable noise sampled from Gaussian and Uniform noise distributions.</p></center>
