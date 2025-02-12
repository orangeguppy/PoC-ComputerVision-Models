# PoC ComputerVision Models
## Neural Style Transfer
Style Transfer outputs can be viewed under the 'Demo Outputs' section.
### Model Architecture
VGG-19 was used as a feature extractor, pretrained on IMAGENET.

### Loss Objective
The loss consists of 3 components: Content Loss, Style Loss, and Total Variation Loss.

```math
L = \alpha L_{\text{content}} + \beta L_{\text{style}} + \lambda_{\text{tv}} L_{\text{tv}}
```
#### Content Loss $L_{\text{content}}$
Deeper layers in the CNN are used to compute content loss because deeper layers usually contain semantic features (high number of channels, small feature maps) that provide the high-level structure of the image content. The content loss measures the MSE between feature representations of an image at a layer $l$.

```math
L_{\text{content}}(\tilde{p}, \tilde{x}, l) = \frac{1}{2} \sum_{i,j} \left( F^l_{ij} - P^l_{ij} \right)^2
```

where:
- $F^l_{ij}$ represents the feature map of the generated image at layer $l$.
- $P^l_{ij}$ represents the feature map of the target image at the same layer.
- The sum is taken over all spatial locations $(i, j)$ in the feature map.

#### Style Loss $L_{\text{style}}$
Earlier layers in the CNN are used to compute style loss because they usually capture local patterns, rather than high-level semantic features. Local patterns involve image characteristics like colour distributions, edges, and textures, making them great for extracting style features from.

We use multiple layers for style because different layers at different depths capture different levels of textures and spatial patterns. An important point to note is that the Gram matrices of the feature maps are used in calculating style loss. This is because the Gram matrix gives us the correlations between different feature maps within the same CNN layer. The Gram matrix of a CNN feature map from layer $l$ is

```math
G^l_{ij} = \sum_k F^l_{ik} F^l_{jk}
```

where
- $G^l_{ij}$ is the inner product between the vectorised feature maps $i$ and $j$ in layer $l$

The style loss contributed by a single layer is defined as

```math
E_l = \frac{1}{4 N_l^2 M_l^2} \sum_{i,j} \left( G^l_{ij} - A^l_{ij} \right)^2
```

where 
- $N_l$ is the number of feature maps of size $M_l$ for layer $l$
- $\frac{1}{4 N_l^2 M_l^2}$ is a normalisation constant to ensure that the feature map's contribution is invariant to scale

Hence, the total style loss is defined as

```math
L_{style}(\tilde{a}, \tilde{x})\sum^L_{l=0}w_lE_l
```
where $w_l$ are tunable weighting factors for each layer's contribution

#### Total Variation Loss $L_{\text{tv}}$
I chose to include TV loss to reduce the noise in the output image. 


### Hyperparameters
Learning Rate: 0.01, Optimiser: Adam, No. Iterations: 1500, (alpha, beta, tv_lambda): (1e3, 5e6, 1e-6)

### Training Procedure Summary
1) Instantiate the model and attach hooks at intermediate layers to capture activation maps for computing Content and Style Loss
2) Forward pass the image through the model. Ensure that gradient tracking is enabled for the image tensor. The generated image is initialised by cloning the original input.
3) Compute the loss (defined earlier) and backpropagate.
4) After training, denormalise the images and permute tensor dimensions for viewing with MatPlotLib.

## Demo Outputs
I'm actually happy with these wahahahaha
![Lilypads with Monet's style](images/style_transfer_1.png)
![Lilypads with Monet's style](images/style_transfer_3.png)
![Lilypads with Monet's style](images/style_transfer_2.png)



## CLIP

## GAN

## StyleGAN
