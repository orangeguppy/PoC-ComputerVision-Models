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
Deeper layers in the CNN are used to compute content loss because deeper layers usually contain semantic features (high number of channels, small feature maps) that provide the high-level structure of the image content. The content loss measures the MSE between feature representations of an image at a given layer \( l \).

```math
L_{\text{content}}(\tilde{p}, \tilde{x}, l) = \frac{1}{2} \sum_{i,j} \left( F^l_{ij} - P^l_{ij} \right)^2
```

where:
- $F^l_{ij}$ represents the feature map of the generated image at layer $l$.
- $P^l_{ij}$ represents the feature map of the target image at the same layer.
- The sum is taken over all spatial locations $(i, j)$ in the feature map.

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
