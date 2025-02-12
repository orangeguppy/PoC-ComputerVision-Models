import torchvision
import torch.nn as nn

class VGG19Model(nn.Module):
    def __init__(self):
        super(VGG19Model, self).__init__()
        # Use a VGG-19 backbone like in the original implementation as a feature extractor
        self.backbone = torchvision.models.vgg19(pretrained=True).features.eval() # Only use the feature extractor, exclude non-classification layers
        print(self.backbone)

        # Avoid accidentally training the model
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Set the layers to compute style/content losses
        self.style_layers = [0, 2, 5, 7]
        self.content_layer = 7

        # Store activations for style layers and content layer
        self.style_activations = OrderedDict()
        self.content_activations = None

        # Register hooks for the forward and back layers
        # Hooks are triggered during forward passes
        self.register_hooks()

    # Register hooks to record outputs at the content and style layers
    def register_hooks(self):
        # We must use partial because if we don't, Python's late binding will make it such that all hooks are registered with the last layer_idx
        for layer_idx in self.style_layers:
            self.backbone[layer_idx].register_forward_hook(partial(self.forward_hook, layer_idx=layer_idx, is_style=True))

        # Only need to register a single layer for the content layer
        self.backbone[self.content_layer].register_forward_hook(partial(self.forward_hook, layer_idx=self.content_layer, is_style=False))

    # Hook that keeps track of activations for the specified modules
    def forward_hook(self, module, input, output, layer_idx, is_style):
        if is_style:
            self.style_activations[layer_idx] = output
        else:
            self.content_activations = output

    def forward(self, img):
        # Re-initialise activations during each forward pass
        self.style_activations = OrderedDict()
        self.content_activations = None

        # Extract features with VGG-19 backbone
        self.backbone(img)

        # Return activation maps for content and style
        return self.content_activations, self.style_activations