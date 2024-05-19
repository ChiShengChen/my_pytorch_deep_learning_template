import torch
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torch import nn
from timm.models import create_model
import captum
from captum.attr import IntegratedGradients, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz
import os

def get_cdict():
    _jet_data = {
              # 'red':   ((0.00, 0, 0),
              #          (0.35, 0, 0),
              #          (0.66, 1, 1),
              #          (0.89, 1, 1),
              #          (1.00, 0.5, 0.5)),
              'red':   ((0.00, 0, 0),
                       (0.35, 0.5, 0.5),
                       (0.66, 1, 1),
                       (0.89, 1, 1),
                       (1.00, 0.8, 0.8)),
             'green': ((0.000, 0, 0),
                       (0.125, 0, 0),
                       (0.375, 1, 1),
                       (0.640, 1, 1),
                       (0.910, 0.3, 0.3),
                       (1.000, 0, 0)),
             # 'blue':  ((0.00, 0.5, 0.5),
             #           (0.11, 1, 1),
             #           (0.34, 1, 1),
             #           (0.65, 0, 0),
             #           (1.00, 0, 0))}
             'blue':  ((0.00, 0.30, 0.30),
                       (0.25, 0.8, 0.8),
                       (0.34, 0.8, 0.8),
                       (0.65, 0, 0),
                       (1.00, 0, 0))
             }
    return _jet_data

# Load your pre-trained model weights
weight_path = "/home/meow/cnfood241/food_code_cn241/outputs/repvit_m0_9_448_cn241/model.pth"
model = create_model('repvit_m0_9', pretrained=True, num_classes=241)
model.load_state_dict(torch.load(weight_path))
model.eval()

print(model)
# target = 63

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((550, 550)),
    transforms.RandomCrop(448, padding=8),
    transforms.ToTensor()
])

transform_normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

# Load and preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image.requires_grad = True  # Ensure requires_grad is True
    return image



data = [
    "015_000101heat", "021_5.015719793109707heat", "027_000059heat", "024_000059heat",
    "046_000010heat", "051_000025heat", "063_63n000001heat", "076_76n000009heat",
    "073_000109heat", "100_000050heat", "109_000009heat", "124_000001heat",
    "122_000033heat", "129_000024heat", "126_000029heat", "134_000042heat",
    "144_000071heat", "156_156n000000heat", "192_410.82251889566675heat", "194_000000heat",
    "207_000031heat", "216_8.78442873590168heat", "218_2.7672608330720756heat",
    "219_1.8203088069711892heat", "225_6.320396054494243heat", "228_8.067893909653899heat",
    "229_5.30089462658354heat", "232_2.980463359419659heat", "234_13.644777726796848heat",
    "237_10.172250266846223heat", "240_1.7018671383949395heat"
]

class_targets = []
file_names = []

for item in data:
    # Split the string at the underscore
    parts = item.split('_')
    # Extract the class target and the file name
    class_target = parts[0]
    file_name = parts[1].replace("heat", "")
    
    # Append the results to the lists
    class_targets.append(class_target)
    file_names.append(file_name)



for i in range(len(class_targets)):

    target = class_targets[i]
    pit_n = file_names[i] + '.jpg'
    # Example usage
    image_path = '/home/meow/cnfood241/food_code_cn241/herbs_heatmap_cnf241_val/' + pit_n
    output_dir = '/home/meow/cnfood241/food_code_cn241/herbs_heatmap_cnf241_val/repvit_heatmap'
    # Print the results
    print("Class targets:", class_targets)
    print("File names:", file_names)

    test_img = Image.open(image_path)
    test_img_data = np.asarray(test_img)

    transformed_img = transform(test_img)
    input_img = transform_normalize(transformed_img)
    input_img = input_img.unsqueeze(0)  # the model requires a dummy batch dimension

    # Identify the target layer for Grad-CAM
    target_layer = model.stages[3].blocks[1].channel_mixer.conv2  # Adjust this to the correct layer in your model

    layer_gradcam = LayerGradCam(model, target_layer)
    attributions_lgc = layer_gradcam.attribute(input_img, target=int(target))

    upsamp_attr_lgc = LayerAttribution.interpolate(attributions_lgc, input_img.shape[2:])
    print(upsamp_attr_lgc[0].shape)

    print(attributions_lgc.shape)
    print(upsamp_attr_lgc.shape)
    print(input_img.shape)

    # Debug: Check for zero scale factor
    attr_np = upsamp_attr_lgc[0].cpu().permute(1, 2, 0).detach().numpy()
    if np.max(attr_np) == np.min(attr_np):
        print("Warning: Attributions have zero scale factor. Normalization will fail.")


    integrated_gradients = IntegratedGradients(model)
    print("here")
    attributions_ig = integrated_gradients.attribute(input_img, target=21, n_steps=200)


    cdict = get_cdict()
    cmap = matplotlib.colors.LinearSegmentedColormap("jet_revice", cdict)

    fig, ax =plt.subplots()
    _ = viz.visualize_image_attr(
                    np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                    np.transpose(transformed_img.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                    method='blended_heat_map',
                    cmap=cmap,
                    show_colorbar=False,
                    sign='positive',
                    outlier_perc=1,
                    plt_fig_axis=(fig, ax)
                    # title='Blended Heat Map'
    )

    # Save the figure
    output_path = os.path.join(output_dir, 'heat_repvit_' + str(target) + pit_n)
    fig.savefig(output_path)