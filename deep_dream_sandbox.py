import os
import copy

import torch
from torch.optim import SGD
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms as T
from PIL import Image

def visualizeOutput(output, layer, filter, resize):
    tensor = output.outputs[layer][0][filter]
    trans = T.ToPILImage()
    img = trans(tensor).convert("L")
    size = (img.size[0]*resize, img.size()[1]*resize)
    img = img.resize(size, Image.ANTIALIAS)
    img.show()

#record the output tensor of the forward pass and store it in a list
class SaveOutput:
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []

class DeepDream():
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0       
        
    def dream(self):
        pass          
               
device = torch.device('cpu')

model = models.resnet50(pretrained=True)
layerNamesList = []
for n, m in model.named_modules():
   layerNamesList.append(n)

cnn_layer = 34
filter_pos = 94 

dd = DeepDream(model, cnn_layer, filter_pos)
dd.dream()

save_output = SaveOutput()

hook_handles = []

#register the hook to each convolutional layer

for name, layer in model.named_modules():
    if name == 'layer4.2.conv3':
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            handle = layer.register_forward_hook(save_output)
            hook_handles.append(handle)

image = Image.open('./cat.jpeg')
transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
X = transform(image).unsqueeze(dim=0).to(device)

out = model(X)
visualizeOutput(save_output,0,0,4)
print(len(save_output.outputs))