import torch
import torchvision.models as models

# Saving Model
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')  # stored in an internal state dictionary aka state_dict

# Loading Model
model = models.vgg16()  # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()  # Sets the dropout and batch normalisation layers to evaluation mode

# Saves and loading model with shapes
torch.save(model, 'model.pth')
model = torch.load('model.pth')
# model.state_dict() is not required
