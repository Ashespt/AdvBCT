import torch
from models import build_bct_models

_,new_model=build_bct_models("base_model",debug=True)
x=torch.zeros((3,3,224,224))
new_model.train()
feature,y = new_model(x)
print(feature.shape)
print(y)