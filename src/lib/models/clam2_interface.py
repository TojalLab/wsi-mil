from .clam_interface import CLAMInterface
from lib.etc import create_imagenet_feature_extractor
import torch
import torchvision

class CLAM2Interface(CLAMInterface):
    def __init__(self, cfg):
        super().__init__(cfg)

        #self.fe = create_imagenet_feature_extractor('resnet50', 'avgpool')
        #m = torchvision.models.resnet50(pretrained=True)
        m = torchvision.models.resnet34(pretrained=True)
        #m = torchvision.models.resnet50(pretrained=True)
        #m = torchvision.models.convnext_tiny(pretrained=True)
        #num_filters = self.backbone.fc.in_features
        layers = list(m.children())[:-1]
        self.backbone = torch.nn.Sequential(*layers, torch.nn.Flatten())
        #self.backbone = self.backbone.eval()
        self.backbone = self.backbone.to(self.device)

    def training_step(self, batch, batch_idx):
        x, label = batch
        x = x.flatten(0,1) # combine batch_size x N
        #with torch.no_grad():
        x2 = self.backbone(x).unsqueeze(0)
        return super().training_step((x2, label), batch_idx)

    def validation_step(self, batch, batch_idx):
        x, label = batch
        x = x.flatten(0,1) # combine batch_size x N
        with torch.no_grad():
            x2 = self.backbone(x).unsqueeze(0)
        return super().validation_step((x2, label), batch_idx)

    def test_step(self, batch, batch_idx):
        x, label = batch
        x = x.flatten(0,1) # combine batch_size x N
        with torch.no_grad():
            x2 = self.backbone(x).unsqueeze(0)
        return super().test_step((x2, label), batch_idx)

    def predict_step(self, batch, batch_idx):
        x, label = batch
        x = x.flatten(0,1) # combine batch_size x N
        with torch.no_grad():
            x2 = self.backbone(x).unsqueeze(0)
        return super().test_step((x2, label), batch_idx)

    def configure_optimizers(self):
        opt = super().configure_optimizers()
        opt.add_param_group({'params': self.backbone.parameters(), 'lr': 1e-6})
        return opt
