import torch
import torch.nn.functional as F

#https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
# additive MIL
# https://arxiv.org/pdf/2206.01794.pdf
class ADDMIL(torch.nn.Module):
    def __init__(self, num_features=800, num_classes=2, dropout=0.25):
        super(ADDMIL, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        L1 = num_features // 2
        L2 = L1 // 2

        self.attnT = torch.nn.Sequential(
            torch.nn.Linear(num_features, L2),
            torch.nn.Tanh(),
            torch.nn.Linear(L2, 1),
            torch.nn.Softmax(dim=0)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_features, L2),
            torch.nn.ReLU(),
            torch.nn.Linear(L2, num_classes),
            torch.nn.Tanh()
        )

    def forward(self, h):
        h = h.squeeze(0) # NxL

        At = self.attnT(h) # Nx1
        #At = F.softmax(At,dim=0)

        M = At * h # NxL
        patch_preds = self.classifier(M) # NxC

        Y_prob = patch_preds.sum(dim=0) # 1xC

        return { "Y_prob": Y_prob, "Y_hat": Y_prob.argmax(dim=0), "patch_preds": patch_preds }

            # C1, C2, C3
# loss = cross_entropy(Y_prob , [1,0,0] )
