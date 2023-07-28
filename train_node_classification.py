import hydra
from hydra.utils import instantiate
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torchmetrics import Accuracy


class Workspace:

    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.feature == 'deeapwalk': self.cfg.n_input = 128
        
        dataset = Planetoid(root='data', name=cfg.dataset)[0]
        if cfg.feature == 'deepwalk':
            self.x = torch.from_numpy(np.load(cfg.x_path)).float().to(cfg.device)
        else:
            self.x = torch.eye(cfg.n_input).float().to(cfg.device)
        self.y = dataset.y.to(cfg.device)
        self.edge_index = dataset.edge_index.to(cfg.device)
        self.train_mask = dataset.train_mask.to(cfg.device)
        self.val_mask = dataset.val_mask.to(cfg.device)
        self.test_mask = dataset.test_mask.to(cfg.device)

        self.model = instantiate(cfg.model_cfg).to(cfg.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.accuracy = Accuracy(task='multiclass', num_classes=cfg.n_output, average='micro').to(cfg.device)
    
    def train(self):
        cnt = 0
        max_acc = 0
        for epoch in range(self.cfg.epochs):
            # train
            self.model.train()

            pred = self.model(self.x, self.edge_index)
            train_loss = F.nll_loss(pred[self.train_mask], self.y[self.train_mask])
            train_acc = self.accuracy(pred[self.train_mask], self.y[self.train_mask])

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            # validation
            self.model.eval()

            pred = self.model(self.x, self.edge_index)
            val_loss = F.nll_loss(pred[self.val_mask], self.y[self.val_mask])
            val_acc = self.accuracy(pred[self.val_mask], self.y[self.val_mask])

            print(f'Epoch: {epoch + 1:04d} \t Train Loss: {train_loss:.4f} \t Train Acc: {train_acc:.4f} \t Val Loss: {val_loss:.4f} \t Val Acc: {val_acc:.4f}')

            if val_acc.item() > max_acc:
                cnt = 0
                max_acc = val_acc.item()
            else:
                cnt += 1
                if cnt == self.cfg.patience:
                    break
    
    def test(self):
        self.model.eval()

        pred = F.log_softmax(self.model(self.x, self.edge_index), dim=1)
        test_loss = F.nll_loss(pred[self.test_mask], self.y[self.test_mask])
        test_acc = self.accuracy(pred[self.test_mask], self.y[self.test_mask])

        print(f'Test Loss: {test_loss:.4f} \t Test Acc: {test_acc:.4f}')
    
    def run(self):
        self.train()
        self.test()


@hydra.main(version_base=None, config_path='.', config_name='config')
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()


if __name__ == '__main__':
    main()
