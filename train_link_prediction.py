import hydra
from hydra.utils import instantiate
import numpy as np
import torch


class Workspace:

    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.feature == 'deeapwalk': self.cfg.n_input = 128

        if cfg.feature == 'deepwalk':
            self.x = torch.from_numpy(np.load(cfg.x_path)).float().to(cfg.device)
        else:
            self.x = torch.eye(cfg.n_input).float().to(cfg.device)
        self.train_edge_index = torch.from_numpy(np.load(cfg.train_edge_index_path)).long().to(cfg.device)
        self.test_edge_index = torch.from_numpy(np.load(cfg.test_edge_index_path)).long().to(cfg.device)
        self.positive = torch.from_numpy(np.load(cfg.positive_path)).long().to(cfg.device)
        self.negative = torch.from_numpy(np.load(cfg.negative_path)).long().to(cfg.device)
        self.train_mask = torch.from_numpy(np.load(cfg.train_mask_path)).bool().to(cfg.device)
        self.val_mask = torch.from_numpy(np.load(cfg.val_mask_path)).bool().to(cfg.device)
        self.test_mask = torch.from_numpy(np.load(cfg.test_mask_path)).bool().to(cfg.device)

        self.epsilon = 1e-15

        self.model = instantiate(cfg.model_cfg).to(cfg.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def predict(self, batch, edge_index):
        out = self.model(self.x, edge_index)
        n1 = out[batch[:, 0]]
        n2 = out[batch[:, 1]]
        return torch.sigmoid(torch.sum(n1 * n2, dim=1))
    
    def train(self):
        cnt = 0
        max_acc = 0
        for epoch in range(self.cfg.epochs):
            # train
            self.model.train()

            positive = self.predict(self.positive[self.train_mask], self.train_edge_index)
            negative = self.predict(self.negative[self.train_mask], self.train_edge_index)
            train_loss = -torch.log(positive + self.epsilon).mean() - torch.log(1 - negative + self.epsilon).mean()
            train_acc = ((positive >= 0.5).sum() + (negative < 0.5).sum()) / (positive.size(0) + negative.size(0))

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

            # validation
            self.model.eval()

            positive = self.predict(self.positive[self.val_mask], self.train_edge_index)
            negative = self.predict(self.negative[self.val_mask], self.train_edge_index)
            val_loss = -torch.log(positive + self.epsilon).mean() - torch.log(1 - negative + self.epsilon).mean()
            val_acc = ((positive >= 0.5).sum() + (negative < 0.5).sum()) / (positive.size(0) + negative.size(0))

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

        positive = self.predict(self.positive[self.test_mask], self.test_edge_index)
        negative = self.predict(self.negative[self.test_mask], self.test_edge_index)
        test_loss = -torch.log(positive + self.epsilon).mean() - torch.log(1 - negative + self.epsilon).mean()
        test_acc = ((positive >= 0.5).sum() + (negative < 0.5).sum()) / (positive.size(0) + negative.size(0))

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
