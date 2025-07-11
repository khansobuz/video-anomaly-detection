import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn import metrics
import numpy as np
import os
from dataset1 import Normal_Loader, Anomaly_Loader
import torch.nn.functional as F
from collections import deque

# Dual Memory Network
class DualMemoryNetwork(nn.Module):
    def __init__(self, input_size, memory_size):
        super(DualMemoryNetwork, self).__init__()
        self.primary_memory = nn.Parameter(torch.zeros(memory_size, input_size))
        self.secondary_memory = nn.Parameter(torch.zeros(memory_size, input_size))

    def forward(self, feature_vector):
        batch_size = feature_vector.size(0)
        primary_memory_expanded = self.primary_memory.unsqueeze(0).expand(batch_size, -1, -1)
        secondary_memory_expanded = self.secondary_memory.unsqueeze(0).expand(batch_size, -1, -1)

        primary_similarity = torch.norm(primary_memory_expanded - feature_vector.unsqueeze(1), dim=2)
        secondary_similarity = torch.norm(secondary_memory_expanded - feature_vector.unsqueeze(1), dim=2)

        return primary_similarity, secondary_similarity

# Mamba Block (Lightweight)
class MambaBlock(nn.Module):
    def __init__(self, input_size, hidden_size, d_state=16, d_conv=4):  # Reduced d_state
        super(MambaBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.d_state = d_state
        self.d_conv = d_conv

        self.in_proj = nn.Linear(input_size, hidden_size * 2)
        self.out_proj = nn.Linear(hidden_size, input_size)
        self.ssm_proj = nn.Linear(hidden_size, d_state)
        self.A = nn.Parameter(torch.ones(d_state, d_state) * -1)
        self.B = nn.Parameter(torch.ones(d_state))
        self.C = nn.Parameter(torch.ones(d_state))
        self.D = nn.Parameter(torch.ones(hidden_size))
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=d_conv, padding=d_conv-1, groups=hidden_size)
        self.act = nn.SiLU()

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.in_proj(x)
        x, gate = x.chunk(2, dim=-1)
        x = x.transpose(1, 2)
        x = self.conv(x)[..., :seq_len]
        x = x.transpose(1, 2)
        x = self.act(x)
        x_ssm = self.ssm_proj(x)
        ssm_out = torch.zeros(batch_size, seq_len, self.hidden_size, device=x.device)
        h = torch.zeros(batch_size, self.d_state, device=x.device)
        for t in range(seq_len):
            h = torch.einsum('ij,bj->bi', self.A, h) + (x_ssm[:, t, :] * self.B)
            ssm_output = torch.einsum('j,bj->b', self.C, h)
            ssm_out[:, t, :] = ssm_output.unsqueeze(-1) * x[:, t, :]
        y = ssm_out * self.act(gate)
        y = self.out_proj(y)
        return y

# VAD_Model with LSTM and Lightweight Mamba
class VAD_Model(nn.Module):
    def __init__(self, input_size=2048, num_classes=1, memory_size=256):
        super(VAD_Model, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=input_size, num_layers=2, batch_first=True)
        self.mamba = MambaBlock(input_size=input_size, hidden_size=input_size, d_state=16)
        self.memory_network = DualMemoryNetwork(input_size=input_size, memory_size=memory_size)
        self.fc = nn.Linear(input_size, num_classes)
        self.projection = nn.Linear(input_size, 128)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, input_size]
        mamba_out = self.mamba(lstm_out)  # Refine LSTM output
        combined_out = lstm_out + 0.2 * mamba_out  # Lower Mamba’s weight
        final_feature = combined_out[:, -1, :]
        primary_similarity, secondary_similarity = self.memory_network(final_feature)
        min_similarity = torch.min(primary_similarity, secondary_similarity)
        anomaly_score = torch.min(min_similarity, dim=1)[0]
        out = self.fc(final_feature)
        proj = self.projection(final_feature)
        return out, anomaly_score, proj

# Contrastive Loss (InfoNCE)
def contrastive_loss(proj, batch_size, device, temperature=0.5):
    proj = F.normalize(proj, dim=1)
    mid = proj.size(0) // 2
    pos_pairs = proj[:mid]
    neg_pairs = proj[mid:]
    logits = torch.matmul(pos_pairs, neg_pairs.T) / temperature
    labels = torch.arange(min(mid, neg_pairs.size(0))).to(device)
    return F.cross_entropy(logits, labels)

# MIL Loss
def MIL(y_pred, batch_size, device):
    loss = torch.tensor(0., device=device)
    sparsity = torch.tensor(0., device=device)
    smooth = torch.tensor(0., device=device)
    frames_per_bag = y_pred.size(0) // batch_size
    y_pred = y_pred.view(batch_size, frames_per_bag)
    for i in range(batch_size):
        mid_point = frames_per_bag // 2
        anomaly_index = torch.randperm(mid_point).to(device)
        normal_index = torch.randperm(mid_point).to(device)
        y_anomaly = y_pred[i, :mid_point][anomaly_index]
        y_normal = y_pred[i, mid_point:][normal_index]
        y_anomaly_max = torch.max(y_anomaly)
        y_normal_max = torch.max(y_normal)
        loss += F.relu(1. - y_anomaly_max + y_normal_max)
        sparsity += torch.sum(y_anomaly) * 0.00008
        smooth += torch.sum((y_pred[i, :frames_per_bag-1] - y_pred[i, 1:frames_per_bag]) ** 2) * 0.00008
    return (loss + sparsity + smooth) / batch_size

# Focal Loss
def focal_loss(y_pred, batch_size, device, gamma=2.0, alpha=0.25):
    y_true = torch.cat([torch.ones(batch_size * 32), torch.zeros(batch_size * 32)]).to(device)
    bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()

# Training function
def train(epoch, model, normal_train_loader, anomaly_train_loader, optimizer, criterion, device, replay_buffer=None):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader)):
        inputs = torch.cat([anomaly_inputs, normal_inputs], dim=1)
        batch_size = inputs.shape[0]
        inputs = inputs.view(-1, inputs.size(-1)).to(device)
        if replay_buffer and len(replay_buffer) > 0:
            num_samples = min(len(replay_buffer), batch_size)
            replay_samples = np.random.choice(len(replay_buffer), num_samples, replace=False)
            replay_inputs = []
            for idx in replay_samples:
                r_inputs = replay_buffer[idx][0]
                start = np.random.randint(0, max(1, r_inputs.size(0) - 32))
                chunk = r_inputs[start:start + 32].to(device)
                if chunk.size(0) < 32:
                    chunk = F.pad(chunk, (0, 0, 0, 32 - chunk.size(0)))
                replay_inputs.append(chunk)
            replay_inputs = torch.cat(replay_inputs, dim=0)
            if replay_inputs.size(0) < batch_size * 32:
                replay_inputs = F.pad(replay_inputs, (0, 0, 0, batch_size * 32 - replay_inputs.size(0)))
            inputs = torch.cat([inputs, replay_inputs[:batch_size * 32]], dim=0)
            if inputs.size(0) > batch_size * 64:
                inputs = inputs[:batch_size * 64]
        outputs, anomaly_score, proj = model(inputs)
        mil_loss = criterion(anomaly_score, batch_size, device)
        cl_loss = contrastive_loss(proj, batch_size, device)
        fl_loss = focal_loss(anomaly_score, batch_size, device)
        loss = mil_loss + 0.5 * cl_loss + 0.1 * fl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if replay_buffer is not None:
            replay_buffer.append((inputs[:batch_size * 64].detach().cpu(), anomaly_score[:batch_size * 64].detach().cpu()))
            if len(replay_buffer) > 1000:
                replay_buffer.popleft()
    print('Loss =', train_loss / len(normal_train_loader))

# Testing function
def test_abnormal(epoch, model, anomaly_test_loader, normal_test_loader, device):
    model.eval()
    global best_auc
    auc = 0
    with torch.no_grad():
        for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
            inputs, gts, frames = data
            inputs = inputs.view(-1, inputs.size(-1)).to(device)
            _, score, _ = model(inputs)
            score = score.cpu().detach().numpy()
            score_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, frames[0] // 16, 33))
            for j in range(32):
                score_list[int(step[j]) * 16:int(step[j + 1]) * 16] = score[j % len(score)]
            gt_list = np.zeros(frames[0])
            for k in range(len(gts) // 2):
                s = max(0, gts[k * 2] - 1)
                e = min(gts[k * 2 + 1], frames[0])
                gt_list[s:e] = 1
            inputs2, gts2, frames2 = data2
            inputs2 = inputs2.view(-1, inputs2.size(-1)).to(device)
            _, score2, _ = model(inputs2)
            score2 = score2.cpu().detach().numpy()
            score_list2 = np.zeros(frames2[0])
            step2 = np.round(np.linspace(0, frames2[0] // 16, 33))
            for kk in range(32):
                score_list2[int(step2[kk]) * 16:int(step2[kk + 1]) * 16] = score2[kk % len(score2)]
            gt_list2 = np.zeros(frames2[0])
            score_list3 = np.concatenate((score_list, score_list2), axis=0)
            gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)
            fpr, tpr, _ = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
            auc += metrics.auc(fpr, tpr)
        avg_auc = auc / 140
        print('AUC =', avg_auc)
        if best_auc < avg_auc:
            print('Saving..')
            state = {'net': model.state_dict()}
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_auc = avg_auc
            print(f'New best AUC: {best_auc:.4f}')

# Main script
if __name__ == '__main__':
    lr = 0.001
    weight_decay = 0.001
    modality = 'TWO'
    input_dim = 2048
    memory_size = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_auc = 0

    normal_train_dataset = Normal_Loader(is_train=1, modality=modality)
    normal_test_dataset = Normal_Loader(is_train=0, modality=modality)
    anomaly_train_dataset = Anomaly_Loader(is_train=1, modality=modality)
    anomaly_test_dataset = Anomaly_Loader(is_train=0, modality=modality)

    normal_train_loader = DataLoader(normal_train_dataset, batch_size=30, shuffle=True)
    normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True)
    anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=30, shuffle=True)
    anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True)

    model = VAD_Model(input_size=input_dim, memory_size=memory_size).to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)  # Back to Adagrad
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50])
    criterion = MIL
    replay_buffer = deque(maxlen=1000)

    # Run for 15 epochs
    for epoch in range(15):
        train(epoch, model, normal_train_loader, anomaly_train_loader, optimizer, criterion, device, replay_buffer)
        test_abnormal(epoch, model, anomaly_test_loader, normal_test_loader, device)
        scheduler.step()