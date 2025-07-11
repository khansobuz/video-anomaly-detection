import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn import metrics
import numpy as np
import os
from dataset import Normal_Loader, Anomaly_Loader
import torch.nn.functional as F
from collections import deque

# Dual Memory Network
class DualMemoryNetwork(nn.Module):
    def __init__(self, input_size, memory_size=128):
        super(DualMemoryNetwork, self).__init__()
        self.primary_memory = nn.Parameter(torch.randn(memory_size, input_size) * 0.01)
        self.secondary_memory = nn.Parameter(torch.randn(memory_size, input_size) * 0.01)

    def forward(self, feature_vector):
        batch_size = feature_vector.size(0)
        primary_memory_expanded = self.primary_memory.unsqueeze(0).expand(batch_size, -1, -1)
        secondary_memory_expanded = self.secondary_memory.unsqueeze(0).expand(batch_size, -1, -1)
        primary_similarity = torch.norm(primary_memory_expanded - feature_vector.unsqueeze(1), dim=2)
        secondary_similarity = torch.norm(secondary_memory_expanded - feature_vector.unsqueeze(1), dim=2)
        return primary_similarity, secondary_similarity

# Custom MambaBlock
class MambaBlock(nn.Module):
    def __init__(self, input_size, hidden_size, d_state=32, d_conv=4, dropout=0.25):
        super(MambaBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.in_proj = nn.Linear(input_size, hidden_size * 2)
        self.out_proj = nn.Linear(hidden_size, input_size)
        self.ssm_proj = nn.Linear(hidden_size, d_state)
        self.A = nn.Parameter(torch.diag(torch.ones(d_state) * -1))
        self.B = nn.Parameter(torch.randn(d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(hidden_size))
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=d_conv, padding=d_conv-1, groups=hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.in_proj(x)
        x, gate = x.chunk(2, dim=-1)
        x_conv = x.transpose(1, 2)
        x_conv = self.conv(x_conv)[..., :seq_len]
        x_conv = x_conv.transpose(1, 2)
        x = self.act(x_conv)
        x_ssm = self.ssm_proj(x)
        h = torch.zeros(batch_size, self.d_state, device=x.device)
        ssm_out = []
        for t in range(seq_len):
            h = torch.einsum('ij,bj->bi', self.A, h) + (x_ssm[:, t, :] * self.B)
            y_t = torch.einsum('j,bj->b', self.C, h).unsqueeze(-1) * x[:, t, :] + self.D * x[:, t, :]
            ssm_out.append(y_t)
        ssm_out = torch.stack(ssm_out, dim=1)
        y = self.dropout(ssm_out * self.act(gate))
        y = self.out_proj(y)
        return y

# VAD_Model
class VAD_Model(nn.Module):
    def __init__(self, input_size=2048, num_classes=1, memory_size=128):
        super(VAD_Model, self).__init__()
        self.mamba = MambaBlock(input_size=input_size, hidden_size=input_size, d_state=32, d_conv=4, dropout=0.25)
        self.memory_network = DualMemoryNetwork(input_size=input_size, memory_size=memory_size)
        self.fc = nn.Linear(input_size, num_classes)
        self.projection = nn.Linear(input_size, 128)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        mamba_out = self.mamba(x)
        final_feature = mamba_out[:, -1, :]
        primary_similarity, secondary_similarity = self.memory_network(final_feature)
        min_similarity = torch.min(primary_similarity, secondary_similarity)
        anomaly_score = torch.min(min_similarity, dim=1)[0]
        out = self.fc(final_feature)
        proj = self.projection(final_feature)
        return out, anomaly_score, proj

# Contrastive Loss
def contrastive_loss(proj, batch_size, device, temperature=0.15):
    proj = F.normalize(proj, dim=1)
    mid = proj.size(0) // 2
    pos_pairs = proj[:mid]
    neg_pairs = proj[mid:]
    logits = torch.matmul(pos_pairs, neg_pairs.T) / temperature
    labels = torch.arange(min(mid, neg_pairs.size(0))).to(device)
    return F.cross_entropy(logits, labels)

# Enhanced MIL Loss with Increased Margin
def MIL(y_pred, batch_size, device, margin=2.0):
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
        loss += F.relu(margin - (y_anomaly_max - y_normal_max))
        sparsity += torch.sum(y_anomaly) * 0.00008
        smooth += torch.sum((y_pred[i, :frames_per_bag-1] - y_pred[i, 1:frames_per_bag]) ** 2) * 0.00008
    return (loss + sparsity + smooth) / batch_size

# Focal Loss
def focal_loss(y_pred, batch_size, device, gamma=2.5, alpha=0.25):
    y_true = torch.cat([torch.ones(batch_size * 32), torch.zeros(batch_size * 32)]).to(device)
    y_pred = 0.8 * torch.sigmoid(y_pred) + 0.2 * torch.sigmoid(y_pred).mean() + 1e-7
    bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()

# Anomaly Score Loss
def anomaly_score_loss(anomaly_score, batch_size, device, margin=1.0):
    mid = anomaly_score.size(0) // 2
    anomaly_scores = anomaly_score[:mid]
    normal_scores = anomaly_score[mid:]
    diff = torch.mean(anomaly_scores) - torch.mean(normal_scores)
    return F.relu(margin - diff)

# Training function
def train(epoch, model, normal_train_loader, anomaly_train_loader, optimizer, criterion, device, replay_buffer=None):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    if epoch < 15:
        lr_scale = (epoch + 1) / 15.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.000510 + (0.001500 - 0.000510) * lr_scale
    print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    try:
        for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader)):
            noise = torch.randn_like(normal_inputs) * 0.07
            inputs = torch.cat([anomaly_inputs + noise, normal_inputs + noise], dim=1)
            batch_size = inputs.shape[0]
            inputs = inputs.view(-1, inputs.size(-1)).to(device)
            expected_size = batch_size * (64 if replay_buffer is None else 96)
            if inputs.size(0) != expected_size and replay_buffer is None:
                continue
            if replay_buffer and len(replay_buffer) > 0:
                num_samples = min(len(replay_buffer), batch_size)
                replay_samples = np.random.choice(len(replay_buffer), num_samples, replace=False)
                replay_inputs = []
                for idx in replay_samples:
                    r_inputs = replay_buffer[idx][0].to(device)
                    if r_inputs.dim() == 3:
                        r_inputs = r_inputs.view(-1, r_inputs.size(-1))
                    start = np.random.randint(0, max(1, r_inputs.size(0) - 32))
                    chunk = r_inputs[start:start + 32]
                    if chunk.size(0) < 32:
                        chunk = F.pad(chunk, (0, 0, 0, 32 - chunk.size(0)))
                    replay_inputs.append(chunk)
                replay_inputs = torch.cat(replay_inputs, dim=0)
                if replay_inputs.size(0) < batch_size * 32:
                    replay_inputs = F.pad(replay_inputs, (0, 0, 0, batch_size * 32 - replay_inputs.size(0)))
                inputs = torch.cat([inputs, replay_inputs[:batch_size * 32]], dim=0)
                if inputs.size(0) > batch_size * 96:
                    inputs = inputs[:batch_size * 96]
            if inputs.size(0) != expected_size:
                continue
            outputs, anomaly_score, proj = model(inputs)
            if anomaly_score.size(0) != expected_size or torch.isnan(anomaly_score).any() or torch.isinf(anomaly_score).any():
                continue
            mil_loss = criterion(anomaly_score, batch_size, device)
            cl_loss = contrastive_loss(proj, batch_size, device)
            fl_loss = focal_loss(anomaly_score, batch_size, device)
            as_loss = anomaly_score_loss(anomaly_score, batch_size, device)
            score_penalty = 0.00005 * anomaly_score.pow(2).mean()
            loss = mil_loss + 1.0 * cl_loss + 1.0 * fl_loss + 1.0 * as_loss + score_penalty
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            train_loss += loss.item()
            if replay_buffer is not None:
                replay_buffer.append((inputs[:batch_size * 64].detach().cpu(), anomaly_score[:batch_size * 64].detach().cpu()))
                if len(replay_buffer) > 1000:
                    replay_buffer.popleft()
            torch.cuda.empty_cache()
        avg_loss = train_loss / len(normal_train_loader)
        print(f'Epoch: {epoch}, Loss: {avg_loss:.4f}, AUC: -')
        return avg_loss
    except Exception as e:
        raise

# Testing function
def test_abnormal(epoch, model, anomaly_test_loader, normal_test_loader, device, best_auc):
    model.eval()
    auc = 0
    try:
        with torch.no_grad():
            for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
                inputs, gts, frames = data
                inputs = inputs.view(-1, inputs.size(-1)).to(device)
                _, score, _ = model(inputs)
                if torch.isnan(score).any() or torch.isinf(score).any():
                    continue
                score = torch.sigmoid(score).cpu().detach().numpy()
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
                inputs2 = inputs2.view(-1, inputs.size(-1)).to(device)
                _, score2, _ = model(inputs2)
                if torch.isnan(score2).any() or torch.isinf(score2).any():
                    continue
                score2 = torch.sigmoid(score2).cpu().detach().numpy()
                score_list2 = np.zeros(frames2[0])
                step2 = np.round(np.linspace(0, frames[0] // 16, 33))
                for kk in range(32):
                    score_list2[int(step2[kk]) * 16:int(step2[kk + 1]) * 16] = score2[kk % len(score2)]
                gt_list2 = np.zeros(frames2[0])
                score_list3 = np.concatenate((score_list, score_list2), axis=0)
                gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)
                fpr, tpr, _ = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
                auc += metrics.auc(fpr, tpr)
                torch.cuda.empty_cache()
            avg_auc = auc / 140
            print(f'Epoch: {epoch}, Loss: -, AUC: {avg_auc:.4f}, Best AUC: {max(best_auc, avg_auc):.4f}')
            state = {'net': model.state_dict()}
            if avg_auc > best_auc:
                print('Saving..')
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, f'./checkpoint/ckpt_{"with" if replay_buffer else "without"}_buffer.pth')
                best_auc = avg_auc
                print(f'New best AUC: {best_auc:.4f}')
            if avg_auc >= 0.80:
                print('Saving high AUC checkpoint..')
                torch.save(state, f'./checkpoint/ckpt_auc_{avg_auc:.4f}_{"with" if replay_buffer else "without"}_buffer.pth')
            return avg_auc
    except Exception as e:
        raise

# Main script
if __name__ == '__main__':
    torch.manual_seed(42)
    modality = 'TWO'
    input_dim = 2048
    memory_size = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 100
    patience = 65
    try:
        normal_train_dataset = Normal_Loader(is_train=1, modality=modality)
        normal_test_dataset = Normal_Loader(is_train=0, modality=modality)
        anomaly_train_dataset = Anomaly_Loader(is_train=1, modality=modality)
        anomaly_test_dataset = Anomaly_Loader(is_train=0, modality=modality)
        normal_train_loader = DataLoader(normal_train_dataset, batch_size=36, shuffle=True)
        normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True)
        anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=36, shuffle=True)
        anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True)
        replay_buffer_none = None
        replay_buffer_with = deque(maxlen=1000)
        for replay_buffer, label in [(replay_buffer_none, 'Without Buffer'), (replay_buffer_with, 'With Buffer')]:
            print(f'\nRunning: {label}')
            best_auc = 0
            patience_counter = 0
            model = VAD_Model(input_size=input_dim, memory_size=memory_size).to(device)
            optimizer = optim.Adagrad(model.parameters(), lr=0.0003, weight_decay=0.005)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.5)
            criterion = MIL
            if os.path.exists('./checkpoint/ckpt_auc_0.8585.pth'):
                checkpoint = torch.load('./checkpoint/ckpt_auc_0.8585.pth', weights_only=False)
                state_dict = checkpoint['net']
                model_state_dict = model.state_dict()
                compatible_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
                model_state_dict.update(compatible_state_dict)
                model_state_dict.update({k: v for k, v in model_state_dict.items() if k not in compatible_state_dict})
                model.load_state_dict(model_state_dict)
                best_auc = 0.8585
                print("Loaded compatible checkpoint weights with AUC 0.8585")
            elif os.path.exists('./checkpoint/ckpt_auc_0.8569.pth'):
                checkpoint = torch.load('./checkpoint/ckpt_auc_0.8569.pth', weights_only=False)
                state_dict = checkpoint['net']
                model_state_dict = model.state_dict()
                compatible_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.shape == model_state_dict[k].shape}
                model_state_dict.update(compatible_state_dict)
                model_state_dict.update({k: v for k, v in model_state_dict.items() if k not in compatible_state_dict})
                model.load_state_dict(model_state_dict)
                best_auc = 0.8569
                print("Loaded compatible checkpoint weights with AUC 0.8569")
            for epoch in range(epochs):
                train_loss = train(epoch, model, normal_train_loader, anomaly_train_loader, optimizer, criterion, device, replay_buffer)
                auc = test_abnormal(epoch, model, anomaly_test_loader, normal_test_loader, device, best_auc)
                if auc > best_auc:
                    best_auc = auc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f'Early stopping triggered after {epoch+1} epochs.')
                        break
                scheduler.step()
                torch.cuda.empty_cache()
            print(f'Final best AUC for {label}: {best_auc:.4f}')
    except Exception as e:
        raise