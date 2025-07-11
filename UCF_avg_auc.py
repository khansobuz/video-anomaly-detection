import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn import metrics
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import Normal_Loader, Anomaly_Loader
import torch.nn.functional as F
from collections import deque
import sys

# UCF-Crime subclasses
class_names = [
    'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion',
    'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting',
    'Stealing', 'Vandalism'
]
class_to_id = {name: i+1 for i, name in enumerate(class_names)}

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

def contrastive_loss(proj, batch_size, device, temperature=0.15):
    proj = F.normalize(proj, dim=1)
    mid = proj.size(0) // 2
    pos_pairs = proj[:mid]
    neg_pairs = proj[mid:]
    logits = torch.matmul(pos_pairs, neg_pairs.T) / temperature
    labels = torch.arange(min(mid, neg_pairs.size(0))).to(device)
    return F.cross_entropy(logits, labels)

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

def focal_loss(y_pred, batch_size, device, gamma=2.5, alpha=0.25):
    y_true = torch.cat([torch.ones(batch_size * 32), torch.zeros(batch_size * 32)]).to(device)
    y_pred = 0.8 * torch.sigmoid(y_pred) + 0.2 * torch.sigmoid(y_pred).mean() + 1e-7
    bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()

def anomaly_score_loss(anomaly_score, batch_size, device, margin=1.0):
    mid = anomaly_score.size(0) // 2
    anomaly_scores = anomaly_score[:mid]
    normal_scores = anomaly_score[mid:]
    diff = torch.mean(anomaly_scores) - torch.mean(normal_scores)
    return F.relu(margin - diff)

def train(epoch, model, normal_train_loader, anomaly_train_loader, optimizer, criterion, device, replay_buffer=None):
    print(f'\nEpoch: {epoch}')
    model.train()
    train_loss = 0
    if epoch < 3:
        lr_scale = (epoch + 1) / 3.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.000510 + (0.001500 - 0.000510) * lr_scale
    print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    for batch_idx, (normal_inputs, anomaly_inputs) in enumerate(zip(normal_train_loader, anomaly_train_loader)):
        noise = torch.randn_like(normal_inputs) * 0.07
        inputs = torch.cat([anomaly_inputs + noise, normal_inputs + noise], dim=1)
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
        as_loss = anomaly_score_loss(anomaly_score, batch_size, device)
        score_penalty = 0.00005 * anomaly_score.pow(2).mean()
        loss = mil_loss + 1.0 * cl_loss + 1.0 * fl_loss + 1.0 * as_loss + score_penalty
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
        if replay_buffer is not None:
            replay_buffer.append((inputs[:batch_size * 64].detach().cpu(), anomaly_score[:batch_size * 64].detach().cpu()))
            if len(replay_buffer) > 1000:
                replay_buffer.popleft()
        del inputs, outputs, anomaly_score, proj
        torch.cuda.empty_cache()
    avg_loss = train_loss / len(normal_train_loader)
    print(f'Epoch: {epoch}, Loss: {avg_loss:.4f}, AUC: -')
    return avg_loss

def test_abnormal(epoch, model, anomaly_test_loader, normal_test_loader, device, roc_data):
    print(f'Starting testing for Epoch: {epoch}')
    model.eval()
    global best_auc
    auc = 0
    class_scores = {i: [] for i in range(1, 14)}
    class_labels = {i: [] for i in range(1, 14)}
    normal_scores = []
    normal_labels = []
    class_fpr_tpr = {i: None for i in range(1, 14)}
    with torch.no_grad():
        for i, (data, data2) in enumerate(zip(anomaly_test_loader, normal_test_loader)):
            try:
                inputs, gts, frames, class_id = data
                class_id = int(class_id.item()) if isinstance(class_id, torch.Tensor) else int(class_id)
            except ValueError:
                inputs, gts, frames = data
                file_path = anomaly_test_loader.dataset.data_list[i].split('|')[0]
                class_id = 1
                for class_name in class_names:
                    if class_name in file_path:
                        class_id = class_to_id[class_name]
                        break
            inputs = inputs.view(-1, inputs.size(-1)).to(device)
            _, score, _ = model(inputs)
            score = torch.sigmoid(score).cpu().numpy()
            score_list = np.zeros(frames[0])
            step = np.round(np.linspace(0, frames[0] // 16, 33))
            for j in range(32):
                score_list[int(step[j]) * 16:int(step[j + 1]) * 16] = score[j % len(score)]
            gt_list = np.zeros(frames[0])
            for k in range(len(gts) // 2):
                s = max(0, gts[k * 2] - 1)
                e = min(gts[k * 2 + 1], frames[0])
                gt_list[s:e] = 1
            if 1 <= class_id <= 13:
                class_scores[class_id].extend(score_list)
                class_labels[class_id].extend(gt_list)
            inputs2, gts2, frames2 = data2
            inputs2 = inputs2.view(-1, inputs2.size(-1)).to(device)
            _, score2, _ = model(inputs2)
            score2 = torch.sigmoid(score2).cpu().numpy()
            score_list2 = np.zeros(frames2[0])
            step2 = np.round(np.linspace(0, frames[0] // 16, 33))
            for kk in range(32):
                score_list2[int(step2[kk]) * 16:int(step2[kk + 1]) * 16] = score2[kk % len(score2)]
            gt_list2 = np.zeros(frames2[0])
            normal_scores.extend(score_list2)
            normal_labels.extend(gt_list2)
            score_list3 = np.concatenate((score_list, score_list2), axis=0)
            gt_list3 = np.concatenate((gt_list, gt_list2), axis=0)
            fpr, tpr, _ = metrics.roc_curve(gt_list3, score_list3, pos_label=1)
            auc += metrics.auc(fpr, tpr)
            del inputs, inputs2, score, score2
            torch.cuda.empty_cache()
        avg_auc = auc / 140
        print(f'Epoch: {epoch}, Loss: -, AUC: {avg_auc:.4f}, Best AUC: {max(best_auc, avg_auc):.4f}')
        class_aucs = []
        for class_id in range(1, 14):
            if class_scores[class_id]:
                scores = np.concatenate((class_scores[class_id], normal_scores))
                labels = np.concatenate((class_labels[class_id], normal_labels))
                if len(np.unique(labels)) > 1:
                    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
                    class_auc = metrics.auc(fpr, tpr)
                    class_aucs.append(class_auc)
                    class_fpr_tpr[class_id] = (fpr, tpr)
                    print(f'{class_names[class_id-1]}: AUC = {class_auc:.4f}')
                else:
                    print(f'{class_names[class_id-1]}: AUC = N/A (insufficient data)')
            else:
                print(f'{class_names[class_id-1]}: AUC = N/A (no data)')
        if class_aucs:
            average_auc = sum(class_aucs) / len(class_aucs)
            print(f'Average AUC across classes: {average_auc:.4f}')
            roc_data[epoch] = {
                'average_auc': average_auc,
                'class_fpr_tpr': class_fpr_tpr,
                'normal_scores': normal_scores,
                'normal_labels': normal_labels
            }
        else:
            print('Average AUC across classes: N/A (no valid AUCs)')
        state = {'net': model.state_dict()}
        if avg_auc > best_auc:
            print('Saving..')
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_auc = avg_auc
            print(f'New best AUC: {best_auc:.4f}')
        if avg_auc >= 0.80:
            print('Saving high AUC checkpoint..')
            torch.save(state, f'./checkpoint/ckpt_auc_{avg_auc:.4f}.pth')
    return avg_auc

def plot_roc_curves(roc_data, best_epoch):
    if best_epoch not in roc_data:
        print(f"No ROC data for best epoch {best_epoch}")
        return
    sns.set_style("whitegrid")
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'font.family': 'Arial'
    })
    os.makedirs('roc_curves', exist_ok=True)
    data = roc_data[best_epoch]
    class_fpr_tpr = data['class_fpr_tpr']
    all_tprs = []
    base_fpr = np.linspace(0, 1, 101)
    for class_id in range(1, 14):
        os.makedirs(f'roc_curves/{class_names[class_id-1]}', exist_ok=True)
        if class_fpr_tpr[class_id] is not None:
            fpr, tpr = class_fpr_tpr[class_id]
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, color='blue', linewidth=2, label='ROC Curve')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {class_names[class_id-1]}')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.grid(True)
            plt.savefig(f'roc_curves/{class_names[class_id-1]}/roc_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
            tpr_interp = np.interp(base_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            all_tprs.append(tpr_interp)
    if all_tprs:
        mean_tpr = np.mean(all_tprs, axis=0)
        mean_tpr[-1] = 1.0
        plt.figure(figsize=(6, 6))
        plt.plot(base_fpr, mean_tpr, color='blue', linewidth=2, label='Average ROC Curve')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Average ROC Curve Across Classes')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(True)
        os.makedirs('roc_curves/Average', exist_ok=True)
        plt.savefig('roc_curves/Average/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    torch.manual_seed(42)
    modality = 'TWO'
    input_dim = 2048
    memory_size = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_auc = 0
    patience = 65
    patience_counter = 0
    roc_data = {}
    normal_train_dataset = Normal_Loader(is_train=1, modality=modality)
    normal_test_dataset = Normal_Loader(is_train=0, modality=modality)
    anomaly_train_dataset = Anomaly_Loader(is_train=1, modality=modality)
    anomaly_test_dataset = Anomaly_Loader(is_train=0, modality=modality)
    normal_train_loader = DataLoader(normal_train_dataset, batch_size=20, shuffle=True)
    normal_test_loader = DataLoader(normal_test_dataset, batch_size=1, shuffle=True)
    anomaly_train_loader = DataLoader(anomaly_train_dataset, batch_size=20, shuffle=True)
    anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1, shuffle=True)
    model = VAD_Model(input_size=input_dim, memory_size=memory_size).to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=0.0003, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 4], gamma=0.5)
    criterion = MIL
    replay_buffer = deque(maxlen=1000)
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
    best_avg_auc = 0
    best_epoch = 0
    for epoch in range(20):
        train_loss = train(epoch, model, normal_train_loader, anomaly_train_loader, optimizer, criterion, device, replay_buffer)
        auc = test_abnormal(epoch, model, anomaly_test_loader, normal_test_loader, device, roc_data)
        if epoch in roc_data and roc_data[epoch]['average_auc'] > best_avg_auc:
            best_avg_auc = roc_data[epoch]['average_auc']
            best_epoch = epoch
        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs.')
                break
        scheduler.step()
    print(f'Final best AUC: {best_auc:.4f}')
    print(f'Best epoch for average AUC: {best_epoch}, Average AUC: {best_avg_auc:.4f}')
    plot_roc_curves(roc_data, best_epoch)