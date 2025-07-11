Name: KHAN MD SABUJ
Email:khansobuz203@Gemail.com
================================================================================================================




🧠 Continual Learning for Weakly Supervised Video Anomaly Detection

This project implements a continual learning framework for Weakly Supervised Video Anomaly Detection (WSVAD) using deep learning techniques. It supports experiments on UCF-Crime and ShanghaiTech datasets.

---

✅ Requirements

Install the required Python libraries:

```bash
pip install torch numpy matplotlib seaborn scikit-learn
etc...................................................
```

---

### 📁 Dataset Structure

Place your UCF-Crime dataset inside a folder named `UCF-Crime` like this:

```
UCF-Crime/
├── all_rgbs/
├── all_flows/
├── train_normal.txt
├── train_anomaly.txt
├── test_normalv2.txt
├── test_anomalyv2.txt
```
ShanghaiTech
./data/
  └── shanghai/
      ├── shanghai-i3d-train-10crop.list
      ├── shanghai-i3d-test-10crop.list
      └── extracted_features/   (all feature files here)

Ensure `.npy` files for RGB and optical flow are stored under `all_rgbs/` and `all_flows/`.

---

----------
The link of datasets:
url: https://drive.google.com/file/d/1bOpDDDa0ZTyV0q9-V8HFEXCYPFeHlDcr/view?usp=sharing
------------

 🚀 How to Run

Use the following scripts for different experiments:

 🔹 UCF-Crime

| Script              | Purpose                         |
| ------------------- | ------------------------------- |
| `UCF_avg_auc.py`    | Calculate average AUC           |
| `UCF_EXP1.py`       | Run overall accuracy experiment |
| `UCF_buffer_exp.py` | Buffer-based continual learning |

 🔹 ShanghaiTech

| Script                | Purpose                              |
| --------------------- | ------------------------------------ |
| `Shanghai_exp1.py`    | Standard ShanghaiTech experiment     |
| `Shanghai_exp1_CL.py` | ShanghaiTech with continual learning |

---

💾 Pretrained Model

A checkpoint file is available:

```text
ckpt_auc_0.8585.pth
```

This checkpoint contains weights with AUC 0.8585 and can be loaded directly.

---

⚙️ System Requirements

 GPU with 16 GB memory recommended
 Python 3.8+
 PyTorch 1.9+

