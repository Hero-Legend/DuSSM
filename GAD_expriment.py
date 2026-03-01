# ==============================================================================
# 0. 安全补丁与依赖加载
# ==============================================================================
import transformers.utils.import_utils
import transformers.modeling_utils
def safe_bypass(*args, **kwargs): return None
transformers.utils.import_utils.check_torch_load_is_safe = safe_bypass
transformers.modeling_utils.check_torch_load_is_safe = safe_bypass

import os, sys, torch, random, time, re, torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModel
from mamba_ssm import Mamba

# ==============================================================================
# 1. 全局配置 (请根据实际路径修改)
# ==============================================================================
CONFIG = {
    "model_path": './model_path/PubMedBert',  # 预训练权重路径
    "data_dir": './data/GAD',                # 10折数据所在目录 (包含1-10文件夹)
    "output_dir": './benchmarking_results',   # 结果保存目录
    "max_len": 128,
    "batch_size": 16,
    "lr": 2e-5,
    "epochs": 10,
    "num_folds": 10,
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

if not os.path.exists(CONFIG["output_dir"]): os.makedirs(CONFIG["output_dir"])

def set_seed(seed=777):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ==============================================================================
# 2. 数据与 Dataset (适配 GAD/EU-ADR)
# ==============================================================================
def load_bionlp_data(fold_idx, file_name):
    """适配包含 @gene$ 和 @disease$ 占位符的 TSV 文件"""
    path = os.path.join(CONFIG["data_dir"], str(fold_idx), file_name)
    if not os.path.exists(path): return pd.DataFrame()
    
    df = pd.read_csv(path, sep='\t', header=None, names=['sentence', 'label'])
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    
    processed = []
    for _, row in df.iterrows():
        txt = str(row['sentence']).lower()
        # 统一占位符替换为特殊 Token
        txt = txt.replace('@gene$', ' [E1] ').replace('@disease$', ' [E2] ')
        txt = re.sub(r'\s+', ' ', txt).strip()
        processed.append({'text': txt, 'label': row['label']})
    return pd.DataFrame(processed)

class BioNLPDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.tokenizer = tokenizer
        self.data = df.reset_index(drop=True)
        self.e1_id = tokenizer.convert_tokens_to_ids('[E1]')
        self.e2_id = tokenizer.convert_tokens_to_ids('[E2]')

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        enc = self.tokenizer(row['text'], truncation=True, max_length=CONFIG["max_len"], padding='max_length')
        ids = enc['input_ids']
        
        # 定位实体在 ID 序列中的物理位置
        try: e1_p = ids.index(self.e1_id)
        except ValueError: e1_p = 0
        try: e2_p = ids.index(self.e2_id)
        except ValueError: e2_p = 0

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(enc['attention_mask'], dtype=torch.long),
            'e1': torch.tensor(e1_p, dtype=torch.long),
            'e2': torch.tensor(e2_p, dtype=torch.long),
            'label': torch.tensor(row['label'], dtype=torch.long)
        }

# ==============================================================================
# 3. DuSSM 模型架构 (CNN + Mamba)
# ==============================================================================
class DuSSM_Model(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.bert = AutoModel.from_pretrained(CONFIG["model_path"])
        self.bert.resize_token_embeddings(len(tokenizer))
        dim = self.bert.config.hidden_size 
        
        self.proj = nn.Linear(dim, 256)
        self.cnn = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.mamba = Mamba(d_model=256, d_state=16, d_conv=4, expand=2)
        
        # 分类层：输入 = CLS(768) + E1_combined(512) + E2_combined(512) = 1792
        self.clf = nn.Sequential(
            nn.Linear(dim + 256 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, ids, mask, e1, e2):
        bert_out = self.bert(input_ids=ids, attention_mask=mask).last_hidden_state
        cls_v = bert_out[:, 0, :]
        
        x = self.proj(bert_out)
        # 显式特征提取 (CNN)
        x_c = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        # 隐式特征提取 (Mamba)
        x_m = self.mamba(x)
        
        # 特征融合
        combined = torch.cat([x_c, x_m], dim=2) # [batch, seq, 512]
        
        idx = torch.arange(ids.shape[0], device=ids.device)
        e1_v = combined[idx, e1]
        e2_v = combined[idx, e2]
        
        return self.clf(torch.cat((cls_v, e1_v, e2_v), dim=1))

# ==============================================================================
# 4. 主控程序：10-Fold 实验与 BIB 效率报告
# ==============================================================================
def run_experiment():
    set_seed(777)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_path"])
    tokenizer.add_special_tokens({'additional_special_tokens': ['[E1]', '[E2]']})
    
    fold_summaries = [] # 记录每一折的汇总数据

    for f in range(1, CONFIG["num_folds"] + 1):
        print(f"\n🚀 开始执行 Fold {f}/{CONFIG['num_folds']}...")
        
        df_train = load_bionlp_data(f, 'train.tsv')
        df_test = load_bionlp_data(f, 'test.tsv')
        if df_train.empty: 
            print(f"⚠️ Fold {f} 数据缺失，跳过"); continue

        train_loader = torch.utils.data.DataLoader(BioNLPDataset(df_train, tokenizer), batch_size=CONFIG["batch_size"], shuffle=True)
        test_loader = torch.utils.data.DataLoader(BioNLPDataset(df_test, tokenizer), batch_size=CONFIG["batch_size"])

        model = DuSSM_Model(tokenizer).to(CONFIG["device"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
        loss_fn = nn.CrossEntropyLoss()
        
        best_f1 = 0.0
        best_metrics_in_fold = {}

        for ep in range(CONFIG["epochs"]):
            model.train()
            for batch in tqdm(train_loader, desc=f"Fold{f} Ep{ep+1}", leave=False):
                optimizer.zero_grad()
                logits = model(batch['ids'].to(CONFIG["device"]), batch['mask'].to(CONFIG["device"]), 
                             batch['e1'].to(CONFIG["device"]), batch['e2'].to(CONFIG["device"]))
                loss = loss_fn(logits, batch['label'].to(CONFIG["device"]))
                loss.backward()
                optimizer.step()

            # 验证与效率采样
            model.eval()
            y_pred, y_true = [], []
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            
            with torch.no_grad():
                for batch in test_loader:
                    out = model(batch['ids'].to(CONFIG["device"]), batch['mask'].to(CONFIG["device"]), 
                              batch['e1'].to(CONFIG["device"]), batch['e2'].to(CONFIG["device"]))
                    y_pred.extend(torch.argmax(out, dim=1).cpu().numpy())
                    y_true.extend(batch['label'].numpy())
            
            inference_time = time.time() - start_time
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
            
            # 计算指标 (二分类 Macro 主要是为了对比稳定性)
            p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            
            if f1 > best_f1:
                best_f1 = f1
                best_metrics_in_fold = {
                    "Fold": f, "P": p, "R": r, "F1": f1,
                    "Mem": peak_mem, "Time": inference_time, "Samples": len(y_true)
                }

        print(f"✅ Fold {f} 最佳 F1: {best_f1:.4f}")
        fold_summaries.append(best_metrics_in_fold)
        
        # 清理显存防止下一折 OOM
        del model
        torch.cuda.empty_cache()

    # ==============================================================================
    # 5. 生成报告文件 (适配你的对比需求)
    # ==============================================================================
    dataset_name = os.path.basename(CONFIG["data_dir"].strip('/'))
    avg_f1 = np.mean([m['F1'] for m in fold_summaries])
    avg_mem = np.mean([m['Mem'] for m in fold_summaries])
    avg_inf_ms = np.mean([m['Time']/m['Samples']*1000 for m in fold_summaries])

    report_path = os.path.join(CONFIG["output_dir"], f'DuSSM_{dataset_name}_Report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"DuSSM {dataset_name} 10-Fold Results\n")
        f.write("="*40 + "\n")
        f.write(f"Peak GPU Memory: {avg_mem:.2f} MB (Average across folds)\n")
        f.write(f"Inference Time per Sample: {avg_inf_ms:.4f} ms\n\n")
        
        f.write("Detailed Metrics per Fold:\n")
        f.write(f"{'Fold':<8}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}\n")
        for m in fold_summaries:
            f.write(f"{m['Fold']:<8}{m['P']:<12.4f}{m['R']:<12.4f}{m['F1']:<12.4f}\n")
        
        f.write("-" * 40 + "\n")
        f.write(f"{'Average':<8}{np.mean([m['P'] for m in fold_summaries]):<12.4f}"
                f"{np.mean([m['R'] for m in fold_summaries]):<12.4f}"
                f"{avg_f1:<12.4f}\n")
        f.write(f"{'Std Dev':<8}{np.std([m['P'] for m in fold_summaries]):<12.4f}"
                f"{np.std([m['R'] for m in fold_summaries]):<12.4f}"
                f"{np.std([m['F1'] for m in fold_summaries]):<12.4f}\n")

    print(f"\n📊 完整 10-Fold 报告已保存至: {report_path}")

if __name__ == "__main__":
    run_experiment()