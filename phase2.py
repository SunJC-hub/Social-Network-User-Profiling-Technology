import os
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW

# ===============================
# 配置部分
# ===============================
MODEL_NAME = "hfl/chinese-roberta-wwm-ext-large"
NUM_LABELS = 8  # 7种情绪 + none
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5
MAX_LENGTH = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# 情绪标签映射
# ===============================
EMOTION_LABELS = {
    "none": 0,
    "happiness": 1,
    "like": 2,
    "anger": 3,
    "sadness": 4,
    "fear": 5,
    "disgust": 6,
    "surprise": 7,
}

# ===============================
# 数据解析函数
# ===============================
def parse_weibo_xml_multiclass(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    texts = []
    labels = []
    count = 0  # 计数器

    for weibo in root.findall("weibo"):
        for sentence in weibo.findall("sentence"):
            if sentence.attrib.get("emotion_tag", "N") == "Y":
                text = sentence.text.strip()
                emo_type = sentence.attrib.get("emotion-1-type", "none")
                label = EMOTION_LABELS.get(emo_type, 0)
                texts.append(text)
                labels.append(label)
                count += 1
                if count >= 5:  # 达到10条即退出
                    return texts, labels

    return texts, labels


# ===============================
# 自定义数据集
# ===============================
class WeiboDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

# ===============================
# 加载LoRA微调的RoBERTa模型
# ===============================
def load_roberta_with_lora():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(model, lora_config)

    return model

# ===============================
# 训练函数
# ===============================
def train(model, dataloader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# ===============================
# 主函数
# ===============================
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 👇 使用你的真实XML路径
    xml_path = "data.xml"
    texts, labels = parse_weibo_xml_multiclass(xml_path)

    dataset = WeiboDataset(texts, labels, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = load_roberta_with_lora().to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LR)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                                 num_warmup_steps=0,
                                 num_training_steps=EPOCHS * len(dataloader))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        avg_loss = train(model, dataloader, optimizer, lr_scheduler, criterion)
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}")

    # 模型保存
    torch.save(model.state_dict(), "weibo_emotion_roberta_lora.pth")  # ✅ 仅保存模型权重

    print("✅ 模型已保存为 weibo_emotion_roberta_lora.pth")
