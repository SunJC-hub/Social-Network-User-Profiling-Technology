import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_scheduler
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ==============================
# 配置
# ==============================
MODEL_NAME = "hfl/chinese-roberta-wwm-ext-large"
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5
MAX_LENGTH = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
XML_PATH = "data.xml"  # 替换成你的XML文件路径

# ==============================
# 数据预处理函数（仅处理前100条）
# ==============================
def parse_xml_and_get_data(xml_file, max_samples=5):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    texts = []
    labels = []

    for weibo in root.findall("weibo"):
        if len(texts) >= max_samples:
            break

        emotion_type = weibo.attrib.get("emotion-type", "none")
        label = 0 if emotion_type == "none" else 1

        for sentence in weibo.findall("sentence"):
            text = sentence.text
            if text:
                texts.append(text.strip())
                labels.append(label)
                if len(texts) >= max_samples:
                    break

    return texts, labels

# ==============================
# 数据集类
# ==============================
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

# ==============================
# 模型定义
# ==============================
class SimpleClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super(SimpleClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 向量
        logits = self.classifier(cls_output)
        return logits

# ==============================
# 训练函数
# ==============================
def train(model, dataloader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# ==============================
# 主函数
# ==============================
if __name__ == "__main__":
    # 加载 tokenizer 和数据
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    texts, labels = parse_xml_and_get_data(XML_PATH, max_samples=5)

    # 创建数据集和 DataLoader
    dataset = WeiboDataset(texts, labels, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 创建模型、优化器、损失函数
    model = SimpleClassifier(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    num_training_steps = EPOCHS * len(dataloader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    criterion = nn.CrossEntropyLoss()

    # 训练
    for epoch in range(EPOCHS):
        avg_loss = train(model, dataloader, optimizer, scheduler, criterion)
        print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "phase1_sentiment_classifier.pth")  # ✅ 仅保存模型权重

    print("✅ 第一阶段模型已保存。")
