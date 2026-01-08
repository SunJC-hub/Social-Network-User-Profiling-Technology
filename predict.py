import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= 第一阶段模型结构：RoBERTa + 全连接分类器 =========
class Phase1FullModel(torch.nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # 取 [CLS] 向量
        logits = self.classifier(cls_output)
        return logits

# ========= 加载第一阶段模型 =========
def load_phase1_model(path, model_name="hfl/chinese-roberta-wwm-ext-large"):
    model = Phase1FullModel(model_name)
    state_dict = torch.load(path, map_location=DEVICE)  # 直接是 state_dict，不用["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model



# ========= 加载第二阶段模型（LoRA） =========
def load_phase2_model(path, model_name="hfl/chinese-roberta-wwm-ext-large", num_labels=8):
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(base_model, lora_config)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# ========= 预测函数 =========
def predict(text, tokenizer, phase1_model, phase2_model, threshold=0.1):
    # 文本编码
    encoding = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    # 第一阶段预测
    with torch.no_grad():
        logits = phase1_model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=1)
        non_neutral_prob = probs[0, 1].item()

    if non_neutral_prob < threshold:
        return "中性或无明显情感倾向"

    # 第二阶段预测
    with torch.no_grad():
        outputs = phase2_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        pred_label_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_label_idx].item()

    EMOTION_LABELS = [    "none",
    "happiness",
    "like",
    "anger",
    "sadness",
    "fear",
    "disgust",
    "surprise"]

    pred_emotion = EMOTION_LABELS[pred_label_idx]

    return f"预测情感：{pred_emotion}，置信度：{confidence:.2f}"

# ========= 主程序 =========
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
    phase1_model = load_phase1_model("phase1_sentiment_classifier.pth")
    phase2_model = load_phase2_model("weibo_emotion_roberta_lora.pth")

    while True:
        text = input("请输入一句话（输入 exit 退出）：")
        if text.strip().lower() == "exit":
            break
        result = predict(text, tokenizer, phase1_model, phase2_model)
        print(result)
