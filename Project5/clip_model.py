import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

df = pd.read_csv('./dataset/val/val.csv')
uid = df['guid'].copy().values
true_label = df['tag'].copy().values
labels = ["positive", "neutral", "negative"]
cnt = 0
total = 0.0

for num in uid:
    img_path = "./dataset/data/" + str(num) + ".jpg"
    text_path = "./dataset/data/" + str(num) + ".txt"
    img = Image.open(img_path).convert("RGB")
    with open(text_path, encoding='utf-8', errors='surrogateescape') as f:
        text = f.read().strip()

    inputs = processor(text=[f"the emotion of this picture is {label}" for label in labels],
                       images=img, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    label_index = probs.argmax(dim=-1)
    print(labels[label_index])

    if true_label[cnt] == labels[label_index]:
        total += 1
    cnt += 1

print(f'training set acc: {total / cnt}')
