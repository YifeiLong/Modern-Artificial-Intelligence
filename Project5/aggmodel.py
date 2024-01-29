import torch
import torch.nn as nn
import torchvision as tv
import transformers


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
            nn.LeakyReLU()
        )

    def forward(self, X):
        return self.fc(X) + X


class DecoderLayer(nn.Module):
    def __init__(self, img_dim, mem_dim):
        super().__init__()
        self.projection = nn.Sequential(
            ResidualBlock(img_dim + mem_dim),
            ResidualBlock(img_dim + mem_dim),
            nn.Linear(img_dim + mem_dim, mem_dim)
        )
        self.text_layer = nn.TransformerDecoderLayer(mem_dim, 4, batch_first=True, norm_first=True)

    def forward(self, feature, image, memory, attention_mask):
        projected = self.projection(torch.concat([feature, image.unsqueeze(-2)], dim=-1))
        return self.text_layer(projected, memory, memory_key_padding_mask=attention_mask) * 0.5 + feature * 0.5


class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        TEXT_DIM = 768
        IMG_DIM = 2048

        self.text_model = transformers.AutoModel.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        # 保留除最后一层之外的层
        self.img_model = nn.Sequential(
            *list(tv.models.resnet50(pretrained=True).children())[:-1]
        )

        self.dec_layer = nn.ModuleList([DecoderLayer(IMG_DIM, TEXT_DIM) for _ in range(2)])
        self.enc_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(TEXT_DIM, 4, batch_first=True, norm_first=True), 2)
        self.fc = nn.Sequential(
            nn.Linear(TEXT_DIM, 3),
            nn.Softmax(dim=-1)
        )


    def forward(self, img, text):
        with torch.no_grad():
            mask = ~text['attention_mask'].to(torch.bool)
            img = self.img_model(img).flatten(-3, -1)
            text = self.text_model(**text).last_hidden_state

        text = self.enc_layer(text, src_key_padding_mask=mask)
        feature = (text * (~mask).unsqueeze(dim=-1)).sum(dim=-2) / ((~mask).sum(dim=-1).unsqueeze(-1) + 1e-10)
        if not self.training:
            feature = (feature + torch.randn_like(feature) * 0.1) / (1.0 + 0.1)
        feature = feature.unsqueeze(-2)

        for layer in self.dec_layer:
            feature = layer(feature, img, text, mask)

        return self.fc(feature.squeeze(-2))


class OnlyTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        TEXT_DIM = 768
        IMG_DIM = 2048

        self.text_model = transformers.AutoModel.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        self.img_model = nn.Sequential(
            *list(tv.models.resnet50(pretrained=True).children())[:-1]
        )
        self.dec_layer = nn.ModuleList([DecoderLayer(IMG_DIM, TEXT_DIM) for _ in range(2)])
        self.enc_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(TEXT_DIM, 4, batch_first=True, norm_first=True), 2)
        self.fc = nn.Sequential(
            nn.Linear(TEXT_DIM, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, img, text):
        with torch.no_grad():
            mask = ~text['attention_mask'].to(torch.bool)
            # 将图片的输入置为0，相当于只输入文本
            img = self.img_model(img).flatten(-3, -1) * 0.0
            text = self.text_model(**text).last_hidden_state

        text = self.enc_layer(text, src_key_padding_mask=mask)
        feature = (text * (~mask).unsqueeze(dim=-1)).sum(dim=-2) / ((~mask).sum(dim=-1).unsqueeze(-1) + 1e-10)
        if not self.training:
            feature = (feature + torch.randn_like(feature) * 0.1) / (1.0 + 0.1)
        feature = feature.unsqueeze(-2)

        for layer in self.dec_layer:
            feature = layer(feature, img, text, mask)

        return self.fc(feature.squeeze(-2))


class OnlyImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        TEXT_DIM = 768
        IMG_DIM = 2048

        self.text_model = transformers.AutoModel.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        self.img_model = nn.Sequential(
            *list(tv.models.resnet50(pretrained=True).children())[:-1]
        )

        self.dec_layer = nn.ModuleList([DecoderLayer(IMG_DIM, TEXT_DIM) for _ in range(2)])
        self.enc_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(TEXT_DIM, 4, batch_first=True, norm_first=True), 2)
        self.fc = nn.Sequential(
            nn.Linear(TEXT_DIM, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, img, text):
        with torch.no_grad():
            mask = ~text['attention_mask'].to(torch.bool)
            img = self.img_model(img).flatten(-3, -1)
            # 将文本输入置为0，相当于只输入图片
            text = self.text_model(**text).last_hidden_state * 0.0

        text = self.enc_layer(text, src_key_padding_mask=mask)
        feature = (text * (~mask).unsqueeze(dim=-1)).sum(dim=-2) / ((~mask).sum(dim=-1).unsqueeze(-1) + 1e-10)
        if not self.training:
            feature = (feature + torch.randn_like(feature) * 0.1) / (1.0 + 0.1)
        feature = feature.unsqueeze(-2)

        for layer in self.dec_layer:
            feature = layer(feature, img, text, mask)

        return self.fc(feature.squeeze(-2))
