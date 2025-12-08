import torch
import timm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0, global_pool='avg').to(device)

# تست با یک ورودی تصادفی
x = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    feat = model(x)
print(feat.shape)  # باید [1, feature_dim] باشه
