import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models
from PIL import Image
from inception import inception_v3
from update import *

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    normalize
])

#train_data = datasets.ImageFolder('/srv/data/datasets/wonchul/datasets/kaggle/cats_vs_dogs/train/', transform=transform_train)
#trainloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=8)
#
#test_data = datasets.ImageFolder('/srv/data/datasets/wonchul/datasets/kaggle/cats_vs_dogs/test/', transform=transform_test)
#testloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=8)
#
classes = {0: 'cat', 1: 'dog'}

model = inception_v3(pretrained=True)
#model = models.inception_v3(pretrained=True)
model.fc = torch.nn.Linear(2048, len(classes))
model.eval()
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

model._modules.get('Mixed_7c').register_forward_hook(hook_feature)
softmax_w = list(model.parameters())[-2].data.cpu().numpy()
print('softmax_w: ', softmax_w.shape)

img = Image.open('./sample.jpg')

img_tensor = preprocess(img).unsqueeze(0)
print('img: ', img_tensor.size())

logit = model(img_tensor)
print('logit: ', logit.size())

feature = features_blobs[0]
print('feature blob: ', feature.shape)

#get_cam(model, features_blobs, img, classes, './sample.jpg')

size_upsample = (256, 256)
bs, nc, h, w = feature.shape # 1, 2048, 5, 5

feature_ = feature.reshape((nc, h*w)) # 2048, 25
cam = softmax_w[0].dot(feature_)
cam = cam.reshape(h, w)
cam_img = cam - np.min(cam)
cam_img = np.uint8(255*cam_img)
output_cam = cv2.resize(cam_img, size_upsample)

print(output_cam.shape)

cv2.imwrite('output_cam.jpg', output_cam)

img = cv2.imread('./sample.jpg')
height, width, _ = img.shape
cam_ = cv2.resize(output_cam, (width, height))
heatmap = cv2.applyColorMap(cam_, cv2.COLORMAP_JET)
res = heatmap*0.3 + img*0.5
cv2.imwrite('res.jpg', res)
