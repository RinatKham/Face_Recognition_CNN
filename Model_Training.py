import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.transforms import transforms
import random


root = './dataset'


class SiameseMobileNetV2(nn.Module):
    def __init__(self):
        super(SiameseMobileNetV2, self).__init__()
        self.mobilenet = mobilenet_v2(weights = MobileNet_V2_Weights.DEFAULT).features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Sequential(nn.Linear(1280, 512),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=0.5))
        self.fc2 = nn.Sequential(nn.Linear(512, 256),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=0.5))
        self.fc3 = nn.Sequential(nn.Linear(256, 128),
                                 nn.ReLU(inplace=True))

    def forward_once(self, x):
        output = self.mobilenet(x)
        output = self.pool(output)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output

    def forward(self, input1, input2 = None, input3 = None):
        if input2 is None:
            return self.forward_once(input1)

        else:
            return self.forward_triplet(input1, input2, input3)

    def forward_triplet(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        return output1, output2, output3


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor_output, positive_output, negative_output):
        distance_positive = F.pairwise_distance(anchor_output, positive_output, 2)
        distance_negative = F.pairwise_distance(anchor_output, negative_output, 2)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


def split_dataset(directory, split=0.9):
    folders = os.listdir(directory)
    num_train = int(len(folders) * split)

    random.shuffle(folders)

    train_list, test_list = {}, {}

    # Creating Train-list
    for folder in folders[:num_train]:
        num_files = len(os.listdir(os.path.join(directory, folder)))
        train_list[folder] = num_files

    # Creating Test-list
    for folder in folders[num_train:]:
        num_files = len(os.listdir(os.path.join(directory, folder)))
        test_list[folder] = num_files

    return train_list, test_list


def create_triplets(directory, folder_list, max_files=10):
    triplets = []
    folders = list(folder_list.keys())

    for folder in folders:
        path = os.path.join(directory, folder)
        files = list(os.listdir(path))[:max_files]
        num_files = len(files)

        for i in range(num_files - 1):
            for j in range(i + 1, num_files):
                anchor = (folder, f"{i}.jpg")
                positive = (folder, f"{j}.jpg")

                neg_folder = folder
                while neg_folder == folder:
                    neg_folder = random.choice(folders)
                neg_file = random.randint(0, folder_list[neg_folder] - 1)
                negative = (neg_folder, f"{neg_file}.jpg")

                triplets.append((anchor, positive, negative))
    print(len(triplets))
    random.shuffle(triplets)
    return triplets


class TripletFaceDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, triplets):
        self.root_dir = root_dir
        self.triplet = triplets
        # путь к корневой директории с изображениями лиц
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # изменение размера изображений до 224х224
            transforms.ToTensor(),  # преобразование изображения в Тензор Torch
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # нормализация тензоров
        ])

    def __len__(self):
        return len(self.triplet)

    def __getitem__(self, idx):
        a,p,n = self.triplet[idx]
        anchor_img = self.transform(Image.open(os.path.join(self.root_dir, a[0], a[1])))
        positive_img = self.transform(Image.open(os.path.join(self.root_dir, p[0], p[1])))
        negative_img = self.transform(Image.open(os.path.join(self.root_dir, n[0], n[1])))
        return anchor_img, positive_img, negative_img


def evaluate(model, out_anchor, out_positive, out_negative, margin=1):
    with torch.no_grad():
        distance_positive = F.pairwise_distance(out_anchor, out_positive, 2)
        distance_negative = F.pairwise_distance(out_anchor, out_negative, 2)
        losses = distance_positive - distance_negative + margin
        return losses.mean()


train_list, test_list = split_dataset(root, split=0.9)

train_triplet = create_triplets(root, train_list)
test_triplet  = create_triplets(root, test_list)
batch_size = 8
# Define dataset and data loader
train_dataset = TripletFaceDataset(root, train_triplet)
test_dataset = TripletFaceDataset(root, test_triplet)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size = batch_size)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size = batch_size)


# Define model, loss function and optimizer
model = SiameseMobileNetV2().cuda()
criterion = TripletLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), amsgrad=True, lr=0.0001)


print(train_loader.__len__())
print(test_loader.__len__())

avg_ac = []
train_loss = []

# Training loop
for epoch in range(10):
    model.train()
    epoch_loss = []
    ac = 0.0
    num = 0
    running_loss = 0.0
    for i, (anchor, positive, negative) in enumerate(train_loader):
        model.train()
        anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
        optimizer.zero_grad()
        anchor_output = model(anchor)
        positive_output = model(positive)
        negative_output = model(negative)
        loss = criterion(anchor_output, positive_output, negative_output)
        epoch_loss.append(loss)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    for i, (anchor, positive, negative) in enumerate(test_loader):
        model.eval()
        anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
        anchor_output = model(anchor)
        positive_output = model(positive)
        negative_output = model(negative)
        temp = evaluate(model, anchor_output, positive_output, negative_output)
        if temp < 0:
            ac += 1
        num = i
    avg_ac.append(ac/num)
    epoch_loss = sum(epoch_loss) / len(epoch_loss)
    train_loss.append(epoch_loss)
    print(f"Accuracy on test = {ac/num:.5f}")
    print('Epoch %d, Loss: %.3f' % (epoch + 1, running_loss / (i + 1)))
model.eval()

# Save model
torch.save(model.state_dict(), 'model_weights.pth')




