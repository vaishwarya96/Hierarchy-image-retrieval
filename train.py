import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
import torch.utils.data as data
import torch.nn as nn
from torch import optim
import pickle
import torchvision.transforms as transforms
import torchvision
from torch.utils.tensorboard import SummaryWriter

from config import get_cfg_defaults 
from dataset_utils import utils, load_dataset
from models.wide_resnet import WideResNet
from models.efficientnet import EfficientNet, ResNet, EfficientNet_new, DogBreedPretrainedWideResnet
from train_utils.loss import TripletLoss
from train_utils.cluster_utils import get_clustered_model_and_dataset
from val import validate

cfg = get_cfg_defaults()
torch.multiprocessing.set_sharing_strategy('file_system')

#Set random seed
np.random.seed(cfg.SYSTEM.RANDOM_SEED)
torch.random.manual_seed(cfg.SYSTEM.RANDOM_SEED)

#Summary writer
writer = SummaryWriter(cfg.MODEL.CHECKPOINT_DIR)

#Create model checkpoint directory
model_path = cfg.MODEL.CHECKPOINT_DIR
if not os.path.exists(model_path):
    os.makedirs(model_path, exist_ok=True)
model_path = os.path.join(model_path, cfg.MODEL.EXPERIMENT)

#Load dataset
id_map = utils.get_id_map(cfg.DATASET.ID_MAP_PATH)
n_classes = len(list(id_map.keys()))

with open(os.path.join(cfg.MODEL.CHECKPOINT_DIR,'id_map.pkl'), 'wb') as f:
    pickle.dump(id_map, f)




X,y,_ = utils.get_dataset(cfg.DATASET.DATASET_PATH, id_map)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg.DATASET.VAL_SIZE, random_state=cfg.SYSTEM.RANDOM_SEED, stratify=y)


train_data_loader = load_dataset.LoadDataset(X_train, y_train)
test_data_loader = load_dataset.LoadDataset(X_test, y_test, train=False)
cluster_data_loader = load_dataset.LoadDataset(X_train, y_train)

train_data = data.DataLoader(train_data_loader, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.SYSTEM.NUM_WORKERS)
cluster_data = data.DataLoader(cluster_data_loader, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.SYSTEM.NUM_WORKERS)
test_data = data.DataLoader(test_data_loader, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False, num_workers=cfg.SYSTEM.NUM_WORKERS)



label_dict = {}
for i in range(n_classes):
    label_dict[i] = i


img_list = X_train
label_list = y_train

    
#Model
model = EfficientNet_new(num_classes = n_classes, in_channels=cfg.DATASET.NUM_CHANNELS)
model = model.cuda()

#Losses
loss1 = nn.CrossEntropyLoss()
loss2 = TripletLoss(margin=cfg.TRAIN.TRIPLET_LOSS_MARGIN)

#Optimizer
optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
sched = optim.lr_scheduler.OneCycleLR(optimizer, cfg.TRAIN.LEARNING_RATE, epochs=cfg.TRAIN.NUM_EPOCHS, steps_per_epoch=len(train_data))

min_loss = 1e5
best_acc = 0
lr = cfg.TRAIN.LEARNING_RATE

for epoch in range(cfg.TRAIN.NUM_EPOCHS):
    emb_database = []
    label_database = []

    if epoch % 10 == 0 and epoch != 0:
    #if epoch == 10:
        lr /= 10
        lr = max(lr, 0.0001)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
        sched = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=cfg.TRAIN.NUM_EPOCHS, steps_per_epoch=len(train_data))

    if epoch % cfg.TRAIN.VAL_EPOCHS == 0 and epoch != 0:
        model, cluster_data, label_dict, label_counts, img_list, label_list = get_clustered_model_and_dataset(train_data, model, C_norm)
    
    for it, (img, label, _) in enumerate(cluster_data):
        b_images = img.cuda()
        b_labels = label.cuda()

        optimizer.zero_grad()
        emb, output = model(b_images)
        emb_database.extend(emb.detach().cpu().numpy())
        label_database.extend(label)
        logit = nn.Softmax(dim=-1)(output)
        pred = torch.argmax(logit, dim=1)       

        loss_value = 1 * loss1(output, b_labels)# + 1*loss2(emb, b_labels)
        writer.add_scalar("Loss/train", loss_value, epoch)
        loss_value.backward()

        nn.utils.clip_grad_value_(model.parameters(), cfg.TRAIN.GRAD_CLIP)

        optimizer.step()
        sched.step()


    acc, C_norm = validate(model, test_data, label_dict, emb_database, label_database)
    writer.add_scalar("Accuracy/val", acc, epoch)

    print("Epoch %d: %f" %(epoch, acc))

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), model_path)
        np.save(os.path.join(cfg.MODEL.CHECKPOINT_DIR,'train_img_list.npy'), img_list)
        np.save(os.path.join(cfg.MODEL.CHECKPOINT_DIR,'train_label_list.npy'), label_list)
        np.save(os.path.join(cfg.MODEL.CHECKPOINT_DIR,'confusion_matrix.npy'), C_norm)
        with open(os.path.join(cfg.MODEL.CHECKPOINT_DIR,'label_dict.pkl'), 'wb') as f:
            pickle.dump(label_dict,f)
        print("Model saved")

print("Training finished")
writer.flush()
writer.close()

