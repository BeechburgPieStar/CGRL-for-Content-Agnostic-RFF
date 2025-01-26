import argparse
import os
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from util.get_dataset import TrainDataset, TestDataset
import torch
from util.CNNmodel_CAM import *
from util.mmd_loss import MMDLoss as MMD
from torchsummary import summary
import torch.nn.functional as F
import numpy as np
import random

def parse_args():
    parser = argparse.ArgumentParser(description="CGRL")
    parser.add_argument('--gpu', type=str, default="0", help='Number of GPU')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num_cls', type=int, default=5, help='Number of devices')
    parser.add_argument('--snr', type=int, default=5, help='Signal to Noise Ratio')
    parser.add_argument('--len_mark', type=int, default=32, help='Length of mark')
    parser.add_argument('--mark_state', type=str, default='w', help='w or w/o mark')
    parser.add_argument('--lam_ACR', type=float, default=0.001, help='Lambda for attention consistency regularization (Lambda2)')
    parser.add_argument('--lam_SCR', type=float, default=0.01, help='Lambda for semantic consistency regularization (Lambda1)')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay for the optimizer')
    parser.add_argument('--code_state', type=str, default="only_test", help='Three mode: only_train, only_test, train_test')
    return parser.parse_args()

conf = parse_args()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(2023)

def get_cam_for_target(cam, target):
    cam_for_target  = torch.stack([cam[i, target[i], :] for i in range(target.size()[0])])
    min_vals = cam_for_target.min(dim=1, keepdim=True)[0]
    cam_for_target  -= min_vals
    max_vals = cam_for_target.max(dim=1, keepdim=True)[0]
    normalized_cam_for_target = cam_for_target/max_vals
    return normalized_cam_for_target

def train(model, loss, train_dataloader, optimizer, epoch):
    model.train()
    correct = 0
    all_loss = 0
    for data_nn in train_dataloader:
        data, target = data_nn
        target = target.long()
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        data_mask = data.clone()
        data_mask[:, :, 0:int(conf.len_mark*8)] = 0

        optimizer.zero_grad()
        cam, embedding, output = model(data)
        output = F.log_softmax(output, dim=1)
        cam = get_cam_for_target(cam, target)

        cam_mask, embedding_mask, output_mask = model(data_mask)
        output_mask = F.log_softmax(output_mask, dim=1)
        cam_mask = get_cam_for_target(cam_mask, target)

        CLS_loss = loss(output_mask, target)
        ACR_loss = MMD()(cam, cam_mask)
        SCR_loss = MMD()(embedding, embedding_mask)

        result_loss = CLS_loss + conf.lam_ACR*ACR_loss + conf.lam_SCR*SCR_loss
        result_loss.backward()

        optimizer.step()
        all_loss += result_loss.item()*data.size()[0]
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    print('Train Epoch: {} \tLoss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        all_loss / len(train_dataloader.dataset),
        correct,
        len(train_dataloader.dataset),
        100.0 * correct / len(train_dataloader.dataset))
    )

def evaluate(model, loss, test_dataloader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            cam, embedding, output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += loss(output, target).item()*data.size()[0]
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    fmt = '\nValidation set: Loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            test_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )

    return test_loss

def test(model, test_dataloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            cam, embedding, output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print(correct / len(test_dataloader.dataset))


def train_and_evaluate(model, loss_function, train_dataloader, val_dataloader, optimizer, epochs, save_path):
    current_min_test_loss = 100
    for epoch in range(1, epochs + 1):
        train(model, loss_function, train_dataloader, optimizer, epoch)
        test_loss = evaluate(model, loss_function, val_dataloader, epoch)
        if test_loss < current_min_test_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_test_loss, test_loss))
            current_min_test_loss = test_loss
            torch.save(model, save_path)
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")

def TrainDataset_prepared(mark_state, snr, len_mark):
    x_train, x_val, y_train, y_val = TrainDataset(mark_state, snr, len_mark)
    return x_train, x_val, y_train, y_val

def TestDataset_prepared(mark_state, snr, len_mark):
    x_test, y_test = TestDataset(mark_state, snr, len_mark)
    return x_test, y_test

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = conf.gpu
    modelweightfile = f'model/CGRL_{conf.snr}dB_{conf.len_mark}_{conf.lam_ACR}_{conf.lam_SCR}.pth'

    x_train, x_val, y_train, y_val = TrainDataset_prepared(conf.mark_state, conf. snr, conf.len_mark)

    train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
    val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=True)

    model = ResNet20(conf.num_cls).cuda()

    optim = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay = conf.wd)

    loss = nn.NLLLoss()
    if torch.cuda.is_available():
        loss = loss.cuda()

    if conf.code_state == "only_train" or conf.code_state == "train_test":
        train_and_evaluate(model, 
            loss_function=loss, 
            train_dataloader=train_dataloader, 
            val_dataloader=val_dataloader, 
            optimizer=optim, 
            epochs=conf.epochs, 
            save_path=modelweightfile)

    if conf.code_state == "only_test" or conf.code_state == "train_test":
        modelweightfile = f'model/CGRL_{conf.snr}dB_{conf.len_mark}_{conf.lam_ACR}_{conf.lam_SCR}.pth'
        x_test, y_test = TestDataset_prepared('wrong', conf.snr, conf.len_mark)# test when identifier has been tampered with
        test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
        model = torch.load(modelweightfile)
        test(model,test_dataloader)

if __name__ == '__main__':
    main()