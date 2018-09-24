from dataset import *
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.backends import cudnn

from model import * 

import argparse
import model

parser = argparse.ArgumentParser(description='')

parser.add_argument('--img_dir', default='./data', help='path to directory containing the images')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train')
parser.add_argument('--learning_rate',type=float,default=0.005,help='learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')

opts = parser.parse_args()

dataset = Prostate_Dataset('./data/slice_data')
dataloader = DataLoader(dataset,batch_size=opts.batch_size,drop_last=True,shuffle=True)


model = Model().cuda()

try:
    model = torch.load('./model/Resnet.pkl')
    print("\n----------------------Model restored----------------------\n")
except:
    print("\n----------------------Model not restored----------------------\n")
    pass

optimizer = torch.optim.Adam(list(model.resnet.parameters()),lr=opts.learning_rate)
criterion = nn.BCELoss()


cudnn.benchmark = True


for epoch in tqdm(range(opts.epochs)):
    for index, sample in enumerate(dataloader):
        
        img = Variable(sample['image'].float()).cuda()
        label = Variable(sample['label'].float()).cuda()
        label = label.unsqueeze(dim=1)
        #print(label.shape)

        #onehot = torch.FloatTensor(opts.batch_size, 2).cuda()
        #onehot.zero_()
        #onehot.scatter_(1,label,1)


        #one_hot = label.scatter_(1,label,1)
        #print(onehot)
        #one_hot = torch.FloatTensor(opts.batch_size, 2, n).zero_()

        prediction_prob = model.forward(img)
        #print(prediction_prob)
        #print(label)
        #print(prediction)
        
        loss = criterion(prediction_prob,label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        if index % 500 == 0:
            print('prediction prob {}'.format(prediction_prob))
            prediction = prediction_prob > 0.5
            
            # print('Ground truth {}'.format(label))
            
            print("Loss {}".format(loss))
            #corrects = torch.sum(prediction == onehot)
            #print("Correct predictions {}".format(corrects))

            #print('prediction prob {}'.format(prediction_prob))
            print('ground truth: {}'.format(label))
            print('prediction {} '.format(prediction))

            label = label.cpu().numpy()
            prediction = prediction.cpu().numpy()
            correct = sum((label == prediction))
            accuracy = correct/opts.batch_size
            print('Accuracy {}'.format(accuracy))
            torch.save(model,'./model/Resnet.pkl')

            img = img[0,:,:].data.cpu().numpy()
            print(img.shape)
            img = np.transpose(img,[1,2,0])
            imsave('test_train_img.png',img)



            #print('one hot {}' .format(onehot))

            #print("Prediction")


        
        