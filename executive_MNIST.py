###encoding='utf-8'####

import os

import torch
import torchvision.models as models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image,ImageChops
import matplotlib.pyplot as plt

from MyTransform import comb

from Module.Pile_Residual import Pile_Residual
filename = 'super_pile_residual_on_MNIST_non_invert'
filepath = 'Params/%s/%s.pt'%(filename,filename)

cnn = Pile_Residual([28,28],10,[1,16,32])
if os.path.exists('Params/%s/%s.pt'%(filename,filename)):
	cnn.load_state_dict(torch.load(filepath))
else:
	try:
		os.mkdir('Params/'+filename)
	except OSError:
		pass
	with open('测试数据.txt','a') as f:
		f.write('\n------模型%s 测试结果------\n'%filename)
####################改变模型看这里#############################

this = comb(invert=False,rotate=False,degree=45,flip='')

that = comb(invert=False,rotate=False,degree=45,flip='')


#超参数
device='cuda:0'

torch.device(device)

EPOCH=1

BATCH_SIZE=50

LR=0.00001

# MNIST数据集
train_set = dsets.MNIST(root='./MNIST',transform=this.transform,train=True,download=True)
test_set = dsets.MNIST(root='./MNIST',transform=that.transform,train=False,download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=BATCH_SIZE,shuffle=True)
test_x = Variable(torch.unsqueeze(test_set.test_data,dim=1)).detach().type(torch.FloatTensor)
test_y = test_set.test_labels

criterion = torch.nn.CrossEntropyLoss(reduction='sum')

optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)

#开始学习
for epoch in range(EPOCH):
	for i,(x,y) in enumerate(train_loader):
		x = Variable(x)
		y = Variable(y)
		output = cnn(x)
		loss=criterion(output,y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if i%100 == 0:
			print('Epoch{} [{}] loss:{}'.format(epoch,i,loss.item()))
			# print(output)
			# print(labels)

#保存模型
torch.save(cnn.state_dict(),'Params/%s/%s.pt'%(filename,filename))

#检验学习成果
n = len(test_x)
pils = transforms.ToPILImage()
test_x=torch.tensor(list(map(lambda x:that.transform(pils(x)).numpy().tolist(),test_x)))
output=cnn(test_x)
output=torch.max(output,1)[1].cpu().detach().numpy()
test_y=test_y.cpu().numpy()
r=0
for i in range(n):
	if test_y[i] == output[i]:
		r+=1

#更新检测数据
views=[]
views.append('Epoch: '+str(EPOCH))
views.append('Learning rate: '+str(LR))
views.append('predict: '+str(output[:20]))
views.append('reality: '+str(test_y[:20]))
views.append('accuracy: '+str(r*1.0/n))

printout='\t'+'\n\t'.join(views)+'\n'
print(printout)
with open('测试数据.txt','a') as f:
	f.write(printout)












# train_set.classes=featurized_list(train_set.classes)