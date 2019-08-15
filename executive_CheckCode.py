import torch
import torchvision.models as models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import os

from PIL import Image,ImageChops

from Module.Pile_Residual import Pile_Residual
filename='super_pile_residual_on_checkCode_non_invert'
cnn = Pile_Residual([210,100],40,[3,16,32])
if os.path.exists('%s/%s.pt'%(filename,filename)):
	with open('%s/times_trial.txt'%filename,'r+') as f:
		times_trial=int(f.read())+1
		f.write(str(times_trial))
	cnn.load_state_dict(torch.load('%s/%s.pt'%(filename,filename)))
else:
	try:
		os.mkdir(filename)
	except OSError:
		pass
	with open('%s/times_trial.txt'%filename,'a'):
		pass
	with open('%s/times_trial.txt'%filename,'w') as f:
		times_trial=1
		f.write(str(times_trial))
	with open('测试数据.txt','a') as f:
		f.write('\n------模型%s 测试结果------\n'%filename)
####################改变模型看这里#############################
def featurize(string):
	feature=[]
	nums=list(map(int,list(string)))
	for i in nums:
		for j in range(i):
			feature.append(0)
		feature.append(1)
		for j in range(9-i):
			feature.append(0)
	return feature

def unfeaturize(feature):
	#transform a feature to a python integer.
	n = len(feature)//10
	i=0
	out=0
	for i in range(n):
		mx=-10000
		idx=0
		position=feature[10*i:10*i+10]
		for k,j in enumerate(position):
			if position[k]>mx:
				mx=j
				idx=k
		out+=idx*(10**(n-i-1))
	return out

def featurized_list(inp):
	return list(map(featurize,inp))

def unfeaturized_list(inp):
	return list(map(unfeaturize,inp))

def this(img): #transform
	trans=transforms.ToTensor()
	avgpool=torch.nn.AvgPool2d(3,stride=1,padding=1)
	maxpool=torch.nn.MaxPool2d(3,stride=1,padding=1)
	img=ImageChops.invert(img)
	img=trans(img)
	# for i in range(2):
	# 	img=maxpool(img)
	return img

def that(img): #transform
	trans=transforms.ToTensor()
	img=ImageChops.invert(img)
	img=trans(img)
	return img

EPOCH=10

BATCH_SIZE=50

LR=0.00001

#ImageFolder
train_set = dsets.ImageFolder('./checkCode',transform=this)
test_set = dsets.ImageFolder('./checkCode',transform=that)
train_class = train_set.classes
train_class = featurized_list(train_class)
train_loader = torch.utils.data.DataLoader(dataset=train_set,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=BATCH_SIZE,shuffle=False,drop_last=True)

criterion = torch.nn.MSELoss(reduction='sum')

optimizer = torch.optim.Adam(cnn.parameters(),lr=LR)

for epoch in range(EPOCH):
	for i,(images,labels) in enumerate(train_loader):
		images = Variable(images)
		classes = []
		for label in labels:
			classes.append(train_class[label])
		try:
			classes = torch.tensor(classes,dtype=torch.float)
		except ValueError:
			continue
		labels = Variable(classes)
		output = cnn(images)
		# print(labels.size(),output.size(),images.size())
		# print(labels.dtype,output.dtype)
		loss=criterion(output,labels)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if i%100 == 0:
			print('Epoch{} [{}] loss:{}'.format(epoch,i,loss.item()))
			# print(output)
			# print(labels)

torch.save(cnn.state_dict(),'%s/%s.pt'%(filename,filename))
####################改变模型看这里###########################

for i,(images,labels) in enumerate(test_loader):
	if i > 3:
		break
	output=cnn(images)
	output=unfeaturized_list(output)
	classes = []
	for label in labels:
		classes.append(int(unfeaturized_list(train_class)[label]))
	print(classes)
	print(output)
	n=len(classes)
	r=0
	for i in range(n):
		if classes[i] == output[i]:
			r+=1
	print('accuracy:',r*1.0/n)













# train_set.classes=featurized_list(train_set.classes)