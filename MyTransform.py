import torchvision.transforms as transforms
from PIL import Image,ImageChops

class comb:
	def __init__(self,invert=False,rotate=False,degree=45,flip=''):
		self.invert=invert
		self.rotate=rotate
		self.degree=degree
		self.flip=flip

	def transform(self,img):
		trans=transforms.ToTensor()
	
		if self.invert:
			img=ImageChops.invert(img)
		if self.rotate:
			ig=img.rotate(degree)
		if self.flip == 'TOP_BOTTOM':
			img=img.transpose(Image.FLIP_TOP_BOTTOM)
		elif self.flip == 'LEFT_RIGHT':
			img=img.transpose(Image.FLIP_LEFT_RIGHT)
		img=trans(img)
		return img