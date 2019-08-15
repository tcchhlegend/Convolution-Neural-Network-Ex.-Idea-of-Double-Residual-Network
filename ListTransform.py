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