import readtr as rt
from sklearn import svm
import readf as rd
import numpy as np

clf=svm.SVC()
a,b=rt.read()

def init(i=0,j=62):
	global clf
	global a
	global b
	clf=svm.SVC()
	a,b=rt.read(i,j)
	clf.fit(a[0],a[1])

def main(testfile=None):
	global clf
	global a
	global b
	if testfile:
		tf=rd.read(testfile)
		ts=np.array(tf)
		return(clf.predict(ts))
	else:
		pred=[int(a+.1) for a in clf.predict(b[0])]
		nc=sum(int(x==y) for x,y in zip(pred,b[1]))
		print "{0} of {1} correct = {2}%".format(nc,len(b[1]),(100.0*float(nc)/len(b[1])))
		

if __name__=="__main__":
	init()
	main()
