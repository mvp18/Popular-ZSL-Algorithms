import numpy as np
import argparse
from scipy import io
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description="ESZSL")

parser.add_argument('-data', '--dataset', help='choose between APY, AWA2, AWA1, CUB, SUN', default='AWA2', type=str)
parser.add_argument('-mode', '--mode', help='train/test, if test set alpha, gamma to best values below', default='train', type=str)
parser.add_argument('-alpha', '--alpha', default=0, type=int)
parser.add_argument('-gamma', '--gamma', default=0, type=int)

"""

Alpha --> Regularizer for Kernel/Feature Space
Gamma --> Regularizer for Attribute Space

Best Values of (Alpha, Gamma) found by validation & corr. test accuracies:

AWA1 -> (3, 0)  -> Test Acc : 0.5680
AWA2 -> (3, 0)  -> Test Acc : 0.5482
CUB  -> (3, -1) -> Test Acc : 0.5394
SUN  -> (3, 2)  -> Test Acc : 0.5569
APY  -> (3, -1) -> Test Acc : 0.3856

"""

class ESZSL():
	
	def __init__(self):

		data_folder = '../xlsa17/data/'+args.dataset+'/'
		res101 = io.loadmat(data_folder+'res101.mat')
		att_splits=io.loadmat(data_folder+'att_splits.mat')

		train_loc = 'train_loc'
		val_loc = 'val_loc'
		test_loc = 'test_unseen_loc'

		feat = res101['features']
		# Shape -> (dxN)
		self.X_train = feat[:, np.squeeze(att_splits[train_loc]-1)]
		self.X_val = feat[:, np.squeeze(att_splits[val_loc]-1)]
		self.X_trainval = np.concatenate((self.X_train, self.X_val), axis=1)
		self.X_test = feat[:, np.squeeze(att_splits[test_loc]-1)]

		print('Tr:{}; Val:{}; Ts:{}\n'.format(self.X_train.shape[1], self.X_val.shape[1], self.X_test.shape[1]))

		labels = res101['labels']
		labels_train = labels[np.squeeze(att_splits[train_loc]-1)]
		self.labels_val = labels[np.squeeze(att_splits[val_loc]-1)]
		labels_trainval = np.concatenate((labels_train, self.labels_val), axis=0)
		self.labels_test = labels[np.squeeze(att_splits[test_loc]-1)]

		train_labels_seen = np.unique(labels_train)
		val_labels_unseen = np.unique(self.labels_val)
		trainval_labels_seen = np.unique(labels_trainval)
		test_labels_unseen = np.unique(self.labels_test)

		i=0
		for labels in train_labels_seen:
			labels_train[labels_train == labels] = i    
			i+=1
		
		j=0
		for labels in val_labels_unseen:
			self.labels_val[self.labels_val == labels] = j
			j+=1
		
		k=0
		for labels in trainval_labels_seen:
			labels_trainval[labels_trainval == labels] = k
			k+=1
		
		l=0
		for labels in test_labels_unseen:
			self.labels_test[self.labels_test == labels] = l
			l+=1

		self.gt_train = np.zeros((labels_train.shape[0], len(train_labels_seen)))
		self.gt_train[np.arange(labels_train.shape[0]), np.squeeze(labels_train)] = 1

		self.gt_trainval = np.zeros((labels_trainval.shape[0], len(trainval_labels_seen)))
		self.gt_trainval[np.arange(labels_trainval.shape[0]), np.squeeze(labels_trainval)] = 1

		sig = att_splits['att']
		# Shape -> (Number of attributes, Number of Classes)
		self.train_sig = sig[:, train_labels_seen-1]
		self.val_sig = sig[:, val_labels_unseen-1]
		self.trainval_sig = sig[:, trainval_labels_seen-1]
		self.test_sig = sig[:, test_labels_unseen-1]

	def find_W(self, X, y, sig, alpha, gamma):

		part_0 = np.linalg.pinv(np.matmul(X, X.T) + (10**alpha)*np.eye(X.shape[0]))
		part_1 = np.matmul(np.matmul(X, y), sig.T)
		part_2 = np.linalg.pinv(np.matmul(sig, sig.T) + (10**gamma)*np.eye(sig.shape[0]))

		W = np.matmul(np.matmul(part_0, part_1), part_2) # Feature Dimension x Number of Attributes

		return W

	def fit(self):

		print('Training...\n')

		best_acc = 0.0

		for alph in range(-3, 4):
			for gamm in range(-3, 4):
				W = self.find_W(self.X_train, self.gt_train, self.train_sig, alph, gamm)
				acc = self.zsl_acc(self.X_val, W, self.labels_val, self.val_sig)
				print('Val Acc:{}; Alpha:{}; Gamma:{}\n'.format(acc, alph, gamm))
				if acc>best_acc:
					best_acc = acc
					alpha = alph
					gamma = gamm

		print('\nBest Val Acc:{} with Alpha:{} & Gamma:{}\n'.format(best_acc, alpha, gamma))
		
		return alpha, gamma

	def zsl_acc(self, X, W, y_true, sig): # Class Averaged Top-1 Accuarcy

		class_scores = np.matmul(np.matmul(X.T, W), sig) # N x Number of Classes
		predicted_classes = np.array([np.argmax(output) for output in class_scores])
		cm = confusion_matrix(y_true, predicted_classes)
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		acc = sum(cm.diagonal())/sig.shape[1]

		return acc

	# def zsl_acc(self, X, W, y_true, classes, sig): # Class Averaged Top-1 Accuarcy

	# 	class_scores = np.matmul(np.matmul(X.T, W), sig) # N x Number of Classes
	# 	y_pred = np.array([np.argmax(output) for output in class_scores])
	# 	y_true = np.squeeze(y_true)

	# 	per_class_acc = np.zeros(len(classes))

	# 	for i in range(len(classes)):
	# 		is_class = y_true==classes[i]
	# 		per_class_acc[i] = ((y_pred[is_class]==y_true[is_class]).sum())/is_class.sum()
		
	# 	return per_class_acc.mean()

	def evaluate(self, alpha, gamma):

		print('Testing...\n')

		best_W = self.find_W(self.X_trainval, self.gt_trainval, self.trainval_sig, alpha, gamma) # combine train and val

		test_acc = self.zsl_acc(self.X_test, best_W, self.labels_test, self.test_sig)

		print('Test Acc:{}'.format(test_acc))

if __name__ == '__main__':
	
	args = parser.parse_args()
	print('Dataset : {}\n'.format(args.dataset))
	
	clf = ESZSL()
	
	if args.mode=='train': 
		args.alpha, args.gamma = clf.fit()
	
	clf.evaluate(args.alpha, args.gamma)
