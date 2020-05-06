import numpy as np
import argparse
from scipy import io, spatial
import time
from random import shuffle
import random
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description="GZSL with SJE")

parser.add_argument('-data', '--dataset', help='choose between APY, AWA2, AWA1, CUB, SUN', default='AWA2', type=str)
parser.add_argument('-e', '--epochs', default=100, type=int)
parser.add_argument('-es', '--early_stop', default=10, type=int)
parser.add_argument('-norm', '--norm_type', help='std(standard), L2, None', default='std', type=str)
parser.add_argument('-lr', '--lr', default=0.01, type=float)
parser.add_argument('-mr', '--margin', default=1, type=float)
parser.add_argument('-seed', '--rand_seed', default=42, type=int)

class SJE():
	
	def __init__(self, args):

		self.args = args

		random.seed(self.args.rand_seed)
		np.random.seed(self.args.rand_seed)

		data_folder = '../xlsa17/data/'+args.dataset+'/'
		res101 = io.loadmat(data_folder+'res101.mat')
		att_splits=io.loadmat(data_folder+'att_splits.mat')

		train_loc = 'train_loc'
		val_loc = 'val_loc'
		trainval_loc = 'trainval_loc'
		test_seen_loc = 'test_seen_loc'
		test_unseen_loc = 'test_unseen_loc'

		feat = res101['features']
		# Shape -> (dxN)
		self.X_trainval_gzsl = feat[:, np.squeeze(att_splits[trainval_loc]-1)]
		self.X_test_seen = feat[:, np.squeeze(att_splits[test_seen_loc]-1)]
		self.X_test_unseen = feat[:, np.squeeze(att_splits[test_unseen_loc]-1)]

		labels = res101['labels']
		self.labels_trainval_gzsl = np.squeeze(labels[np.squeeze(att_splits[trainval_loc]-1)])
		self.labels_test_seen = np.squeeze(labels[np.squeeze(att_splits[test_seen_loc]-1)])
		self.labels_test_unseen = np.squeeze(labels[np.squeeze(att_splits[test_unseen_loc]-1)])
		self.labels_test = np.concatenate((self.labels_test_seen, self.labels_test_unseen), axis=0)

		train_classes = np.unique(np.squeeze(labels[np.squeeze(att_splits[train_loc]-1)]))
		val_classes = np.unique(np.squeeze(labels[np.squeeze(att_splits[val_loc]-1)]))
		trainval_classes_seen = np.unique(self.labels_trainval_gzsl)
		self.test_classes_seen = np.unique(self.labels_test_seen)
		self.test_classes_unseen = np.unique(self.labels_test_unseen)
		test_classes = np.unique(self.labels_test) # All Classes of the dataset

		train_gzsl_indices=[]
		val_gzsl_indices=[]

		for cl in train_classes:
			train_gzsl_indices = train_gzsl_indices + np.squeeze(np.where(self.labels_trainval_gzsl==cl)).tolist()

		for cl in val_classes:
			val_gzsl_indices = val_gzsl_indices + np.squeeze(np.where(self.labels_trainval_gzsl==cl)).tolist()

		train_gzsl_indices = sorted(train_gzsl_indices)
		val_gzsl_indices = sorted(val_gzsl_indices)
		
		self.X_train_gzsl = self.X_trainval_gzsl[:, np.array(train_gzsl_indices)]
		self.labels_train_gzsl = self.labels_trainval_gzsl[np.array(train_gzsl_indices)]
		
		self.X_val_gzsl = self.X_trainval_gzsl[:, np.array(val_gzsl_indices)]
		self.labels_val_gzsl = self.labels_trainval_gzsl[np.array(val_gzsl_indices)]

		print('Tr:{}; Val:{}; Tr+Val:{}; Test Seen:{}; Test Unseen:{}\n'.format(self.X_train_gzsl.shape[1], self.X_val_gzsl.shape[1], 
			                                                                    self.X_trainval_gzsl.shape[1], self.X_test_seen.shape[1], 
			                                                                    self.X_test_unseen.shape[1]))

		i=0
		for labels in trainval_classes_seen:
			self.labels_trainval_gzsl[self.labels_trainval_gzsl == labels] = i    
			i+=1

		j=0
		for labels in train_classes:
			self.labels_train_gzsl[self.labels_train_gzsl == labels] = j
			j+=1

		k=0
		for labels in val_classes:
			self.labels_val_gzsl[self.labels_val_gzsl == labels] = k
			k+=1

		sig = att_splits['att']
		# Shape -> (Number of attributes, Number of Classes)
		self.trainval_sig = sig[:, trainval_classes_seen-1]
		self.train_sig = sig[:, train_classes-1]
		self.val_sig = sig[:, val_classes-1]
		self.test_sig = sig[:, test_classes-1] # Entire Signature Matrix

		if self.args.norm_type=='std':
			scaler_train = preprocessing.StandardScaler()
			scaler_trainval = preprocessing.StandardScaler()
			
			scaler_train.fit(self.X_train_gzsl.T)
			scaler_trainval.fit(self.X_trainval_gzsl.T)

			self.X_train_gzsl = scaler_train.transform(self.X_train_gzsl.T).T
			self.X_val_gzsl = scaler_train.transform(self.X_val_gzsl.T).T
			
			self.X_trainval_gzsl = scaler_trainval.transform(self.X_trainval_gzsl.T).T
			self.X_test_seen = scaler_trainval.transform(self.X_test_seen.T).T
			self.X_test_unseen = scaler_trainval.transform(self.X_test_unseen.T).T

		if self.args.norm_type=='L2':
			self.X_train_gzsl = self.normalizeFeature(self.X_train_gzsl.T).T
			self.X_trainval_gzsl = self.normalizeFeature(self.X_trainval_gzsl.T).T
			# self.X_val = self.normalizeFeature(self.X_val.T).T
			# self.X_test = self.normalizeFeature(self.X_test.T).T

	def normalizeFeature(self, x):
	    # x = N x d (d:feature dimension, N:number of instances)
	    x = x + 1e-10
	    feature_norm = np.sum(x**2, axis=1)**0.5 # l2-norm
	    feat = x / feature_norm[:, np.newaxis]

	    return feat

	def find_compatible_y(self, X_n, W, y_n, sig):

		XW = np.dot(X_n, W)
		# Scale the projected vector
		XW = preprocessing.scale(XW)
		scores = np.zeros(sig.shape[1])
		scores[y_n] = 0.0
		gt_class_score = np.dot(XW, sig[:, y_n])
		
		for i in range(sig.shape[1]):
			if i!=y_n:
				scores[i] = self.args.margin + np.dot(XW, sig[:, i]) - gt_class_score

		return np.argmax(scores)

	def update_W(self, X, labels, sig, W, idx):
		
		for j in idx:
			X_n = X[:, j]
			y_n = labels[j]
			y = self.find_compatible_y(X_n, W, y_n, sig)
			
			if y!=y_n:
				Y = np.expand_dims(sig[:, y_n]-sig[:, y], axis=0)
				W += self.args.lr*np.dot(np.expand_dims(X_n, axis=1), Y)
		
		return W

	def fit_train(self):

		print('Training on train set...\n')

		best_val_acc = 0.0
		best_tr_acc = 0.0
		best_val_ep = -1
		best_tr_ep = -1
		
		rand_idx = np.arange(self.X_train_gzsl.shape[1])

		W = np.random.rand(self.X_train_gzsl.shape[0], self.train_sig.shape[0])
		W = self.normalizeFeature(W.T).T

		for ep in range(self.args.epochs):

			start = time.time()

			shuffle(rand_idx)

			W = self.update_W(self.X_train_gzsl, self.labels_train_gzsl, self.train_sig, W, rand_idx)
			
			val_acc = self.zsl_acc(self.X_val_gzsl, W, self.labels_val_gzsl, self.val_sig)
			tr_acc = self.zsl_acc(self.X_train_gzsl, W, self.labels_train_gzsl, self.train_sig)

			end = time.time()
			
			elapsed = end-start
			
			print('Epoch:{}; Train Acc:{}; Val Acc:{}; Time taken:{:.0f}m {:.0f}s\n'.format(ep+1, tr_acc, val_acc, elapsed//60, elapsed%60))
			
			if val_acc>best_val_acc:
				best_val_acc = val_acc
				best_val_ep = ep+1
			
			if tr_acc>best_tr_acc:
				best_tr_ep = ep+1
				best_tr_acc = tr_acc

			if ep+1-best_val_ep>self.args.early_stop:
				print('Early Stopping by {} epochs. Exiting...'.format(self.args.epochs-(ep+1)))
				break

		print('Best Val Acc:{} @ Epoch {}. Best Train Acc:{} @ Epoch {}\n'.format(best_val_acc, best_val_ep, best_tr_acc, best_tr_ep))
		
		return best_val_ep

	def fit_trainval(self):

		print('\nTraining on trainval set for GZSL...\n')

		best_tr_acc = 0.0
		best_tr_ep = -1
		
		rand_idx = np.arange(self.X_trainval_gzsl.shape[1])

		W = np.random.rand(self.X_trainval_gzsl.shape[0], self.trainval_sig.shape[0])
		W = self.normalizeFeature(W.T).T

		for ep in range(self.num_epochs_trainval):

			start = time.time()

			shuffle(rand_idx)

			W = self.update_W(self.X_trainval_gzsl, self.labels_trainval_gzsl, self.trainval_sig, W, rand_idx)
			
			tr_acc = self.zsl_acc(self.X_trainval_gzsl, W, self.labels_trainval_gzsl, self.trainval_sig)

			end = time.time()
			
			elapsed = end-start
			
			print('Epoch:{}; Trainval Acc:{}; Time taken:{:.0f}m {:.0f}s\n'.format(ep+1, tr_acc, elapsed//60, elapsed%60))
						
			if tr_acc>best_tr_acc:
				best_tr_ep = ep+1
				best_tr_acc = tr_acc
				best_W = np.copy(W)

		print('Best Trainval Acc:{} @ Epoch {}\n'.format(best_tr_acc, best_tr_ep))
		
		return best_W

	def zsl_acc(self, X, W, y_true, sig): # Class Averaged Top-1 Accuarcy

		XW = np.dot(X.T, W)# N x k
		dist = 1-spatial.distance.cdist(XW, sig.T, 'cosine')# N x C(no. of classes)
		predicted_classes = np.array([np.argmax(output) for output in dist])
		cm = confusion_matrix(y_true, predicted_classes)
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		acc = sum(cm.diagonal())/sig.shape[1]

		return acc

	def zsl_acc_gzsl(self, X, W, y_true, classes, sig): # Class Averaged Top-1 Accuarcy

		class_scores = np.matmul(np.matmul(X.T, W), sig) # N x Number of Classes
		y_pred = np.array([np.argmax(output)+1 for output in class_scores])

		per_class_acc = np.zeros(len(classes))

		for i in range(len(classes)):
			is_class = y_true==classes[i]
			per_class_acc[i] = ((y_pred[is_class]==y_true[is_class]).sum())/is_class.sum()
		
		return per_class_acc.mean()

	def evaluate(self):

		self.num_epochs_trainval = self.fit_train()

		best_W = self.fit_trainval()

		print('Testing...\n')

		acc_seen_classes = self.zsl_acc_gzsl(self.X_test_seen, best_W, self.labels_test_seen, self.test_classes_seen, self.test_sig)
		acc_unseen_classes = self.zsl_acc_gzsl(self.X_test_unseen, best_W, self.labels_test_unseen, self.test_classes_unseen, self.test_sig)
		HM = 2*acc_seen_classes*acc_unseen_classes/(acc_seen_classes+acc_unseen_classes)

		print('U:{}; S:{}; H:{}'.format(acc_unseen_classes, acc_seen_classes, HM))

if __name__ == '__main__':
	
	args = parser.parse_args()
	print('Dataset : {}\n'.format(args.dataset))
	
	clf = SJE(args)	
	clf.evaluate()
