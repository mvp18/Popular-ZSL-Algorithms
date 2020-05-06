import numpy as np
import argparse
from scipy import io, spatial, linalg
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description="GZSL for SAE")

parser.add_argument('-data', '--dataset', help='choose between APY, AWA2, AWA1, CUB, SUN', default='AWA2', type=str)
parser.add_argument('-mode', '--mode', help='train/test, if test set alpha, gamma to best values below', default='train', type=str)
parser.add_argument('-ld1', '--ld1', default=5, help='best value for F-->S during test, lower bound of variation interval during train', type=float)
parser.add_argument('-ld2', '--ld2', default=5, help='best value for S-->F during test, upper bound of variation interval during train', type=float)

"""
Range of Lambda for Validation:

AWA1 -> 0.05-5
AWA2 -> 0.05-1.6
CUB  -> 5-5000 for [F-->S] and 0.05-5 for [S-->F]
SUN  -> 0.005-5
APY  -> 0.05-5

Best Value of Lambda found by validation & corr. test accuracies:

AWA1 -> Seen : 0.8052 Unseen : 0.0529 HM : 0.0992 @ 3.2 [F-->S] Seen : 0.8293 Unseen : 0.1472 HM : 0.2500 @ 0.8 [S-->F]		   				
AWA2 -> Seen : 0.8142 Unseen : 0.05 HM : 0.0942 @ 0.8 [F-->S] Seen : 0.8720 Unseen : 0.1286 HM : 0.2241 @ 0.2 [S-->F]
CUB  -> Seen : 0.4988 Unseen : 0.1386 HM : 0.2169 @ 80 [F-->S] Seen : 0.5702 Unseen : 0.1572 HM : 0.2464 @ 0.2 [S-->F]
SUN  -> Seen : 0.2469 Unseen : 0.1681 HM : 0.2 @ 0.32 [F-->S] Seen : 0.3120 Unseen : 0.1903 HM : 0.2364 @ 0.08 [S-->F]
APY  -> Seen : 0.2797 Unseen : 0.0828 HM : 0.1277 @ 0.16 [F-->S] Seen : 0.5662 Unseen : 0.0948 HM : 0.1624 @ 2.56 [S-->F]

"""
class SAE():
	
	def __init__(self, args):

		self.args = args

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

		# Train and Val are first separated to find the best hyperparamters on val and then to finally use them to train on trainval set.

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

		self.train_att_gzsl = np.zeros((self.X_train_gzsl.shape[1], self.train_sig.shape[0]))
		for i in range(self.train_att_gzsl.shape[0]):
		    self.train_att_gzsl[i] = self.train_sig.T[self.labels_train_gzsl[i]]

		self.trainval_att_gzsl = np.zeros((self.X_trainval_gzsl.shape[1], self.trainval_sig.shape[0]))
		for i in range(self.trainval_att_gzsl.shape[0]):
		    self.trainval_att_gzsl[i] = self.trainval_sig.T[self.labels_trainval_gzsl[i]]

		self.X_train_gzsl = self.normalizeFeature(self.X_train_gzsl.T).T
		self.X_trainval_gzsl = self.normalizeFeature(self.X_trainval_gzsl.T).T

	def normalizeFeature(self, x):
	    # x = N x d (d:feature dimension, N:number of instances)
	    x = x + 1e-10
	    feature_norm = np.sum(x**2, axis=1)**0.5 # l2-norm
	    feat = x / feature_norm[:, np.newaxis]

	    return feat

	def find_W(self, X, S, ld):

		# INPUTS:
	    # X: d x N - data matrix
	    # S: Number of Attributes (k) x N - semantic matrix
	    # ld: regularization parameter
	    #
	    # Return :
	    # 	W: kxd projection matrix

	    A = np.dot(S, S.T)
	    B = ld*np.dot(X, X.T)
	    C = (1+ld)*np.dot(S, X.T)
	    W = linalg.solve_sylvester(A, B, C)
	    
	    return W

	def fit(self):

		print('Training...\n')

		best_acc_F2S = 0.0
		best_acc_S2F = 0.0

		ld = self.args.ld1

		while (ld<=self.args.ld2):
			
			W = self.find_W(self.X_train_gzsl, self.train_att_gzsl.T, ld)
			acc_F2S, acc_S2F = self.zsl_acc(self.X_val_gzsl, W, self.labels_val_gzsl, self.val_sig)
			print('Val Acc --> [F-->S]:{} [S-->F]:{} @ lambda = {}\n'.format(acc_F2S, acc_S2F, ld))
			
			if acc_F2S>best_acc_F2S:
				best_acc_F2S = acc_F2S
				lambda_F2S = ld

			if acc_S2F>best_acc_S2F:
				best_acc_S2F = acc_S2F
				lambda_S2F = ld
			
			ld*=0.2

		print('\nBest Val Acc --> [F-->S]:{} @ lambda = {} [S-->F]:{} @ lambda = {}\n'.format(best_acc_F2S, lambda_F2S, best_acc_S2F, lambda_S2F))
		
		return lambda_F2S, lambda_S2F

	def zsl_acc_gzsl(self, X, W, y_true, classes, sig, mode): # Class Averaged Top-1 Accuarcy

		if mode=='F2S':
			# [F --> S], projecting data from feature space to semantic space
			F2S = np.dot(X.T, self.normalizeFeature(W).T)# N x k
			dist = 1-spatial.distance.cdist(F2S, sig.T, 'cosine')# N x C(no. of classes)

		if mode=='S2F':
			# [S --> F], projecting from semantic to visual space
			S2F = np.dot(sig.T, self.normalizeFeature(W))# N x k
			dist = 1-spatial.distance.cdist(X.T, self.normalizeFeature(S2F), 'cosine')# N x C(no. of classes)

		y_pred = np.array([np.argmax(output)+1 for output in dist])

		per_class_acc = np.zeros(len(classes))

		for i in range(len(classes)):
			is_class = y_true==classes[i]
			per_class_acc[i] = ((y_pred[is_class]==y_true[is_class]).sum())/is_class.sum()
		
		return per_class_acc.mean()

	def zsl_acc(self, X, W, y_true, sig): # Class Averaged Top-1 Accuarcy

		# [F --> S], projecting data from feature space to semantic space
		F2S = np.dot(X.T, self.normalizeFeature(W).T)# N x k
		dist_F2S = 1-spatial.distance.cdist(F2S, sig.T, 'cosine')# N x C(no. of classes)
		# [S --> F], projecting from semantic to visual space
		S2F = np.dot(sig.T, self.normalizeFeature(W))
		dist_S2F = 1-spatial.distance.cdist(X.T, self.normalizeFeature(S2F), 'cosine')
		
		pred_F2S = np.array([np.argmax(y) for y in dist_F2S])
		pred_S2F = np.array([np.argmax(y) for y in dist_S2F])
		
		cm_F2S = confusion_matrix(y_true, pred_F2S)
		cm_F2S = cm_F2S.astype('float')/cm_F2S.sum(axis=1)[:, np.newaxis]

		cm_S2F = confusion_matrix(y_true, pred_S2F)
		cm_S2F = cm_S2F.astype('float')/cm_S2F.sum(axis=1)[:, np.newaxis]
		
		acc_F2S = sum(cm_F2S.diagonal())/sig.shape[1]
		acc_S2F = sum(cm_S2F.diagonal())/sig.shape[1]

		# acc = acc_F2S if acc_F2S>acc_S2F else acc_S2F

		return acc_F2S, acc_S2F

	def evaluate(self, ld1, ld2):

		print('Testing...\n')

		best_W_F2S = self.find_W(self.X_trainval_gzsl, self.trainval_att_gzsl.T, ld1)
		best_W_S2F = self.find_W(self.X_trainval_gzsl, self.trainval_att_gzsl.T, ld2)
		
		# F-->S
		acc_F2S_seen = self.zsl_acc_gzsl(self.X_test_seen, best_W_F2S, self.labels_test_seen, self.test_classes_seen, self.test_sig, 'F2S')
		acc_F2S_unseen = self.zsl_acc_gzsl(self.X_test_unseen, best_W_F2S, self.labels_test_unseen, self.test_classes_unseen, self.test_sig, 'F2S')
		HM_F2S = 2*acc_F2S_seen*acc_F2S_unseen/(acc_F2S_seen+acc_F2S_unseen)
		
		# S-->F
		acc_S2F_seen = self.zsl_acc_gzsl(self.X_test_seen, best_W_S2F, self.labels_test_seen, self.test_classes_seen, self.test_sig, 'S2F')
		acc_S2F_unseen = self.zsl_acc_gzsl(self.X_test_unseen, best_W_S2F, self.labels_test_unseen, self.test_classes_unseen, self.test_sig, 'S2F')
		HM_S2F = 2*acc_S2F_seen*acc_S2F_unseen/(acc_S2F_seen+acc_S2F_unseen)

		print('[F-->S]\n')
		print('U:{}; S:{}; H:{}\n'.format(acc_F2S_unseen, acc_F2S_seen, HM_F2S))

		print('[S-->F]\n')
		print('U:{}; S:{}; H:{}'.format(acc_S2F_unseen, acc_S2F_seen, HM_S2F))

if __name__ == '__main__':

	args = parser.parse_args()
	print('Dataset : {}\n'.format(args.dataset))
	
	clf = SAE(args)
	
	if args.mode=='train':
		args.ld1, args.ld2 = clf.fit()
	
	clf.evaluate(args.ld1, args.ld2)
