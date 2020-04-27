import numpy as np
import argparse
from scipy import io

parser = argparse.ArgumentParser(description="ESZSL")

parser.add_argument('-data', '--dataset', help='choose between APY, AWA2, CUB, SUN', default='AWA2', type=str)
parser.add_argument('-alpha', '--alpha', default=0, type=int)
parser.add_argument('-gamma', '--gamma', default=0, type=int)

"""

Best Values of (Alpha, Gamma) found by validation & corr. test accuracies:

AWA2 -> (3, 0)  -> Seen : 0.8884 Unseen : 0.0404 HM : 0.0772
CUB  -> (3, -1) -> Seen : 0.6380 Unseen : 0.1263 HM : 0.2108
SUN  -> (3, 2)  -> Seen : 0.2841 Unseen : 0.1375 HM : 0.1853
APY  -> (3, -1) -> Seen : 0.8017 Unseen : 0.0241 HM : 0.0468

"""

class ESZSL():
	
	def __init__(self, args):

		self.args = args

		data_folder = '../datasets/'+args.dataset+'/'
		res101 = io.loadmat(data_folder+'res101.mat')
		att_splits=io.loadmat(data_folder+'att_splits.mat')

		trainval_loc = 'trainval_loc'
		test_seen_loc = 'test_seen_loc'
		test_unseen_loc = 'test_unseen_loc'

		feat = res101['features']
		# Shape -> (dxN)
		self.X_trainval = feat[:, np.squeeze(att_splits[trainval_loc]-1)]
		self.X_test_seen = feat[:, np.squeeze(att_splits[test_seen_loc]-1)]
		self.X_test_unseen = feat[:, np.squeeze(att_splits[test_unseen_loc]-1)]

		labels = res101['labels']
		self.labels_trainval = labels[np.squeeze(att_splits[trainval_loc]-1)]
		self.labels_test_seen = labels[np.squeeze(att_splits[test_seen_loc]-1)]
		self.labels_test_unseen = labels[np.squeeze(att_splits[test_unseen_loc]-1)]
		self.labels_test = np.concatenate((self.labels_test_seen, self.labels_test_unseen), axis=0)

		trainval_classes_seen = np.unique(self.labels_trainval)
		self.test_classes_seen = np.unique(self.labels_test_seen)
		self.test_classes_unseen = np.unique(self.labels_test_unseen)
		test_classes = np.unique(self.labels_test)

		i=0
		for labels in trainval_classes_seen:
			self.labels_trainval[self.labels_trainval == labels] = i    
			i+=1

		self.gt_trainval = np.zeros((self.labels_trainval.shape[0], len(trainval_classes_seen)))
		self.gt_trainval[np.arange(self.labels_trainval.shape[0]), np.squeeze(self.labels_trainval)] = 1

		sig = att_splits['att']
		# Shape -> (Number of attributes, Number of Classes)
		self.trainval_sig = sig[:, trainval_classes_seen-1]
		self.test_sig_seen = sig[:, self.test_classes_seen-1]
		self.test_sig_unseen = sig[:, self.test_classes_unseen-1]
		self.test_sig = sig[:, test_classes-1]

	def find_W(self, X, y, sig, alpha, gamma):

		part_0 = np.linalg.pinv(np.matmul(X, X.T) + (10**alpha)*np.eye(X.shape[0]))
		part_1 = np.matmul(np.matmul(X, y), sig.T)
		part_2 = np.linalg.pinv(np.matmul(sig, sig.T) + (10**gamma)*np.eye(sig.shape[0]))

		W = np.matmul(np.matmul(part_0, part_1), part_2) # Feature Dimension x Number of Attributes

		return W

	def zsl_acc(self, X, W, y_true, classes): # Class Averaged Top-1 Accuarcy

		class_scores = np.matmul(np.matmul(X.T, W), self.test_sig) # N x Number of Classes
		y_pred = np.array([np.argmax(output)+1 for output in class_scores])
		y_true = np.squeeze(y_true)

		per_class_acc = np.zeros(len(classes))

		for i in range(len(classes)):
			is_class = y_true==classes[i]
			per_class_acc[i] = ((y_pred[is_class]==y_true[is_class]).sum())/is_class.sum()
		
		return per_class_acc.mean()

	def evaluate(self):

		alpha, gamma = self.args.alpha, self.args.gamma

		best_W = self.find_W(self.X_trainval, self.gt_trainval, self.trainval_sig, alpha, gamma) # combine train and val

		acc_seen_classes = self.zsl_acc(self.X_test_seen, best_W, self.labels_test_seen, self.test_classes_seen)
		acc_unseen_classes = self.zsl_acc(self.X_test_unseen, best_W, self.labels_test_unseen, self.test_classes_unseen)
		HM = 2*acc_seen_classes*acc_unseen_classes/(acc_seen_classes+acc_unseen_classes)

		print('U:{}; S:{}; H:{}'.format(acc_unseen_classes, acc_seen_classes, HM))

args = parser.parse_args()
print('Dataset : {}\n'.format(args.dataset))
model = ESZSL(args)
model.evaluate()
