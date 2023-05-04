import pandas as pd
import numpy as np
import math
import scipy
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.sparse.linalg import eigs

class JDA1:
    def __init__(self, n_iter = 10, mu = 1, neighbors = 2, gamma = 1, top_k=2, final_classifier = None, final_dict={}, print=True, print_final =True):
        self.n_iter = n_iter
        self.mu = mu
        self.neighbors = neighbors
        self.gamma = gamma
        self.top_k = top_k
        self.f_params = final_dict
        self.print = print
        self.print_final = print_final
        if final_classifier == 'knn':
            self.f_clf = KNeighborsClassifier(**self.f_params)
        elif final_classifier == 'rf':
            self.f_clf = RandomForestClassifier(**self.f_params)
        elif final_classifier == 'svm':
            self.f_clf = SVC(**self.f_params)
        elif final_classifier == 'lr':
            self.f_clf = LogisticRegression(**self.f_params)
        elif final_classifier is None:
            self.f_clf = None
        self.train_df = None
        # storing MMD matrices from 1 -> C
        self.M_dict = {}
    
    
    def fit(self, Xs, y_s, Xt, y_t):
        # transform source and target datasets separately.
        X_s = pd.DataFrame(StandardScaler().fit_transform(Xs))
        X_t = pd.DataFrame(StandardScaler().fit_transform(Xt))
        
        self.len_s = len(X_s)
        self.len_t = len(X_t)
        self.TOTAL_LEN = len(X_s) + len(X_t)

        # the centering matrix H in the algorihtm
        self.H = np.identity(n = self.TOTAL_LEN) - 1/(self.TOTAL_LEN) * np.ones((self.TOTAL_LEN, self.TOTAL_LEN))
        
        self.num_classes = len(y_s.unique())
        self.labels = list(y_s.unique())

        # compute the MMD matrix M0
        self.make_M_0()

        for l in self.labels:
            self.M_dict[l] = np.zeros((self.TOTAL_LEN, self.TOTAL_LEN))
        
        for i in range(self.n_iter):
            # learn the mapping
            K, A = self.learn_mapping(X_s, X_t, Ms = list(self.M_dict.values()))
            
            # compute the coordinates of the data points on the new data space
            Z = K.dot(A)
            # Z = StandardScaler().fit_transform(Z)
            # self.Z = Z
            S_new = Z[:len(X_s), :] # new source dataset
            # S_new = StandardScaler().fit_transform(S_new)
            T_new = Z[len(X_s):, :] # new target dataset
            # T_new = StandardScaler().fit_transform(T_new)

            clf = KNeighborsClassifier(n_neighbors=self.neighbors).fit(S_new, y_s)
            
            y_pred = clf.predict(T_new)
            if self.print:
                print("[%i/%i] Accuracy: %0.4f" % (i+1, self.n_iter, accuracy_score(y_t, y_pred)))

            new_y = pd.Series(list(y_pred))
            # remake MMD matrices
            for l in self.labels:
                self.M_dict[l] = self.make_M_c(y_s, new_y, l)
        
        if self.f_clf is not None:
            self.f_clf.fit(S_new, y_s)
            self.f_clf_pred = self.f_clf.predict(T_new)
            self.final_acc = self.f_clf.score(T_new, y_t)
        else: 
            self.f_clf = clf
            self.f_clf_pred = y_pred
            self.final_acc =  accuracy_score(y_t, y_pred)

        if self.print_final: 
            print("Final accuracy: ", self.final_acc)
        self.A = A
        return Z
    
    # make M0 matrix
    def make_M_0(self):
        e = np.vstack((1 / self.len_s * np.ones((self.len_s, 1)), -1 / self.len_t * np.ones((self.len_t, 1))))
        M_0 = e * e.T
        assert (len(M_0) == self.TOTAL_LEN)
        self.M0 = M_0
    
    # for each class label, compute its own MMD matrix
    def make_M_c(self, y_s, y_t, c):
        ns_c = len(y_s[y_s == c])   
        nt_c = len(y_t[y_t == c])       
        class_bool_s = np.array(y_s == c).reshape(-1, 1)
        class_bool_t = np.array(y_t == c).reshape(-1, 1)

        if nt_c == 0:
            e = np.vstack((1 / ns_c * class_bool_s, -1 * class_bool_t))
        else:
            e = np.vstack((1 / ns_c * class_bool_s, -1 / nt_c * class_bool_t))
        M = e * e.T
        assert (len(M) == self.TOTAL_LEN)
        return M

    def learn_mapping(self, X_s, X_t, Ms):
        X = pd.concat([X_s, X_t], ignore_index = True)
        if self.train_df is None:
            self.train_df = X
        M = self.M0.copy()
        for m in Ms:
            M += m
        M /= np.linalg.norm(M, 'fro')
        self.M = M
        # compute kernel matrix
        K = rbf_kernel(X, None, gamma = self.gamma)


        # compute the left-side matrix
        left_mat =  (K @ (M) @ (K.T)) +  self.mu * np.identity(self.TOTAL_LEN)

        # compute the right-side matrix
        right_mat = K @ self.H @ K.T

        J = np.dot(np.linalg.inv(left_mat), right_mat)

        # eigenvector decomposition as solution to trace minimization
        _, A_ = eigs(J, k=self.top_k)

        # transformation/embedding matrix
        A = np.real(A_)

        return K, A
    
    # predict new incoming data points -- needs fixing.
    def predict(self, test_X, test_y = None):
        test_X2 = pd.DataFrame(StandardScaler().fit_transform(test_X))
        # compute kernel matrix on test set
        K = rbf_kernel(test_X2, self.train_df, gamma = self.gamma)
        assert(K.shape == (len(test_X), len(self.train_df)))
        Z = K.dot(self.A)
        # Z = StandardScaler().fit_transform(Z)
        if test_y is None:
            return self.f_clf.predict(Z)
        else:
            return self.f_clf.predict(Z), self.f_clf.score(Z, test_y)
    
