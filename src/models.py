import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        
    def fit(self, X, y):
        n_features = X.shape[1]
        
        I = np.eye(n_features)

        I[0, 0] = 0 

        xtx = X.T @ X
        xty = X.T @ y
        
        matrix_to_invert = xtx + self.alpha * I
        
        self.weights = np.linalg.solve(matrix_to_invert, xty)
        
    def predict(self, X):
        return X @ self.weights
    
class LassoRegression:
    def __init__(self, alpha=0.1, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.cost_history = []

    def _soft_threshold(self, rho, lam):
        if rho < - lam:
            return rho + lam
        elif rho > lam:
            return rho - lam
        else:
            return 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        
        x_sq = np.sum(X**2, axis=0)
        
        for iteration in range(self.max_iter):
            max_w_change = 0
            
            for j in range(n_features):
                old_w = self.weights[j]
                
                if j == 0:
                    y_pred_no_j = X @ self.weights - X[:, j] * old_w
                    residual = y - y_pred_no_j
                    rho = np.dot(X[:, j], residual)
                    self.weights[j] = rho / x_sq[j] 
                else:
                    y_pred_no_j = X @ self.weights - X[:, j] * old_w
                    residual = y - y_pred_no_j
                    rho = np.dot(X[:, j], residual)
                    
                    self.weights[j] = self._soft_threshold(rho, self.alpha) / x_sq[j]
                
                max_w_change = max(max_w_change, abs(self.weights[j] - old_w))
            
            if max_w_change < self.tol:
                break
                
    def predict(self, X):
        return X @ self.weights

def rmse_score(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def train_test_split_numpy(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    
    indices = np.random.permutation(X.shape[0])
    test_samples = int(X.shape[0] * test_size)
    
    test_idx = indices[:test_samples]
    train_idx = indices[test_samples:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def k_fold_split(n_samples, k=5, random_state=42):
    np.random.seed(random_state)
    
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    folds = np.array_split(indices, k)
    
    splits = []
    for i in range(k):
        val_idx = folds[i]
        
        train_folds = [folds[j] for j in range(k) if j != i]
        train_idx = np.concatenate(train_folds)
        
        splits.append((train_idx, val_idx))
        
    return splits

def grid_search_cv(model_class, X, y, param_grid, k=5):
    best_score = -np.inf
    best_params = None
    results = [] 

    splits = k_fold_split(X.shape[0], k=k)
        
    for alpha in param_grid:
        fold_scores = []
        
        for train_idx, val_idx in splits:
            X_train_fold, y_train_fold = X[train_idx], y[train_idx]
            X_val_fold, y_val_fold = X[val_idx], y[val_idx]
            
            model = model_class(alpha=alpha)
            model.fit(X_train_fold, y_train_fold)
            
            y_pred = model.predict(X_val_fold)
            score = r2_score(y_val_fold, y_pred)
            fold_scores.append(score)
        
        avg_score = np.mean(fold_scores)
        results.append((alpha, avg_score))
        #in ga để debug thui
        #print(f"Alpha: {alpha:.4f} | Avg R2: {avg_score:.4f}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_params = alpha
            
    print(f"\nBest Alpha: {best_params} | Best R2: {best_score:.4f}")
    return best_params, results