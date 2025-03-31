import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

class LogRegCCD(): 
    def __init__(self, seed=42,): 
        self.optimized = False
        self.seed = seed


    def __sigmoid(self, x): 
            return 1 / (1 + np.exp(-x))
        
        
        

    def single_fit(self, X_train, y_train, lambda_=0.01, nr_iter=100, tol=1e-5): 
        def compute_cost(X_train, y_train, lambda_):
            n = len(y_train)
            y_dash = X_train @ self.beta + self.beta_zero
            p = self.__sigmoid(y_dash)
            cost = - (1/n) * np.sum(y_train * np.log(p) + (1 - y_train) * np.log(1 - p))
            reg_term = lambda_ * np.sum(np.abs(self.beta)) / n  # L1 regularization
            return cost + reg_term
        
        def soft_threshold(a, b):
            return np.sign(a) * np.maximum(np.abs(a) - b, 0)
                  
        def calculate_coordinate_descent(X_train, y_train, lambda_, j): 
            y_dash = X_train @ self.beta + self.beta_zero
            p = self.__sigmoid(y_dash)
            p = np.clip(p, 1e-5, 1 - 1e-5)
            w = p * (1 - p)
            z = y_dash + (y_train - p) / w
            # self.beta_zero = np.sum(w * z) / np.sum(w)

            residual = z - y_dash + self.beta[j] * X_train.iloc[:, j]
            st_nom = np.sum(w * X_train.iloc[:, j] * residual)

            st_denom = np.sum(w * X_train.iloc[:, j] ** 2)

            return w, z, soft_threshold(st_nom, lambda_)/st_denom
        
        np.random.seed(self.seed)
        g = X_train.shape[1]
        
        
        self.beta_zero = 0
        self.beta = np.zeros(g)
        self.lambda_ = lambda_

        self.iter_hist = []
        beta_old = np.ones(g)
        old_cost = np.inf


        for i in range(nr_iter): 
            for j in range(g): 
                if np.abs(self.beta[j] - beta_old[j]) < tol:
                    continue
                w, z, beta_j = calculate_coordinate_descent(X_train, y_train, lambda_, j)
                self.beta[j], beta_old[j] = beta_j, self.beta[j]
                self.beta_zero = np.sum(w * z) / np.sum(w)

            self.cost = compute_cost(X_train, y_train, lambda_)
                # Save iteration history
            self.iter_hist.append({
                "lambda": lambda_,
                "iteration": i,
                "cost": self.cost,
                "beta_zero": self.beta_zero,
                "beta": self.beta.copy()
            })
            # Convergence check
            if np.max(np.abs(self.beta - beta_old)) < tol:
                break
            if np.abs(self.cost-old_cost) < tol: 
                break
            old_cost = self.cost

        return self
        
    



    def validate(self, X_valid, y_valid, measure): 
        """
        Return score of measure of X_valid, y_valid

        Parameters
        ----------
        X_valid: validation dataframe

        y_valid: validation labels

        measure: measure to score the data        

        """
        # Get predicted probabilities
        y_prob = self.predict_proba(X_valid)

        # Convert probabilities to binary predictions 
        y_pred = (y_prob >= 0.5).astype(int)

        # Compute the performance measure
        score = measure(y_valid, y_pred)
        return score

    def predict_proba(self, X_test): 
        """
        Return probability of X_test. 

        Parameters
        ----------
        X_test: test dataframe

        """
        return self.__sigmoid(X_test.dot(self.beta) + self.beta_zero)
        

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, eps=0.001, K=10, measure=accuracy_score, nr_iter=100, tol=1e-5):
        """
        Fit weights based on X_train, y_train and X_valid, y_valid

        Parameters
        ----------
        X_test: train dataframe

        y_train: train labels
        
        X_valid: validation dataframe

        y_valid: validation labels

        K, eps: parameters to fit lambda

        measure: measure to score the data and select based weights based on X_valid, y_valid

        nr_iter, tol: parameters for single fit on given lambda


        """
        

        if X_valid is None and y_valid is None:
            X_valid = X_train
            y_valid = y_train

        n = len(y_train)

        inner_products = np.abs(np.dot(X_train.T, y_train)) / 4
        lambda_max = inner_products.max()
        lambda_min = eps * lambda_max
        
        lambdas = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_min), K))
        
        betas = []
        betas_zero = []
        costs = []
        scores = []
        full_history = [] 
        
        for lam in lambdas:
            self.single_fit(X_train, y_train, lambda_=lam, nr_iter=nr_iter, tol=tol)
            betas.append(self.beta.tolist())
            betas_zero.append(self.beta_zero)
            costs.append(self.cost)
            score = self.validate(X_valid, y_valid, measure)
            scores.append(score)
            full_history.extend(self.iter_hist) 

            
        
        p = X_train.shape[1]
        columns = [f"Beta_{i + 1}" for i in range(p)]
        self.coeffs_df = pd.DataFrame(betas, columns=columns)
        self.coeffs_df['Intercept'] = betas_zero
        self.coeffs_df['Lambda'] = lambdas
        self.coeffs_df['Cost'] = costs
        self.coeffs_df['ValidationScore'] = scores
        
        best_idx = np.argmax(scores)
        self.lambda_ = lambdas[best_idx]
        self.beta = betas[best_idx]
        self.beta_zero = betas_zero[best_idx]
        self.score = scores[best_idx]
        self.iter_hist = full_history[best_idx]
        self.full_history = pd.DataFrame(full_history)
        self.cost = costs[best_idx]
        return self

    def plot_coeff(self, filename):
        """
        Plot the coefficient values vs. log(Lambda)
        """
        
        plt.figure(figsize=(10, 6))
        log_lambda = np.log(self.coeffs_df['Lambda'])

        for col in self.coeffs_df.columns:
            if col not in ['Lambda', 'ValidationScore', 'Intercept', 'Cost']:
                plt.plot(log_lambda, self.coeffs_df[col], marker='o', color='blue', linewidth=1)
        
        plt.xlabel("log(Lambda)")
        plt.ylabel("Coefficient Value")
        plt.title("Coefficient values vs. log(Lambda)")
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()

    def plot_score(self, filename):
        """
        Runs fit_optimize_lambda with the provided measure, then plots the validation score vs. log(Lambda).
        """
      
        plt.figure(figsize=(10, 6))
        plt.plot(np.log(self.coeffs_df['Lambda']), self.coeffs_df['ValidationScore'],
                 marker='o', color='blue', linewidth=2)
        plt.xlabel("log(Lambda)")
        plt.ylabel("Validation Score")
        plt.title("Validation Score vs. log(Lambda)\nBest lambda: {:.4f} with score {:.4f}".format(self.lambda_, self.score))
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        
        return self.lambda_, self.score

