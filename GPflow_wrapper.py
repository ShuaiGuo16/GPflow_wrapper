#####################################################
# /  _____/\______   \_   _____/|  |   ______  _  __
#/   \  ___ |     ___/|    __)  |  |  /  _ \ \/ \/ /
#\    \_\  \|    |    |     \   |  |_(  <_> )     / 
# \______  /|____|    \___  /   |____/\____/ \/\_/  
#        \/               \/                        
# Content: Wrapper class for GPflow
# Author: Shuai Guo
# Date: Feb, 2024
#####################################################

import numpy as np
from scipy.stats import qmc
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import copy
import logging
import datetime
import os

import gpflow
import gpflow.utilities.model_utils as util
import tensorflow as tf
import logging


class GPflowWrapper:
    """
    Wrapper class for GPflow.
    """

    def __init__(self, kernel="squared_exponential", mean_func="const", n_restarts=5):
        """
        Initialize the GPflowWrapper class.

        Args:
        - kernel (str): The kernel function to use. Default is "squared_exponential".
        - mean_func (str): The mean function to use. Default is "const".
        - n_restarts (int): The number of random restarts for hyperparameter optimization. Default is 5.
        """
        self.kernel = kernel
        self.mean_func = mean_func
        self.n_restarts = n_restarts
        self.GP = None

        # Setup logger
        self.logger = LoggerUtil.__call__().getLogger()


    def fit(self, X_train, y_train, noise_variance=1e-5, device="CPU:0", verbose=True):
        """
        Fit the GP model.

        Args:
        - X_train (np.ndarray): The training input data.
        - y_train (np.ndarray): The training output data.
        - verbose (bool): Whether to print out the optimization process. Default is True.
        """
        self.dim = X_train.shape[1]
        
        models = []
        log_likelihoods = []

        # Generate initial guesses for length scale
        self._init_kernel_params(y_train)
        kernel_list = self._build_kernels()

        if verbose:
            self.logger.info(f"Training dataset with {self.dim} features and {X_train.shape[0]} samples.")
            self.logger.info(f"{self.kernel} kernel and {self.mean_func} mean function are selected for the GP model.")
            self.logger.info(f"GP model will be trained with {self.n_restarts} random restarts:")

        with tf.device(device):

            for i, kernel in enumerate(kernel_list):
                if verbose:
                    self.logger.info(f"Performing {i+1}-th optimization:")

                # Set up the model
                if noise_variance is not None:
                    model = gpflow.models.GPR(
                        (X_train, y_train.reshape(-1, 1)),
                        kernel=kernel,
                        mean_function=self._build_mean_function(),
                        noise_variance=noise_variance
                    )

                    # Avoid training the likelihood noise variance
                    gpflow.set_trainable(model.likelihood.variance, False)

                else:
                    # Trainable likelihood noise variance
                    model = gpflow.models.GPR(
                        (X_train, y_train.reshape(-1, 1)),
                        kernel=kernel,
                        mean_function=self._build_mean_function(),
                    )

                # Training
                opt = gpflow.optimizers.Scipy()
                opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))

                # Record keeping
                models.append(model)
                log_likelihoods.append(model.log_marginal_likelihood().numpy())
                if verbose:
                    self.logger.info(f"Log likelihood: {log_likelihoods[-1]}")

        # Select the best model
        best_model_index = np.argmax(log_likelihoods)
        self.GP = models[best_model_index]
        if verbose:
            self.logger.info(f"GP model training completed!")




    def _init_kernel_params(self, y):
        """
        Initialize the kernel parameters.

        Args:
        - y (np.ndarray): The training output data.
        """

        if self.kernel == "squared_exponential":
            # Default range
            lb, ub = -2, 2

            # Latin Hypercube sampling (in log space)
            lhd = qmc.LatinHypercube(d=self.dim, seed=42).random(self.n_restarts)
            lhd = (ub-lb)*lhd + lb

            # Convert back to the original scale
            self.length_scales = 10**lhd

            # Init process variance
            self.variances = np.var(y)*np.ones(self.n_restarts)
        
        else:
            error_message = f"{self.kernel} kernel not supported yet!"
            self.logger.error(error_message)
            raise KeyError(f"{self.kernel} kernel not supported yet!")

        


    def _build_kernels(self):
        """
        Build a list of kernels. Each kernel has a different initial hyperparameters. 
        """
        kernel_list = []

        if self.kernel == "squared_exponential":
            for i in range(self.n_restarts):
                kernel_list.append(gpflow.kernels.SquaredExponential(
                    variance=self.variances[i], 
                    lengthscales=self.length_scales[i]))
            
        else:
            error_message = f"{self.kernel} kernel not supported yet!"
            self.logger.error(error_message)
            raise KeyError(f"{self.kernel} kernel not supported yet!")
        
        return kernel_list
        


    def _build_mean_function(self):
        """
        Build the mean function.
        """
        if self.mean_func == "const":
            return gpflow.functions.Polynomial(0)
        
        elif isinstance(self.mean_func, dict):
            if self.mean_func["type"] == "multi-fidelity":
                return copy.deepcopy(self.mean_func["value"])
            
            else:
                error_message = f"{self.mean_func['type']} mean function not supported yet!"
                self.logger.error(error_message)
                raise KeyError(error_message)

        else:
            error_message = f"{self.mean_func} mean function not supported yet!"
            self.logger.error(error_message)
            raise KeyError(error_message)



    def predict(self, X_test, full_cov=False):
        """
        Predict the mean and variance of the test data.

        Args:
        - X_test (np.ndarray): The test input data.
        - full_cov (bool): Whether to return the full covariance matrix. Default is False.

        Returns:
        - f_mean (np.ndarray): The mean of the predicted output.
        - f_var (np.ndarray): The variance of the predicted output.
        """

        f_mean, f_var = self.GP.predict_f(X_test, full_cov=full_cov)
        f_mean = f_mean.numpy().flatten()
        f_var = f_var.numpy().flatten()

        return f_mean, f_var
    


    def LOOCV(self):
        """
        Calculate leave-one-out cross-validation error
        """
        X, y = self.GP.data
        
        # Extract kernel & mean function
        K = self.GP.kernel(X)
        ks = util.add_likelihood_noise_cov(K, self.GP.likelihood, X)
        m = self.GP.mean_function(X)

        # Cholesky decomposition
        L = tf.linalg.cholesky(ks)
        
        # Calculate LOOCV error
        Q = tf.linalg.cholesky_solve(L, y-m)
        LOO = tf.reshape(Q, [-1])/tf.linalg.diag_part(tf.linalg.cholesky_solve(L, tf.eye(tf.shape(X)[0], dtype=K.dtype)))

        return LOO
    


    def acquisition_AL(self, candidate, batch_mode=False, batch_size=None):
        """
        Compute the acquisition function value for a given candidate point.

        Args:
        - candidate (numpy.ndarray): The candidate point for which to compute the acquisition function value.
        - diagnose (bool, optional): Flag indicating whether to return additional diagnostic information. Defaults to False.

        Returns:
        - expected_error (numpy.ndarray): The acquisition function value (expected prediction error) for the candidate point.
        - indices (int or numpy.ndarray): The index of the selected candidate point(s).
        """
        
        # Compute cross-validation error
        LOO = self.LOOCV().numpy().flatten()

        # Compute prediction variance
        f_mean, f_var = self.GP.predict_f(candidate, full_cov=False)
        f_mean = f_mean.numpy().flatten()
        f_var = f_var.numpy().flatten()

        # Calculate bias
        bias = np.zeros(candidate.shape[0])
        for i in range(candidate.shape[0]):
            # Determine bias
            X, _ = self.GP.data
            distance_sqr = np.sum((candidate[[i],:]-X.numpy())**2, axis=1)
            closest_index = np.argmin(distance_sqr.flatten())
            bias[i] = LOO[closest_index]**2

        # Calculate expected prediction error
        expected_error = bias + f_var

        # Acquisition index
        if batch_mode:
            # Batch selection mode
            expected_error_normalied = MinMaxScaler().fit_transform(expected_error.reshape(-1, 1))
            indices = self._select_diverse_batch(candidate, expected_error_normalied.flatten(), batch_size=batch_size)
        
        else:
            # Single point selection mode
            indices = np.argmax(expected_error)

        return expected_error, indices

        


    def _select_diverse_batch(self, samples, acq, batch_size=5, pre_filter=False):
        """
        Select a diverse batch of samples based on the acquisition function values.

        Args:
        - samples (np.ndarray): The candidate samples.
        - acq (np.ndarray): The acquisition function values for the candidate samples.
        - batch_size (int): The number of samples to select. Default is 5.
        - pre_filter (float): The quantile threshold for pre-filtering the samples. Default is False.
        """
        
        if pre_filter:
            thred = np.quantile(acq, pre_filter)
            filtered_indices = np.arange(len(samples))[acq>thred]
            samples = samples[acq>thred]
            acq = acq[acq>thred]
        
        else:
            filtered_indices = np.arange(len(samples))
            
        # Perform weighted K-means clustering on the samples
        kmeans = KMeans(n_clusters=batch_size, n_init=10, random_state=0).fit(samples, sample_weight=acq)
        cluster_labels = kmeans.labels_

        # Find the highest acquisition value sample in each cluster
        selected_indices = []
        for cluster_idx in range(batch_size):
            cluster_indices = np.where(cluster_labels == cluster_idx)[0]
            cluster_acquisition_values = acq[cluster_indices]
            best_index_in_cluster = cluster_indices[np.argmax(cluster_acquisition_values)]
            selected_indices.append(best_index_in_cluster)

        return filtered_indices[selected_indices]
    



class SingletonType(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Calls the singleton logger
        """
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class LoggerUtil(object, metaclass=SingletonType):
    _logger = None

    def __init__(self):
        self._logger = logging.getLogger("defcon_log")
        self._logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s \t [%(levelname)s | %(filename)s:%(lineno)s] > %(message)s')

        now = datetime.datetime.now()
        dirname = "./log"

        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        fileHandler = logging.FileHandler(dirname + "/log_" + now.strftime("%Y-%m-%d")+".log")

        streamHandler = logging.StreamHandler()

        fileHandler.setFormatter(formatter)
        streamHandler.setFormatter(formatter)

        self._logger.addHandler(fileHandler)
        self._logger.addHandler(streamHandler)

    def getLogger(self):
        """
        Gets the singleton logger
        """
        return self._logger