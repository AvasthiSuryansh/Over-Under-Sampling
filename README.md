# Over-Under-Sampling
The code performs Over and Under Sampling of a dataset using SMOTE and Nearmiss of imblearn

One thing to keep in mind is that the dataset should enough samples in majority or in minority for the usage of SMOTE, other wise we'll end up with the errors like this:
      ValueError: Expected n_neighbors <= n_samples,  but n_samples = 1, n_neighbors = 6
      
