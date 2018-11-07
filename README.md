# PCA_SVM

#In this exercise, we perform a PCA on a face recognition function, divide our dataset of names of presidents and associated faces formed by pixels. One of the sub-datasets will be the training dataset and once we train the model, we apply it and visualize how faces were classified.

Exercise done together with Alvaro Gutierrez (alvarogutiierrez@gmail.com) and Andrea Clavijo Ricaldi (andrea.clavijo.r@gmail.com)

As an example of SVM and PCA in action, you will solve the facial recognition problem. We will use the Labeled Faces in the Wild dataset, which consists of several thousand collated photos of various public figures 3,000 pixels each. A fetcher for the dataset is built into Scikit-Learn:

Try to plot the faces from faces.images matrix and faces.target_names as labels. TIP: use subplots and imshow functions from matplotlib

Make pipeline of SVM and RandomizedPCA model using sklearn library (make_pipeline command).TIP: use nonlinear Gaussian kernel in SVM (rbf) and number of PCA components (try 50 and 150).

For the sake of testing our classifier output, you have to will split the data into a training and testing set. TIP: use training_test_split function from _sklearn.crossvalidation

Perform a grid search cross-validation to explore combinations of parameters. Here we will adjust C (which controls the margin hardness) and gamma (which controls the size of the radial basis function kernel) in SVM, to find the best model. TIP: use GridSearchCV from _sklearn.gridsearch, use following values for 'svnc': [1, 5, 10, 50] and 'svcgamma: [0.0001, 0.0005, 0.001, 0.005]. Fit the training data to find out the parameters.

Print best parameters from grid search. If The optimal values fall fell at the edges, we would want to expand the grid to make sure we have found the true optimum.

Use best estimator (bestestimator function) from grid search to predict labels for test sample generated in 3). TIP: use predict function

Use few test images to check the fit accuracy and plot images with estimator assigned labels like in 1).

Each image contains [62×47] or nearly 3,000 pixels. We could proceed by simply using each pixel value as a feature, but often it is more effective to use some sort of preprocessor to extract more meaningful features; here we will use a principal component analysis to extract 150 fundamental components to feed into our support vector machine classifier. We can do this most straightforwardly by packaging the preprocessor and the classifier into a single pipeline using make_pipeline function

