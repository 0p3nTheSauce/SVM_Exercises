import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
np.random.seed(4242)
#-------------------------------------------------------------------------------------------------------------------
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay
# -------------------------------------------------------------------------------------------------------------------
# Exercise 1
n_samples = 500
n_features = 2
X1 = np.random.rand(n_samples, n_features)
y1 = np.ones((n_samples, 1))
idx_neg = (X1[:, 0] - 0.5) ** 2 + (X1[:, 1] - 0.5) ** 2 < 0.03
y1[idx_neg] = 0
y1_flat = y1.ravel()

plt.figure(figsize=(10, 6))
#plt.scatter(np.reshape(X1[:, 0],-1), np.reshape(X1[:, 1],-1), c=np.reshape(y1,-1),s=100)
plt.scatter(X1[:, 0], X1[:,1], c=y1_flat, s=100)

#plt.show()
# Code solution 1 here:
def bestParamsInitial(X, y):
    #Because the computation takes so long this has some parameter ranges narrowed down
    #Should be used to find initial parameters, then use bestParamsNumeric to increase accuracy
    
    #Define pipeline 
    pipeline = Pipeline([
       ('classifier', SVC())
    ])
    #Hyperparameter grid
    c_values = [0.01, 0.1, 1]
    degree_values = np.arange(2, 4, 1)
    coef0_values = [-1, 0, 1]
    tol_values = [1e-3, 1e-2]
    param_grid = {
       'classifier__C': c_values,
       'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],# (took out linear becasue definitely not)
       'classifier__degree': degree_values,
       'classifier__gamma': ['scale', 'auto'],
       'classifier__coef0': coef0_values, #only significant in poly and sigmoid kernels
       'classifier__shrinking': [True, False],
       'classifier__tol': tol_values,
       'classifier__class_weight': [None, 'balanced'],
       'classifier__decision_function_shape': ['ovo', 'ovr']
    }
    #perform grid search with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    #Get best parameters for SVM classifier 
    best_params = grid_search.best_params_
    print("Best Parameters: ", best_params)
    best_params_dictionary = {
        'C': best_params['classifier__C'],
        'class_weight': best_params['classifier__class_weight'],
        'coef0': best_params['classifier__coef0'],
        'decision_function_shape': best_params['classifier__decision_function_shape'],
        'degree': best_params['classifier__degree'],
        'gamma': best_params['classifier__gamma'],
        'kernel': best_params['classifier__kernel'],
        'shrinking': best_params['classifier__shrinking'],
        'tol': best_params['classifier__tol']
    }
    return best_params_dictionary
    
def bestParamsFurther(X, y, c_values=[0.01, 0.1, 1], kernel=['linear', 'poly', 'rbf', 'sigmoid'],
                      degree_values=np.arange(2, 4, 1),gamma=['scale', 'auto'],
                      coef0_values=[-1, 0, 1],shrinking=[True, False],
                      tol_values=[1e-3, 1e-2], class_weight=[None, 'balanced'],
                      decision_function_shape=['ovo', 'ovr']):
    #Because some parameters didnt seem to change these were left out for increased performance
    #Define pipeline 
    pipeline = Pipeline([
       ('classifier', SVC())
    ])
    #Hyperparameter grid
    param_grid = {
       'classifier__C': c_values,
       'classifier__kernel': kernel,
       'classifier__degree': degree_values,
       'classifier__gamma': gamma,
       'classifier__coef0': coef0_values, #only significant in poly and sigmoid kernels
       'classifier__shrinking': shrinking,
       'classifier__tol': tol_values,
       'classifier__class_weight': class_weight,
       'classifier__decision_function_shape': decision_function_shape
    }
    #perform grid search with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    #Get best parameters for SVM classifier 
    best_params = grid_search.best_params_
    print("Best Parameters: ", best_params)
    best_params_dictionary = {
        'C': best_params['classifier__C'],
        'class_weight': best_params['classifier__class_weight'],
        'coef0': best_params['classifier__coef0'],
        'decision_function_shape': best_params['classifier__decision_function_shape'],
        'degree': best_params['classifier__degree'],
        'gamma': best_params['classifier__gamma'],
        'kernel': best_params['classifier__kernel'],
        'shrinking': best_params['classifier__shrinking'],
        'tol': best_params['classifier__tol']
    }
    return best_params_dictionary   
    
#best_params = bestParamsInitial(X1, y1_flat)
#Best Parameters:  {'classifier__C': 1, 'classifier__class_weight': None, 'classifier__coef0': 1, 'classifier__decision_function_shape': 'ovo', 'classifier__degree': 3,
# 'classifier__gamma': 'scale', 'classifier__kernel': 'poly', 'classifier__shrinking': True, 'classifier__tol': 0.001}

#this was from a run with all the parameters of Initial with the extended parameters of Further and took a long time to run
#Best Parameters:  {'classifier__C': 2, 'classifier__class_weight': None, 'classifier__coef0': 1, 'classifier__decision_function_shape': 'ovo', 'classifier__degree': 4, 'classifier__gamma': 'scale', 'classifier__kernel': 'poly', 'classifier__shrinking': True, 'classifier__tol': 0.0001}
c_values = [0.01, 0.1, 1, 2, 3]
degree_values = np.arange(2, 6, 1)
tol_values = [1e-5, 1e-4, 1e-3, 1e-2]
best_params = bestParamsFurther(X1, y1_flat, c_values=c_values, degree_values=degree_values, coef0_values=[1], tol_values=tol_values, 
                                class_weight=[None], decision_function_shape=['ovo'], gamma=['scale'], kernel=['poly'], shrinking=[True])

def plotGraphs(X, y, best_params, x_min, x_max, y_min, y_max):
    #Function to plot the graph of the input values with the decision boundaries 
    
    #Initialise classifier
    best_SVC = SVC(**best_params)
    #Fit 
    best_SVC.fit(X, y)
    y_pred = best_SVC.predict(X)
    #Evaluate
    accuracy = accuracy_score(y, y_pred)
    print("Accuracy: ", accuracy)
    
    #Settings for plotting
    _, ax = plt.subplots(figsize=(10, 6))
    ax.set(title=f"Accuracy: {accuracy}", xlim=(x_min, x_max), ylim=(y_min, y_max))
    #plot decision boundary and margins
    parameters = {"estimator": best_SVC, "X": X, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **parameters,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        **parameters,    
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )
    #plot samples by color and add legend
    ax.scatter(X[:, 0], X[:, 1], c=y, s=100, edgecolors="k")
    plt.show()
    
x_min, x_max, y_min, y_max = -0.05, 1.05, -0.05, 1.05
plotGraphs(X1, y1_flat, best_params, x_min, x_max, y_min, y_max)

#breakpoint()
# -------------------------------------------------------------------------------------------------------------------
# Exercise 2
X2 = np.random.rand(n_samples, n_features)
y2 = np.ones((n_samples, 1))
idx_neg = (X2[:, 0] < 0.5) * (X2[:, 1] < 0.5) + (X2[:, 0] > 0.5) * (X2[:, 1] > 0.5)
y2[idx_neg] = 0
y2_flat = y2.ravel()
plt.figure(figsize=(10, 6))
plt.scatter(np.reshape(X2[:, 0],-1), np.reshape(X2[:, 1],-1), c=np.reshape(y2,-1),s=100)

# Code solution 2 here:
#best_params = bestParamsInitial(X2, y2_flat)
#Best Parameters:  {'classifier__C': 1, 'classifier__class_weight': None, 'classifier__coef0': 1, 'classifier__decision_function_shape': 'ovo', 'classifier__degree': 3, 'classifier__gamma': 'scale', 'classifier__kernel': 'poly', 'classifier__shrinking': True, 'classifier__tol': 0.001}
c_values = [0.01, 0.1, 1, 2, 3, 4]
degree_values = np.arange(2, 8, 1)
tol_values = [1e-08, 1e-09]
best_params = bestParamsFurther(X2, y2_flat, c_values=c_values, degree_values=degree_values, coef0_values=[1], tol_values=tol_values, 
                                class_weight=[None], decision_function_shape=['ovo'], gamma=['scale'], kernel=['poly'], shrinking=[True])
#Best Parameters:  {'classifier__C': 3, 'classifier__class_weight': None, 'classifier__coef0': 1, 'classifier__decision_function_shape': 'ovo', 'classifier__degree': 5,
#'classifier__gamma': 'scale', 'classifier__kernel': 'poly', 'classifier__shrinking': True, 'classifier__tol': 1e-05}
#Best Parameters:  {'classifier__C': 2, 'classifier__class_weight': None, 'classifier__coef0': 1, 'classifier__decision_function_shape': 'ovo', 'classifier__degree': 6, 
# 'classifier__gamma': 'scale', 'classifier__kernel': 'poly', 'classifier__shrinking': True, 'classifier__tol': 1e-06}
#Accuracy:  0.994
#Best Parameters:  {'classifier__C': 2, 'classifier__class_weight': None, 'classifier__coef0': 1, 'classifier__decision_function_shape': 'ovo', 'classifier__degree': 6, 
# 'classifier__gamma': 'scale', 'classifier__kernel': 'poly', 'classifier__shrinking': True, 'classifier__tol': 1e-07}
#Accuracy:  0.994
#Best Parameters:  {'classifier__C': 2, 'classifier__class_weight': None, 'classifier__coef0': 1, 'classifier__decision_function_shape': 'ovo', 'classifier__degree': 6, 
# 'classifier__gamma': 'scale', 'classifier__kernel': 'poly', 'classifier__shrinking': True, 'classifier__tol': 1e-08}
#Accuracy:  0.994
x_min, x_max, y_min, y_max = -0.05, 1.05, -0.05, 1.05
plotGraphs(X2, y2_flat, best_params, x_min, x_max, y_min, y_max)

# breakpoint()
# -------------------------------------------------------------------------------------------------------------------
# Exercise 3
rho_pos = np.random.rand(n_samples // 2, 1) / 2.0 + 0.5
rho_neg = np.random.rand(n_samples // 2, 1) / 4.0
rho = np.vstack((rho_pos, rho_neg))
phi_pos = np.pi * 0.75 + np.random.rand(n_samples // 2, 1) * np.pi * 0.5
phi_neg = np.random.rand(n_samples // 2, 1) * 2 * np.pi
phi = np.vstack((phi_pos, phi_neg))
X3 = np.array([[r * np.cos(p), r * np.sin(p)] for r, p in zip(rho, phi)])
y3 = np.vstack((np.ones((n_samples // 2, 1)), np.zeros((n_samples // 2, 1))))
y3_flat = y3.ravel()
X3 = np.squeeze(X3)

plt.figure(figsize=(10, 6))
plt.scatter(np.reshape(X3[:, 0],-1), np.reshape(X3[:, 1],-1), c=np.reshape(y3,-1),s=100)
#plt.show()

# Code solution 3 here:

#best_params = bestParamsInitial(X3, y3_flat)
#Best Parameters:  {'classifier__C': 0.01, 'classifier__class_weight': None, 'classifier__coef0': -1, 'classifier__decision_function_shape': 'ovo', 'classifier__degree': 2, 
# 'classifier__gamma': 'scale', 'classifier__kernel': 'rbf', 'classifier__shrinking': True, 'classifier__tol': 0.001}
#Accuracy:  1.0

c_values = [0.001, 0.01, 0.1]
degree_values = np.arange(2, 4, 1)
tol_values = [1e-2, 1e-3, 1e-4]
coef0_values = [-2, -1, 0]
best_params = bestParamsFurther(X3, y3_flat, c_values=c_values, class_weight=[None], coef0_values=coef0_values, decision_function_shape=['ovo'], degree_values=degree_values,
                                gamma=['scale'], kernel=['rbf'], shrinking=[True], tol_values=tol_values)
#Best Parameters:  {'classifier__C': 0.001, 'classifier__class_weight': None, 'classifier__coef0': -2, 'classifier__decision_function_shape': 'ovo', 'classifier__degree': 2, 'classifier__gamma': 'scale', 'classifier__kernel': 'rbf', 'classifier__shrinking': True, 'classifier__tol': 0.01}
#Accuracy:  1.0

x_min, x_max, y_min, y_max = -1.05, 0.7, -0.7, 0.75
plotGraphs(X3, y3_flat, best_params, x_min, x_max, y_min, y_max)
#print(X3.shape)
#breakpoint()
# -------------------------------------------------------------------------------------------------------------------
# Exercise 4
# rho_pos = np.linspace(0, 2, n_samples // 2)
# rho_neg = np.linspace(0, 2, n_samples // 2) + 0.5
# rho = np.vstack((rho_pos, rho_neg))
# phi_pos = 2 * np.pi * rho_pos
# phi = np.vstack((phi_pos, phi_pos))
# X4 = np.array([[r * np.cos(p), r * np.sin(p)] for r, p in zip(rho, phi)])
# y4 = np.vstack((np.ones((n_samples // 2, 1)), np.zeros((n_samples // 2, 1))))

# Generate rho_pos and rho_neg with n_samples // 2 samples each
rho_pos = np.linspace(0, 2, n_samples // 2)
rho_neg = np.linspace(0, 2, n_samples // 2) + 0.5

# Stack rho_pos and rho_neg vertically to create rho
rho = np.vstack((rho_pos, rho_neg))

# Generate phi_pos based on rho_pos
phi_pos = 2 * np.pi * rho_pos

# Stack phi_pos twice vertically to create phi
phi = np.vstack((phi_pos, phi_pos))

# Combine rho and phi to create X4 with n_samples samples
X4 = np.array([[r * np.cos(p), r * np.sin(p)] for r, p in zip(rho.ravel(), phi.ravel())])

# Generate y4 with n_samples samples, where the first half are ones and the second half are zeros
y4 = np.vstack((np.ones((n_samples // 2, 1)), np.zeros((n_samples // 2, 1))))
y4_flat = y4.ravel()
# Reshape X4 to have a shape of (n_samples, 2)
X4 = X4.reshape(n_samples, 2)

print(X4.shape)
plt.figure(figsize=(10, 6))
plt.scatter(np.reshape(X4[:, 0],-1), np.reshape(X4[:, 1],-1), c=np.reshape(y4,-1),s=100)
#plt.show()

# Code solution 4 here:
#best_params = bestParamsInitial(X4, y4_flat)
#Best Parameters:  {'classifier__C': 0.1, 'classifier__class_weight': None, 'classifier__coef0': -1, 'classifier__decision_function_shape': 'ovo', 'classifier__degree': 2,
# 'classifier__gamma': 'auto', 'classifier__kernel': 'sigmoid', 'classifier__shrinking': True, 'classifier__tol': 0.01}
#Accuracy:  0.624
degree_values = np.arange(2, 4, 1)
coef0_values = [-2, -1, 0]
tol_values = [0.1, 0.01, 0.001]
best_params = bestParamsFurther(X4, y4_flat, c_values=[0.1], class_weight=[None], coef0_values=coef0_values, decision_function_shape=['ovo'], degree_values=[2],
                                gamma=['auto'], kernel=['sigmoid'], shrinking=[True], tol_values=tol_values)
x_min, x_max, y_min, y_max = -2.2, 3.0, -2.5, 2.0
plotGraphs(X4, y4_flat, best_params, x_min, x_max, y_min, y_max)
#breakpoint()
# -------------------------------------------------------------------------------------------------------------------
# Exercise 5
X5, y5 = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=42)
plt.figure(figsize=(10, 6))
plt.scatter(np.reshape(X5[:, 0],-1), np.reshape(X5[:, 1],-1), c=np.reshape(y5,-1),s=100)
#plt.show()

#best_params = bestParamsInitial(X5, y5)
#Best Parameters:  {'classifier__C': 0.1, 'classifier__class_weight': None, 'classifier__coef0': 1, 'classifier__decision_function_shape': 'ovo', 'classifier__degree': 3, 
# 'classifier__gamma': 'scale', 'classifier__kernel': 'poly', 'classifier__shrinking': True, 'classifier__tol': 0.001}
#Accuracy:  1.0
best_params = bestParamsFurther(X5, y5, c_values=[0.1], class_weight=[None], coef0_values=[1], decision_function_shape=['ovo'], degree_values=[3],
                                gamma=['scale'], kernel=['poly'], shrinking=[True], tol_values=[0.001])
x_min, x_max, y_min, y_max = -1.25, 2.25, -0.75, 1.25
plotGraphs(X5, y5, best_params, x_min, x_max, y_min, y_max)

# Code solution 5 here:
