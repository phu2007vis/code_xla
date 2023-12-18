import numpy as np
from knn.main import KNN,generate_data

def calculate_precision_recall(y_true, y_pred):
    # True positive, false positive, false negative counts
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    # Calculate precision and recall
    # != 0 else 0 tranh truong hop chia cho 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    
    return precision, recall



X_train, X_test, Y_train, Y_test,name_class = generate_data(r"knn/Image")
models = [ KNN(X_train,Y_train,name_class,k_neighbor = k) for k  in [3,5,7]]

precisions  = []
recals = []

y_predicts = [[model.predict_single(X)[0] for X in X_test] for model in models]


print(y_predicts)
value = [calculate_precision_recall(Y_test, y_predict) for y_predict in y_predicts]

# print("Precision:", precision)
# print("Recall:", recall)
