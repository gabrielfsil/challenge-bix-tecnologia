from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def train_model(X_train, X_test, y_train, y_test):
    print("--------------------------------------------------------------------------------")
    print("Training model with Logistic Regression...")
    log_reg = LogisticRegression(max_iter=10000, solver='saga')

    log_reg.fit(X_train, y_train['class'])

    y_pred = log_reg.predict(X_test)

    accuracy_log_reg = accuracy_score(y_test, y_pred)
    precision_log_reg = precision_score(y_test, y_pred, average='macro')
    sensitivity_log_reg = recall_score(y_test, y_pred, average='macro')
    conf_matrix_log_reg = confusion_matrix(y_test, y_pred)

    
    print("Test Metrics:")
    print(f"Accuracy: {accuracy_log_reg:.2f}")
    print(f"Precision: {precision_log_reg:.2f}")
    print(f"Sensitivity (Recall): {sensitivity_log_reg:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix_log_reg)
    print("--------------------------------------------------------------------------------")

    return log_reg