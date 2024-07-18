from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)

    accuracy_log_reg_under = accuracy_score(y_val, y_pred)
    precision_log_reg_under = precision_score(y_val, y_pred, average='macro')
    sensitivity_log_reg_under = recall_score(y_val, y_pred, average='macro')
    conf_matrix_log_reg_under = confusion_matrix(y_val, y_pred)

    print("--------------------------------------------------------------------------------")
    print("Validation Metrics:")
    print(f"Accuracy: {accuracy_log_reg_under:.2f}")
    print(f"Precision: {precision_log_reg_under:.2f}")
    print(f"Sensitivity (Recall): {sensitivity_log_reg_under:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix_log_reg_under)
    print("--------------------------------------------------------------------------------")