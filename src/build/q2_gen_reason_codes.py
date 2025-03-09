import shap
shap.initjs()

def get_top_reasons(explainer, X, shap_values, i, n=3):
    shap_instance = shap_values[i].values
    features = X.columns
    
    top = np.argsort(np.abs(shap_instance))[-n:]
    filtered_shap = shap_instance[top]
    filtered_features = features[top]
    feature_labels = [
        f"{feature}_HIGH" if shap_value > 0 else f"{feature}_LOW"
        for feature, shap_value in zip(filtered_features, filtered_shap)
    ]

    return feature_labels

def get_reason_codes(explainer, X, shap_values):
    reason_codes_by_consumer = []
    for i in range(len(X)):
        top_3_features = get_top_reasons(explainer, X, shap_values, i)
        reason_codes_by_consumer.append(list(top_3_features))
    reason_codes_by_consumer = np.array(reason_codes_by_consumer)
    return reason_codes_by_consumer

def plot_reason_codes_distribution(reason_codes_by_consumer):
    reason_codes = reason_codes_by_consumer.ravel()
    unique_reason_codes, reason_codes_counts = np.unique(reason_codes, return_counts=True)
    unique_reason_codes = unique_reason_codes[np.argsort(-reason_codes_counts)][:25]
    reason_codes_counts = reason_codes_counts[np.argsort(-reason_codes_counts)][:25]
    reason_codes_props = reason_codes_counts[np.argsort(-reason_codes_counts)][:25] / reason_codes_counts.sum()
    
    plt.figure(figsize=(10,5))
    plt.bar(unique_reason_codes, reason_codes_props, color='skyblue', edgecolor='black')
    plt.xticks(rotation=45, ha="right")
    
    for i, (prop, count) in enumerate(zip(reason_codes_props, reason_codes_counts)):
        plt.text(i, prop + 0.001, str(count), ha="center", fontsize=10)
    
    plt.title("Distribution of Reason Codes")
    plt.xlabel("Reason Code")
    plt.ylabel("Proportion")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('../../q2_result/reason_codes_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()