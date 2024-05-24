import pandas as pd

# Import libs to solve classification task
from catboost import CatBoostClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Make prediction
def make_pred(dt, path_to_file):

    print('Importing pretrained model...')
    # Import model
    model = CatBoostClassifier()
    model.load_model('./models/catboost_model.cbm')

    # Make submission dataframe
    submission = pd.DataFrame({
        'client_id':  pd.read_csv(path_to_file, encoding='utf-8')['client_id'],
        'preds': (model.predict(dt)) * 1
    })
    print('Prediction complete!')

    return submission, model, model.predict_proba(dt)[:, 1]

def get_top_features(model, feature_names, top_n=5):
    importances = model.get_feature_importance()
    feature_importances = {feature: importance for feature, importance in zip(feature_names, importances)}
    sorted_features = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)[:top_n])
    print(sorted_features)
    return sorted_features

def plot_prediction_distribution(predictions, file_name='prediction_distribution.png'):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(predictions, shade=True, color='blue')
    plt.title('Density Plot of Predicted Scores')
    plt.xlabel('Predicted Score')
    plt.ylabel('Density')
    plt.savefig(file_name)
    plt.close()
