import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import pickle

INPUT_SIZE = (224, 224)
BATCH_SIZE = 32
N_SPLITS = 10
TREES_CONFIG = [100, 150, 200]

BASE_DIR = './imagens'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')
TEST_DIR = os.path.join(BASE_DIR, 'test')

CLASSES = ['bacterial', 'normal', 'viral']
NUM_CLASSES = len(CLASSES)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

def load_images_from_folder(folder, datagen=None, max_samples=None):
    images = []
    labels = []
    
    for class_name in CLASSES:
        class_dir = os.path.join(folder, class_name)
        class_label = CLASSES.index(class_name)
        samples_loaded = 0
        
        for filename in tqdm(os.listdir(class_dir), desc=f'Carregando {class_name} de {folder}'):
            if max_samples and samples_loaded >= max_samples:
                break
                
            img_path = os.path.join(class_dir, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(INPUT_SIZE)
                img_array = np.array(img)
                
                if datagen:
                    img_array = datagen.random_transform(img_array)
                else:
                    img_array = preprocess_input(img_array)
                
                images.append(img_array)
                labels.append(class_label)
                samples_loaded += 1
            except Exception as e:
                print(f"Erro ao carregar {img_path}: {e}")
    
    return np.array(images), np.array(labels)

def get_feature_extractor():
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(*INPUT_SIZE, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)
    return model

def extract_features(model, images):
    features = model.predict(images, batch_size=BATCH_SIZE, verbose=0)
    return features

def calculate_metrics(y_true, y_pred, y_proba):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_proba, multi_class='ovr')
    }
    
    cm = confusion_matrix(y_true, y_pred)
    
    for i, class_name in enumerate(CLASSES):
        tp = cm[i,i]
        fn = cm[i,:].sum() - tp
        fp = cm[:,i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics[f'sensitivity_{class_name}'] = sensitivity
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics[f'specificity_{class_name}'] = specificity
    
    return metrics

def run_cross_validation(features, labels, n_estimators):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight={0: 1, 1: 1, 2: 3},
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_val)
        y_proba = rf.predict_proba(X_val)
        
        metrics = calculate_metrics(y_val, y_pred, y_proba)
        metrics['fold'] = fold + 1
        metrics['n_estimators'] = n_estimators
        results.append(metrics)
    
    return pd.DataFrame(results)

print("Carregando imagens de treino com aumento de dados...")
train_images, train_labels = load_images_from_folder(TRAIN_DIR, train_datagen)
print("Carregando imagens de validação...")
val_images, val_labels = load_images_from_folder(VAL_DIR)
print("Carregando imagens de teste...")
test_images, test_labels = load_images_from_folder(TEST_DIR)

print("\nCarregando modelo DenseNet121...")
feature_extractor = get_feature_extractor()

print("\nExtraindo características das imagens...")
train_features = extract_features(feature_extractor, train_images)
val_features = extract_features(feature_extractor, val_images)
test_features = extract_features(feature_extractor, test_images)

all_features = np.concatenate([train_features, val_features])
all_labels = np.concatenate([train_labels, val_labels])

all_results = []
for n_est in TREES_CONFIG:
    print(f"\nExecutando {N_SPLITS}-fold CV com {n_est} árvores...")
    results = run_cross_validation(all_features, all_labels, n_est)
    all_results.append(results)

df_results = pd.concat(all_results)
mean_results = df_results.groupby('n_estimators').mean().reset_index()

plt.figure(figsize=(12, 8))

available_cols = mean_results.columns.tolist()
sensitivity_cols = [f'sensitivity_{c}' for c in CLASSES]
sensitivity_cols = [col for col in sensitivity_cols if col in available_cols]

specificity_cols = [f'specificity_{c}' for c in CLASSES]
specificity_cols = [col for col in specificity_cols if col in available_cols]

if sensitivity_cols:
    mean_results['mean_sensitivity'] = mean_results[sensitivity_cols].mean(axis=1)
    plt.subplot(2, 2, 3)
    plt.plot(mean_results['n_estimators'], mean_results['mean_sensitivity'], marker='o')
    plt.title('Sensitivity (média) por número de árvores')
    plt.xlabel('Número de árvores')
    plt.ylabel('Sensitivity')

if specificity_cols:
    mean_results['mean_specificity'] = mean_results[specificity_cols].mean(axis=1)
    plt.subplot(2, 2, 4)
    plt.plot(mean_results['n_estimators'], mean_results['mean_specificity'], marker='o')
    plt.title('Specificity (média) por número de árvores')
    plt.xlabel('Número de árvores')
    plt.ylabel('Specificity')

plt.tight_layout()
plt.savefig('random_forest_metrics.png')
plt.show()

print("\nAvaliando no conjunto de teste com 200 árvores...")

best_rf = RandomForestClassifier(
    n_estimators=200,
    class_weight={0: 1, 1: 1, 2: 3},
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
best_rf.fit(all_features, all_labels)

test_preds = best_rf.predict(test_features)
test_proba = best_rf.predict_proba(test_features)

test_metrics = calculate_metrics(test_labels, test_preds, test_proba)

print("\nMétricas no conjunto de teste:")
print(f"► Acurácia: {test_metrics['accuracy']:.4f}")
print(f"► AUC: {test_metrics['auc']:.4f}\n")

print("Métricas por classe:")
for class_name in CLASSES:
    print(f"\nClasse {class_name.upper()}:")
    print(f"  - Sensitivity: {test_metrics[f'sensitivity_{class_name}']:.4f}")
    print(f"  - Specificity: {test_metrics[f'specificity_{class_name}']:.4f}")

mean_results.to_csv('cross_validation_results.csv', index=False)
print("\nResultados salvos em 'cross_validation_results.csv'")

with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(best_rf, f)
print("Modelo Random Forest salvo como 'random_forest_model.pkl'")