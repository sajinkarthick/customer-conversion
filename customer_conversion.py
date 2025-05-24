from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

# Sample setup
numeric_features = ['session_duration', 'page_views', 'clicks']
categorical_features = ['device', 'referrer', 'browser']

# Preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# Pipeline
clf_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestClassifier(class_weight='balanced'))
])

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y_classification, test_size=0.2, random_state=42)
clf_pipeline.fit(X_train, y_train)

# Evaluate
y_pred = clf_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, clf_pipeline.predict_proba(X_test)[:, 1]))
