import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# إنشاء بيانات تجريبية
project_data = pd.DataFrame({
    "Budget": np.random.randint(5000, 100000, 100),
    "Deadline_Days": np.random.randint(30, 365, 100),
    "Risk_Level": np.random.choice([0, 1, 2], 100),
    "Status": np.random.choice([0, 1, 2], 100)
})

# تقسيم البيانات
X = project_data[["Budget", "Deadline_Days", "Risk_Level"]]
y = project_data["Status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب النموذج
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# حفظ النموذج
joblib.dump(model, "models/project_classifier.pkl")
