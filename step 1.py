import pandas as pd

# 1. Column ke naam (Features) load karna
# features.txt me 561 sensor readings ke naam hain 
features_df = pd.read_csv('UCI HAR Dataset/features.txt', sep='\s+', header=None, names=['Index', 'Feature_Name'])
feature_columns = features_df['Feature_Name'].tolist()

# 2. Training Data load karna (Jis par model seekhega)
# X_train me actual sensor data hai
X_train = pd.read_csv('UCI HAR Dataset/train/X_train.txt', sep='\s+', header=None, names=feature_columns)

# y_train me activity ke labels hain (1 se 6 tak)
y_train = pd.read_csv('UCI HAR Dataset/train/y_train.txt', sep='\s+', header=None, names=['Activity_Label'])

# 3. Testing Data load karna (Jis par model ka test hoga)
X_test = pd.read_csv('UCI HAR Dataset/test/X_test.txt', sep='\s+', header=None, names=feature_columns)
y_test = pd.read_csv('UCI HAR Dataset/test/y_test.txt', sep='\s+', header=None, names=['Activity_Label'])

# 4. Activity ke asli naam load karna (Walking, Sitting, etc.)
activity_labels = pd.read_csv('UCI HAR Dataset/activity_labels.txt', sep='\s+', header=None, names=['Label', 'Activity_Name'])

# Data ko print karke dekhna
print("Training Data  size:", X_train.shape)
print("Labels size:", y_train.shape)
print(X_train.head())
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("\nModel train hona shuru ho gaya hai. Isme 1-2 minute lag sakte hain...")

# 1. Random Forest Model initialize karna
# n_estimators=100 ka matlab hai yeh 100 chote-chote 'decision trees' milakar ek strong model banayega
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 2. Model ko Training Data par sikhana (Training phase)
# .values.ravel() use kiya hai taaki y_train ka shape sahi rahe
rf_model.fit(X_train, y_train.values.ravel())

# 3. Test Data par Model ka test lena (Testing phase)
print("Training puri hui! Ab test data par check kar rahe hain...")
y_pred = rf_model.predict(X_test)

# 4. Accuracy check karna (Kitne % answers sahi diye)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel ki Accuracy: {accuracy * 100:.2f}%")

# (Optional) Har activity ki alag accuracy dekhne ke liye
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=activity_labels['Activity_Name'].tolist()))

# 5. Model ko save karna taaki frontend (Streamlit) me use kar sakein
joblib.dump(rf_model, 'har_model.pkl')
print("\nModel successfully 'har_model.pkl' ke naam se save ho gaya hai!")
print("\nHar alag activity ki ek test CSV file bana rahe hain...")

# Har activity (1 se 6) ke liye loop chalana
for i in range(1, 7):
    # Us activity ki sabse pehli row dhundhna
    index = y_test[y_test['Activity_Label'] == i].index[0]
    single_row = X_test.iloc[[index]]
    
    # Activity ka asli naam lena
    actual_activity = activity_labels.iloc[i - 1]['Activity_Name']
    
    # File save karna
    file_name = f"test_data_actual_{actual_activity}.csv"
    single_row.to_csv(file_name, index=False, header=False)
    print(f"'{file_name}' save ho gayi hai.")