import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from tkinter import Tk
from tkinter.filedialog import askopenfilename


data = pd.read_csv('C:\\Users\\DELL\\Downloads\\synthetic_student_data.csv')

# Encode categorical variables
encoders = {}
categorical_columns = ['Gender', 'SES', 'Extracurricular_Participation', 'Academic_Support', 'Enrolled', 'Graduated']

for column in categorical_columns:
    encoders[column] = LabelEncoder()
    data[column] = encoders[column].fit_transform(data[column])
#Define features and target variable
features = ['Age', 'Gender', 'SES', 'High_School_GPA', 'SAT_Score', 'Academic_Failures', 
            'Attendance_Percentage', 'Extracurricular_Participation', 'Academic_Support', 'Online_Learning_Hours']
X = data[features]
y = data['Enrolled']

# Standardize numerical features
scaler = StandardScaler()
X[['Age', 'High_School_GPA', 'SAT_Score', 'Attendance_Percentage', 'Online_Learning_Hours']] = scaler.fit_transform(
    X[['Age', 'High_School_GPA', 'SAT_Score', 'Attendance_Percentage', 'Online_Learning_Hours']]
)

#Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


param_grid = {
    'n_estimators': [50, 100, 200],        
    'max_depth': [None, 10, 20, 30],       
    'min_samples_leaf': [1, 2, 4],       
    'class_weight': ['balanced', None]     
}


rf_model = RandomForestClassifier(random_state=42)


grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)


best_rf_model = grid_search.best_estimator_


y_pred = best_rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


def feature_importance_report(model, feature_names):
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    
    print("\nTop Factors Influencing Enrollment Predictions:")
    print(feature_importance.head(5))  # Display the top 5 features for clarity


feature_importance_report(best_rf_model, features)

# Save the best model to the Desktop
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
model_filename = os.path.join(desktop_path, 'best_rf_model.pkl')
joblib.dump(best_rf_model, model_filename)
print(f"Model saved to: {model_filename}")


def predict_enrollment(student_data):
    student_df = pd.DataFrame([student_data])
    

    for column in categorical_columns[:-2]: 
        if column in student_df.columns:
            if column in encoders:
                known_classes = encoders[column].classes_.tolist()
                student_df[column] = student_df[column].apply(lambda x: x if x in known_classes else known_classes[0])
                student_df[column] = encoders[column].transform(student_df[column])
    
 
    student_df[['Age', 'High_School_GPA', 'SAT_Score', 'Attendance_Percentage', 'Online_Learning_Hours']] = scaler.transform(
        student_df[['Age', 'High_School_GPA', 'SAT_Score', 'Attendance_Percentage', 'Online_Learning_Hours']]
    )
    
    
    prediction = best_rf_model.predict(student_df[features])
    return "Enrolled" if prediction[0] == 1 else "Not Enrolled"

def analyze_student_data():
   
    Tk().withdraw() 
    csv_path = askopenfilename(filetypes=[("CSV files", "*.csv")], title="Select the student data CSV file")
    
    if not csv_path:
        print("No file selected. Exiting.")
        return

    # Load the new student data
    student_data = pd.read_csv(csv_path)
    
# Encode categorical columns in the new data
    for column in categorical_columns[:-2]:
        if column in student_data.columns:
            if column in encoders:
                known_classes = encoders[column].classes_.tolist()
                student_data[column] = student_data[column].apply(lambda x: x if x in known_classes else known_classes[0])
                student_data[column] = encoders[column].transform(student_data[column])
    
# Scale numerical features in the new data
    student_data[['Age', 'High_School_GPA', 'SAT_Score', 'Attendance_Percentage', 'Online_Learning_Hours']] = scaler.transform(
        student_data[['Age', 'High_School_GPA', 'SAT_Score', 'Attendance_Percentage', 'Online_Learning_Hours']]
    )
    

    student_data['Enrollment_Prediction'] = best_rf_model.predict(student_data[features])
    student_data['Probability'] = best_rf_model.predict_proba(student_data[features])[:, 1]
    
   
    student_data['Support_Needed'] = student_data['Probability'].apply(lambda x: 'Needs Support' if x < 0.6 else 'Likely to Enroll')
    
# Output results
    print("\nEnrollment Predictions and Support Suggestions:")
    print(student_data[['Enrollment_Prediction', 'Probability', 'Support_Needed']])
    
# Save results to a CSV file
    output_file = os.path.join(desktop_path, 'student_analysis_results.csv')
    student_data.to_csv(output_file, index=False)
    print(f"Analysis results saved to: {output_file}")


analyze_student_data()
