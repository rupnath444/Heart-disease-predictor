from predict import HeartDiseasePredictor

def get_input(prompt, valid_type=float, valid_values=None):
    while True:
        try:
            user_input = valid_type(input(prompt))
            if valid_values is not None and user_input not in valid_values:
                print(f"Please enter one of the following valid options: {valid_values}")
                continue
            return user_input
        except ValueError:
            print(f"Invalid input. Please enter a valid {valid_type.__name__}.")

def get_user_data():
    print("🏥 Heart Disease Risk Assessment")
    print("="*40)
    print("Please enter your health information:")
    print("-"*40)

    print("\n📋 Personal Information:")
    print("Age: Your current age in years")
    age = get_input("Age: ", float)

    print("\n👤 Sex:")
    print("ℹ️ Men often have higher heart disease risk at younger ages; women's risk rises after menopause.")
    print("0 = Female")
    print("1 = Male")
    sex = get_input("Enter choice (0 or 1): ", int, [0,1])

    print("\n💔 Chest Pain Type:")
    print("ℹ️ Chest pain caused by reduced blood flow to the heart.")
    print("0 = Typical Angina: crushing chest pain during exercise; classic blocked arteries sign.")
    print("1 = Atypical Angina: unusual chest pain, could indicate heart problems.")
    print("2 = Non-Anginal Pain: chest pain unrelated to heart, e.g. muscle strain.")
    print("3 = Asymptomatic: no chest pain but possible silent heart disease.")
    cp = get_input("Chest Pain Type (0-3): ", int, [0,1,2,3])

    print("\n🩺 Resting Blood Pressure:")
    print("ℹ️ Pressure when heart is at rest. Normal <120 mmHg, high >140 mmHg.")
    trestbps = get_input("Resting Blood Pressure (mmHg, e.g. 120): ", float)

    print("\n🧪 Cholesterol Level:")
    print("ℹ️ Fatty substance that can clog arteries.")
    print("Normal <200 mg/dl; borderline 200-239; high >240 mg/dl.")
    chol = get_input("Cholesterol Level (mg/dl, e.g. 200): ", float)

    print("\n🍬 Fasting Blood Sugar:")
    print("ℹ️ High sugar damages vessels; diabetics have 2-4x heart disease risk.")
    print("0 = No (≤120 mg/dl)")
    print("1 = Yes (>120 mg/dl)")
    fbs = get_input("Fasting Blood Sugar > 120 mg/dl? (0 or 1): ", int, [0,1])

    print("\n📈 Resting ECG (Electrocardiogram):")
    print("0 = Normal rhythm")
    print("1 = ST-T Wave Abnormality (minor changes)")
    print("2 = Left Ventricular Hypertrophy (enlarged heart chamber)")
    restecg = get_input("Resting ECG results (0-2): ", int, [0,1,2])

    print("\n💓 Maximum Heart Rate Achieved:")
    print("Highest heart rate during exercise; lower may indicate problems.")
    thalach = get_input("Max Heart Rate (e.g. 150): ", float)

    print("\n🏃 Exercise Induced Angina:")
    print("Chest pain only during physical activity.")
    print("0 = No")
    print("1 = Yes")
    exang = get_input("Angina during exercise? (0 or 1): ", int, [0,1])

    print("\n📊 ST Depression (Oldpeak):")
    print("Shows heart strain during exercise test, typical 0.0-4.0.")
    oldpeak = get_input("ST Depression value (e.g. 1.5): ", float)

    print("\n📉 Slope of Peak Exercise ST Segment:")
    print("0 = Upsloping (best)")
    print("1 = Flat")
    print("2 = Downsloping (concerning)")
    slope = get_input("Slope type (0-2): ", int, [0,1,2])

    print("\n🩻 Number of Major Vessels Blocked:")
    print("0 = None, 3 = 3 or more vessels blocked.")
    ca = get_input("Blocked vessels (0-3): ", int, [0,1,2,3])

    print("\n🩸 Thalassemia Status:")
    print("1 = Normal")
    print("2 = Fixed Defect (permanent)")
    print("3 = Reversible Defect (temporary)")
    thal = get_input("Thalassemia status (1-3): ", int, [1,2,3])

    return {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

def display_result(result):
    print("\n" + "="*50)
    print("🎯 YOUR HEART DISEASE RISK ASSESSMENT")
    print("="*50)
    print(f"Prediction: {'Heart Disease Risk' if result['prediction'] == 1 else 'Low Heart Disease Risk'}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Probability No Disease: {result['probability_no_disease']}%")
    print(f"Probability of Disease: {result['probability_disease']}%")
    print("="*50)

def main():
    print("🫀 Welcome to the Heart Disease Risk Predictor!")
    predictor = HeartDiseasePredictor()
    while True:
        user_data = get_user_data()
        print("\nAnalyzing your data...")
        result = predictor.predict(user_data)
        if result:
            display_result(result)
        else:
            print("Error processing your data, please try again.")
        again = input("Check another? (y/n): ").lower()
        if again not in ['y', 'yes']:
            print("Thank you! Stay healthy!")
            break

if __name__ == "__main__":
    main()
