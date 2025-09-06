"""
Interactive Heart Disease Prediction
Enter your own health data to get risk assessment
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))
from predict import HeartDiseasePredictor

def get_user_input():
    """Get health data from user input with detailed explanations"""
    print("🏥 Heart Disease Risk Assessment")
    print("=" * 40)
    print("Please enter your health information:")
    print("-" * 40)
    
    try:
        # Personal Information
        print("\n📋 Personal Information:")
        print("Age: Your current age in years")
        age = float(input("Age: "))
        
        print("\n👤 Sex:")
        print("ℹ️  Men typically have higher heart disease risk at younger ages")
        print("   Women's risk increases after menopause")
        print("0 = Female")
        print("1 = Male")
        sex = int(input("Enter choice (0 or 1): "))
        
        # Symptoms
        print("\n💔 Chest Pain Type:")
        print("ℹ️  ANGINA = Chest pain caused by reduced blood flow to the heart")
        print("   Different types indicate different levels of heart problems:")
        print()
        print("0 = Typical Angina:")
        print("    • Crushing chest pain during exercise/stress")
        print("    • Goes away with rest or medication")
        print("    • Classic sign of blocked arteries")
        print()
        print("1 = Atypical Angina:")
        print("    • Unusual chest pain (sharp, stabbing, or burning)")
        print("    • May not follow typical patterns")
        print("    • Could still indicate heart problems")
        print()
        print("2 = Non-Anginal Pain:")
        print("    • Chest pain NOT related to heart problems")
        print("    • Could be muscle strain, acid reflux, etc.")
        print("    • Usually not a heart concern")
        print()
        print("3 = Asymptomatic:")
        print("    • No chest pain at all")
        print("    • 'Silent' heart disease (no obvious symptoms)")
        print("    • Can still have heart problems without pain")
        cp = int(input("\nChest Pain Type (0-3): "))
        
        # Vital Signs
        print("\n🩺 Vital Signs:")
        print("\n🌡️ Resting Blood Pressure:")
        print("ℹ️  Pressure in your arteries when heart is at rest")
        print("   Normal: <120 mmHg | High: >140 mmHg")
        print("   High blood pressure damages arteries over time")
        trestbps = float(input("Resting Blood Pressure (mmHg, e.g. 120): "))
        
        print("\n🧪 Cholesterol Level:")
        print("ℹ️  CHOLESTEROL = Fatty substance that can clog arteries")
        print("   Normal: <200 mg/dl | Borderline: 200-239 | High: >240")
        print("   High cholesterol builds plaque in arteries")
        chol = float(input("Cholesterol Level (mg/dl, e.g. 200): "))
        
        print("\n💓 Maximum Heart Rate:")
        print("ℹ️  Highest heart rate achieved during exercise/stress test")
        print("   Normal max ≈ 220 - your age")
        print("   Lower max rate may indicate heart problems")
        thalach = float(input("Maximum Heart Rate Achieved (e.g. 150): "))
        
        # Medical Tests
        print("\n🔬 Medical Test Results:")
        
        print("\n🍬 Fasting Blood Sugar:")
        print("ℹ️  DIABETES connection to heart disease:")
        print("   High blood sugar damages blood vessels")
        print("   Diabetics have 2-4x higher heart disease risk")
        print("   Normal: <100 mg/dl | Diabetes: >126 mg/dl")
        print("Is your fasting blood sugar > 120 mg/dl?")
        print("0 = No (blood sugar ≤ 120)")
        print("1 = Yes (blood sugar > 120)")
        fbs = int(input("Enter choice (0 or 1): "))
        
        print("\n📈 Resting ECG (Electrocardiogram):")
        print("ℹ️  ECG = Test that measures heart's electrical activity")
        print("   Shows heart rhythm and detects damage")
        print()
        print("0 = Normal:")
        print("    • Regular heart rhythm")
        print("    • No signs of damage or strain")
        print()
        print("1 = ST-T Wave Abnormality:")
        print("    • Minor electrical changes")
        print("    • May indicate mild heart strain")
        print("    • Could be early warning sign")
        print()
        print("2 = Left Ventricular Hypertrophy:")
        print("    • Heart's main pumping chamber is enlarged")
        print("    • Often caused by high blood pressure")
        print("    • Indicates significant heart stress")
        restecg = int(input("\nResting ECG Results (0-2): "))
        
        print("\n🏃 Exercise Induced Angina:")
        print("ℹ️  Chest pain that occurs ONLY during physical activity")
        print("   Strong indicator of blocked coronary arteries")
        print("   Heart needs more oxygen during exercise")
        print("   If arteries are blocked, you get chest pain")
        print("Do you get chest pain during exercise?")
        print("0 = No chest pain during exercise")
        print("1 = Yes, I get chest pain during exercise")
        exang = int(input("Enter choice (0 or 1): "))
        
        print("\n📊 ST Depression (Oldpeak):")
        print("ℹ️  Technical ECG measurement during stress test")
        print("   Shows how much heart struggles during exercise")
        print("   Higher values = more heart strain")
        print("   0 = Normal | 1-2 = Mild concern | >3 = Significant concern")
        print("   (If unsure, typical values: 0.0 - 4.0)")
        oldpeak = float(input("ST Depression value (e.g., 1.5): "))
        
        print("\n📉 Slope of Peak Exercise ST Segment:")
        print("ℹ️  How your heart's electrical activity changes with exercise")
        print("   Shows how well heart adapts to physical stress")
        print()
        print("0 = Upsloping (Best):")
        print("    • Heart adapts well to exercise")
        print("    • Good sign for heart health")
        print()
        print("1 = Flat:")
        print("    • Heart shows some stress during exercise")
        print("    • Moderate concern")
        print()
        print("2 = Downsloping (Concerning):")
        print("    • Heart struggles significantly with exercise")
        print("    • May indicate blocked arteries")
        slope = int(input("\nSlope type (0-2): "))
        
        print("\n🩻 Major Vessels (Cardiac Catheterization):")
        print("ℹ️  FLUOROSCOPY = Special X-ray that shows blood flow")
        print("   Counts how many major heart arteries have blockages")
        print("   More blocked vessels = higher risk")
        print()
        print("0 = 0 vessels blocked (Best)")
        print("1 = 1 vessel has blockage")
        print("2 = 2 vessels have blockages") 
        print("3 = 3+ vessels blocked (Highest risk)")
        print()
        print("(If you haven't had this test, most people are 0)")
        ca = int(input("Number of blocked major vessels (0-3): "))
        
        print("\n🩸 Thalassemia:")
        print("ℹ️  THALASSEMIA = Inherited blood disorder")
        print("   Affects how well blood carries oxygen")
        print("   Can stress the heart over time")
        print()
        print("1 = Normal (No blood disorder)")
        print("2 = Fixed Defect:")
        print("    • Permanent blood flow problem")
        print("    • Usually from previous heart damage")
        print()
        print("3 = Reversible Defect:")
        print("    • Temporary blood flow problem")
        print("    • May improve with treatment")
        print()
        print("(If unsure about blood disorders, choose 1 for Normal)")
        thal = int(input("Thalassemia status (1-3): "))
        
        # Create data dictionary
        user_data = {
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
        
        return user_data
        
    except ValueError:
        print("❌ Invalid input! Please enter numbers only.")
        return None
    except KeyboardInterrupt:
        print("\n\n👋 Assessment cancelled. Take care!")
        return None

def display_results(result):
    """Display prediction results in a nice format"""
    print("\n" + "=" * 50)
    print("🎯 YOUR HEART DISEASE RISK ASSESSMENT")
    print("=" * 50)
    
    # Risk percentage with visual indicator
    risk_pct = result['probability_disease']
    print(f"\n📊 Risk Probability: {risk_pct}%")
    
    # Visual risk bar
    bar_length = 20
    filled_length = int(bar_length * risk_pct / 100)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)
    print(f"    Risk Level: [{bar}] {risk_pct}%")
    
    # Risk categorization with colors (text)
    risk_level = result['risk_level']
    if risk_level == "High Risk":
        print(f"\n🔴 Risk Level: {risk_level}")
        print("⚠️  RECOMMENDATION: Please consult a cardiologist immediately")
        print("    Consider lifestyle changes and medical intervention")
    elif risk_level == "Moderate Risk":
        print(f"\n🟡 Risk Level: {risk_level}")
        print("⚠️  RECOMMENDATION: Schedule regular checkups with your doctor")
        print("    Consider lifestyle modifications (diet, exercise, stress management)")
    else:
        print(f"\n🟢 Risk Level: {risk_level}")
        print("✅ RECOMMENDATION: Maintain your current healthy lifestyle")
        print("    Continue regular preventive care and healthy habits")
    
    # Final diagnosis
    diagnosis = "Heart Disease Risk Detected" if result['prediction'] == 1 else "Low Heart Disease Risk"
    print(f"\n🏥 Diagnosis: {diagnosis}")
    
    # Additional info
    print(f"\n📈 Confidence Scores:")
    print(f"    No Disease: {result['probability_no_disease']}%")
    print(f"    Heart Disease: {result['probability_disease']}%")
    
    print("\n" + "=" * 50)
    print("⚠️  IMPORTANT DISCLAIMER:")
    print("This prediction is for educational purposes only.")
    print("Always consult qualified healthcare professionals")
    print("for actual medical diagnosis and treatment.")
    print("=" * 50)

def main():
    """Main function"""
    print("🫀 Welcome to the Interactive Heart Disease Risk Predictor!")
    print("=" * 60)
    print("This tool will assess your heart disease risk based on")
    print("various health parameters and medical test results.")
    print()
    print("📚 EDUCATIONAL GUIDE:")
    print("• Each question includes detailed medical explanations")
    print("• Don't worry if you don't understand all terms")
    print("• If unsure about a value, choose the 'normal' option")
    print("• This is for learning purposes only!")
    print()
    print("⚠️  PREPARATION:")
    print("• Have recent medical test results ready if available")
    print("• If you haven't had certain tests, we'll guide you")
    print("• The assessment takes about 5-10 minutes")
    print("=" * 60)
    
    while True:
        print("\n" + "=" * 50)
        
        # Initialize predictor
        print("🔄 Loading prediction model...")
        predictor = HeartDiseasePredictor()
        
        # Get user input
        user_data = get_user_input()
        
        if user_data is None:
            break
        
        # Make prediction
        print("\n🔄 Analyzing your health data...")
        print("   Please wait...")
        
        result = predictor.predict(user_data)
        
        if result:
            display_results(result)
        else:
            print("❌ Error: Could not process your data.")
            print("Please check your inputs and try again.")
        
        # Ask if user wants to try again
        print("\n" + "-" * 50)
        try:
            again = input("Would you like to check another person? (y/n): ").strip().lower()
            if again not in ['y', 'yes', '1']:
                break
        except KeyboardInterrupt:
            break
    
    print("\n👋 Thank you for using Heart Disease Risk Predictor!")
    print("Stay healthy and take care! ❤️")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Take care of your heart! ❤️")