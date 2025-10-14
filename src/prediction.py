import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import os
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib.pyplot as plt

def get_student_interests():
    """Get student interests for personalized learning path with single choice"""
    st.subheader("🎯 Student Interest Assessment")
    
    st.info("""
    **To provide you with a personalized learning path recommendation, 
    please select your preferred career interest:**
    """)
    
    interest = st.selectbox(
        "Select your area of interest:",
        ["Choose an option", "Web Development", "Cybersecurity", "Data Analytics", "Full Stack Development"],
        help="Choose one area that interests you the most"
    )
    
    # Add a button to generate learning path
    if interest != "Choose an option":
        if st.button("🚀 Generate Learning Path", type="primary", use_container_width=True):
            return interest
    return None

def display_learning_path_tree(interest):
    """Display personalized learning path tree based on student interest"""
    
    st.subheader(f"🌳 Personalized Learning Path: {interest}")
    
    if interest == "Web Development":
        tree_structure = """
        ```
        🌐 Web Development Learning Path
        │
        ├── HTML?
        │   ├── Yes → CSS?
        │   │   ├── Yes → Bootstrap?
        │   │   │   ├── Yes → JavaScript?
        │   │   │   │   ├── Yes → GitHub → 🎯 Ready for Projects
        │   │   │   │   └── No → Learn JavaScript First
        │   │   │   └── No → Learn Bootstrap First
        │   │   └── No → Learn CSS First
        │   └── No → Start with HTML Basics
        ```
        """
        st.markdown(tree_structure)
        
        st.info("""
        **Web Development Roadmap:**
        1. **HTML**: Structure and semantics
        2. **CSS**: Styling and layout
        3. **Bootstrap**: Responsive design framework
        4. **JavaScript**: Interactive functionality
        5. **GitHub**: Version control and collaboration
        6. **Projects**: Build real-world applications
        """)
        
    elif interest == "Cybersecurity":
        tree_structure = """
        ```
        🔒 Cybersecurity Learning Path
        │
        ├── Computer Networks?
        │   ├── Yes → Operating Systems (Linux)?
        │   │   ├── Yes → Security Tools?
        │   │   │   ├── Yes → Ethical Hacking → 🎯 Security Expert
        │   │   │   └── No → Learn Tools (Wireshark, Kali, Metasploit)
        │   │   └── No → Learn Linux Fundamentals
        │   └── No → Start with Computer Networks
        ```
        """
        st.markdown(tree_structure)
        
        st.info("""
        **Cybersecurity Roadmap:**
        1. **Computer Networks**: TCP/IP, protocols, network architecture
        2. **Linux OS**: Command line, system administration
        3. **Security Tools**: Wireshark, Kali Linux, Metasploit
        4. **Ethical Hacking**: Penetration testing, vulnerability assessment
        5. **Security Frameworks**: NIST, ISO 27001
        6. **Certifications**: CEH, Security+, CISSP
        """)
        
    elif interest == "Data Analytics":
        tree_structure = """
        ```
        📊 Data Analytics Learning Path
        │
        ├── Statistics?
        │   ├── Yes → Excel/Google Sheets?
        │   │   ├── Yes → SQL?
        │   │   │   ├── Yes → Python?
        │   │   │   │   ├── Yes → Power BI/Tableau → 🎯 Data Analyst
        │   │   │   │   └── No → Learn Python for Data Analysis
        │   │   │   └── No → Learn SQL for Data Querying
        │   │   └── No → Master Spreadsheet Analysis
        │   └── No → Learn Statistical Fundamentals
        ```
        """
        st.markdown(tree_structure)
        
        st.info("""
        **Data Analytics Roadmap:**
        1. **Statistics**: Probability, distributions, hypothesis testing
        2. **Spreadsheets**: Excel, Google Sheets for data manipulation
        3. **SQL**: Database querying and management
        4. **Python**: Pandas, NumPy, data visualization
        5. **BI Tools**: Power BI, Tableau for dashboards
        6. **Machine Learning**: Basic predictive modeling
        """)
        
    elif interest == "Full Stack Development":
        tree_structure = """
        ```
        🚀 Full Stack Development Learning Path
        │
        ├── Web Basics (HTML/CSS/JS)?
        │   ├── Yes → ReactJS?
        │   │   ├── Yes → Tailwind CSS?
        │   │   │   ├── Yes → API Handling?
        │   │   │   │   ├── Yes → Database → 🎯 Full Stack Developer
        │   │   │   │   └── No → Learn RESTful APIs
        │   │   │   └── No → Learn Tailwind CSS
        │   │   └── No → Learn ReactJS Framework
        │   └── No → Master Web Fundamentals First
        ```
        """
        st.markdown(tree_structure)
        
        st.info("""
        **Full Stack Development Roadmap:**
        1. **Frontend**: HTML, CSS, JavaScript, React
        2. **Styling**: Tailwind CSS, responsive design
        3. **Backend**: Node.js, Express, API development
        4. **Database**: MongoDB, PostgreSQL, data modeling
        5. **DevOps**: Deployment, cloud platforms
        6. **Projects**: End-to-end application development
        """)

def provide_personalized_recommendations(interest, dropout_prediction):
    """Provide personalized learning recommendations based on interest and risk level"""
    
    st.subheader("💡 Personalized Learning Recommendations")
    
    if dropout_prediction == 1:
        st.warning("""
        **🚨 Based on your high dropout risk assessment, we recommend:**
        
        **Immediate Focus Areas:**
        1. **Foundational Skills First**: Start with basic concepts to build confidence
        2. **Structured Learning Path**: Follow the recommended tree structure step-by-step
        3. **Regular Progress Checks**: Set small, achievable milestones
        4. **Support System**: Engage with mentors and peer groups
        5. **Flexible Pace**: Don't rush - focus on understanding fundamentals
        """)
    else:
        st.success("""
        **✅ Based on your low dropout risk assessment, you can:**
        
        **Accelerated Learning Approach:**
        1. **Comprehensive Coverage**: Follow the complete learning path
        2. **Advanced Topics**: Explore specialized areas within your interests
        3. **Project-Based Learning**: Build real projects to reinforce skills
        4. **Community Engagement**: Participate in open source and coding communities
        5. **Career Preparation**: Focus on portfolio development and interview skills
        """)
    
    # Display learning path for the selected interest
    display_learning_path_tree(interest)
    
    # Add specific recommendations for the interest
    if interest == "Web Development":
        st.info("""
        **Web Development Quick Start:**
        - **Week 1-2**: HTML & CSS fundamentals
        - **Week 3-4**: JavaScript basics and DOM manipulation
        - **Week 5-6**: Responsive design with Bootstrap
        - **Week 7-8**: Version control with Git/GitHub
        - **Week 9-12**: Build a portfolio project
        """)
        
    elif interest == "Cybersecurity":
        st.info("""
        **Cybersecurity Quick Start:**
        - **Week 1-2**: Network fundamentals and protocols
        - **Week 3-4**: Linux command line and system basics
        - **Week 5-6**: Introduction to security tools
        - **Week 7-8**: Vulnerability assessment basics
        - **Week 9-12**: Hands-on security lab exercises
        """)
        
    elif interest == "Data Analytics":
        st.info("""
        **Data Analytics Quick Start:**
        - **Week 1-2**: Statistical fundamentals and Excel
        - **Week 3-4**: SQL for data querying
        - **Week 5-6**: Python basics for data analysis
        - **Week 7-8**: Data visualization with Python/Tableau
        - **Week 9-12**: Complete a data analysis project
        """)
        
    elif interest == "Full Stack Development":
        st.info("""
        **Full Stack Development Quick Start:**
        - **Week 1-3**: Frontend fundamentals (HTML, CSS, JS)
        - **Week 4-6**: React.js and modern frontend development
        - **Week 7-8**: Backend development with Node.js
        - **Week 9-10**: Database integration and APIs
        - **Week 11-12**: Full stack project deployment
        """)

def apply_behavioral_decision_tree(input_data):
    """
    Apply comprehensive decision tree logic for behavioral analysis
    """
    motivation_level = input_data.get('Motivation_Level', 1)
    stress_level = input_data.get('Stress_Level', 1)
    sleep_hours = input_data.get('Sleep_Hours', 7)
    adaptability = input_data.get('Adaptability', 1)
    social_interaction = input_data.get('Social_Interaction', 1)
    
    # Enhanced Decision Tree Logic
    if motivation_level in [1, 2]:  # Medium or High motivation
        return 0  # No Dropout
    else:  # Low motivation
        if stress_level in [1, 2]:  # Medium or High stress
            if adaptability == 1:  # Yes - Good adaptability
                return 0  # No Dropout
            else:  # No - Poor adaptability
                if social_interaction in [1, 2]:  # Medium or High social interaction
                    return 0  # No Dropout
                else:  # Low social interaction
                    return 1  # Dropout
        else:  # Low stress
            if sleep_hours < 4:  # Less than 4 hours
                return 1  # Dropout
            else:  # 4 or more hours
                return 0  # No Dropout

def display_behavioral_decision_tree():
    """Display the enhanced decision tree visualization"""
    st.subheader("🧠 Enhanced Behavioral Analysis Decision Tree")
    
    tree_structure = """
    ```
    🎯 MOTIVATION LEVEL (Root Node)
    │
    ├── High/Medium → ✅ NO DROPOUT
    │
    └── Low → 🎯 STRESS LEVEL (Child Node)
           │
           ├── High/Medium → 🎯 ADAPTABILITY (Child Node)
           │                │
           │                ├── Yes → ✅ NO DROPOUT
           │                │
           │                └── No → 🎯 SOCIAL INTERACTION (Child Node)
           │                           │
           │                           ├── High/Medium → ✅ NO DROPOUT
           │                           │
           │                           └── Low → 🚨 DROPOUT
           │
           └── Low → 🎯 SLEEP HOURS (Leaf Node)
                  │
                  ├── <4 hours → 🚨 DROPOUT
                  │
                  └── ≥4 hours → ✅ NO DROPOUT
    ```
    """
    st.markdown(tree_structure)

def get_behavioral_tree_input():
    """Get user input for all decision tree nodes including new ones"""
    st.subheader("🎯 Behavioral Analysis - Enhanced Decision Tree Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Primary Nodes**")
        st.write("##### 🎯 Root Node")
        motivation_level = st.selectbox(
            "Motivation Level", 
            ["Low", "Medium", "High"],
            key="behavioral_motivation"
        )
        
        st.write("##### 🎯 First Level Child Nodes")
        stress_level = st.selectbox(
            "Stress Level", 
            ["Low", "Medium", "High"],
            key="behavioral_stress"
        )
        
        adaptability = st.selectbox(
            "Adaptability to Course", 
            ["No", "Yes"],
            key="behavioral_adaptability"
        )
    
    with col2:
        st.write("**Secondary Nodes**")
        st.write("##### 🎯 Second Level Child Nodes")
        social_interaction = st.selectbox(
            "Social Interaction Level", 
            ["Low", "Medium", "High"],
            key="behavioral_social"
        )
        
        st.write("##### 🎯 Leaf Node")
        sleep_hours = st.slider(
            "Sleep Hours (per day)", 
            0, 12, 7,
            key="behavioral_sleep"
        )
    
    # Convert to numerical values
    motivation_numeric = {"Low": 0, "Medium": 1, "High": 2}[motivation_level]
    stress_numeric = {"Low": 0, "Medium": 1, "High": 2}[stress_level]
    adaptability_numeric = {"No": 0, "Yes": 1}[adaptability]
    social_numeric = {"Low": 0, "Medium": 1, "High": 2}[social_interaction]
    
    return {
        'Motivation_Level': motivation_numeric,
        'Stress_Level': stress_numeric,
        'Sleep_Hours': sleep_hours,
        'Adaptability': adaptability_numeric,
        'Social_Interaction': social_numeric
    }

def analyze_behavioral_pathway(input_data):
    """Analyze and display the specific pathway taken in the decision tree"""
    motivation_level = input_data.get('Motivation_Level', 1)
    stress_level = input_data.get('Stress_Level', 1)
    sleep_hours = input_data.get('Sleep_Hours', 7)
    adaptability = input_data.get('Adaptability', 1)
    social_interaction = input_data.get('Social_Interaction', 1)
    
    pathway = ["**Decision Tree Pathway:**"]
    
    if motivation_level in [1, 2]:
        pathway.append("✅ High/Medium Motivation → NO DROPOUT")
        return pathway
    
    pathway.append("🔻 Low Motivation → Proceed to Stress Level")
    
    if stress_level in [1, 2]:
        pathway.append("🔻 High/Medium Stress → Proceed to Adaptability")
        
        if adaptability == 1:
            pathway.append("✅ Good Adaptability → NO DROPOUT")
            return pathway
        else:
            pathway.append("🔻 Poor Adaptability → Proceed to Social Interaction")
            
            if social_interaction in [1, 2]:
                pathway.append("✅ High/Medium Social Interaction → NO DROPOUT")
                return pathway
            else:
                pathway.append("🚨 Low Social Interaction → DROPOUT")
                return pathway
    else:
        pathway.append("🔻 Low Stress → Proceed to Sleep Hours")
        
        if sleep_hours < 4:
            pathway.append("🚨 Insufficient Sleep (<4 hours) → DROPOUT")
            return pathway
        else:
            pathway.append("✅ Adequate Sleep (≥4 hours) → NO DROPOUT")
            return pathway

def combine_predictions(ml_prediction, ml_proba, behavioral_prediction, input_data):
    """
    Combine ML model prediction with enhanced behavioral analysis
    """
    ml_confidence = max(ml_proba)
    
    # Get behavioral factors for weighting
    motivation_level = input_data.get('Motivation_Level', 1)
    stress_level = input_data.get('Stress_Level', 1)
    sleep_hours = input_data.get('Sleep_Hours', 7)
    adaptability = input_data.get('Adaptability', 1)
    social_interaction = input_data.get('Social_Interaction', 1)
    
    # Calculate behavioral risk score (0-5)
    behavioral_risk_score = 0
    if motivation_level == 0:  # Low motivation
        behavioral_risk_score += 1
    if stress_level == 0:  # Low stress
        behavioral_risk_score += 1
    if sleep_hours < 4:  # Low sleep
        behavioral_risk_score += 1
    if adaptability == 0:  # Poor adaptability
        behavioral_risk_score += 1
    if social_interaction == 0:  # Low social interaction
        behavioral_risk_score += 1
    
    # Strong behavioral signal overrides ML
    if behavioral_risk_score >= 3:  # High behavioral risk
        final_prediction = 1  # Dropout
        final_confidence = max(ml_confidence, 0.85)
    elif behavioral_prediction == 1 and ml_confidence < 0.7:
        final_prediction = 1  # Dropout
        final_confidence = 0.75
    elif behavioral_prediction == 0 and ml_confidence < 0.6:
        final_prediction = 0  # No Dropout
        final_confidence = 0.70
    else:
        # Trust ML prediction
        final_prediction = ml_prediction
        final_confidence = ml_confidence
    
    return final_prediction, final_confidence

def display_prediction_results(dropout_prediction, prediction_proba, input_data, model_name, behavioral_prediction, ml_prediction):
    """Display comprehensive prediction results with insights"""
    
    st.subheader("📊 Prediction Results")
    
    # Result cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if dropout_prediction == 1:
            st.error(f"🚨 **HIGH RISK**")
            risk_level = "High"
        else:
            st.success(f"✅ **LOW RISK**")
            risk_level = "Low"
    
    with col2:
        confidence = max(prediction_proba) * 100
        st.metric("ML Confidence", f"{confidence:.1f}%")
    
    with col3:
        st.metric("Model Used", model_name)
    
    with col4:
        behavioral_result = "High Risk" if behavioral_prediction == 1 else "Low Risk"
        st.metric("Behavioral Analysis", behavioral_result)
    
    # Show prediction breakdown
    st.subheader("🔍 Prediction Breakdown")
    
    breakdown_col1, breakdown_col2 = st.columns(2)
    
    with breakdown_col1:
        st.write("**Machine Learning Model:**")
        ml_result = "Dropout" if ml_prediction == 1 else "No Dropout"
        st.info(f"Prediction: {ml_result}")
        st.info(f"Confidence: {max(prediction_proba)*100:.1f}%")
    
    with breakdown_col2:
        st.write("**Behavioral Analysis:**")
        behavioral_result = "Dropout" if behavioral_prediction == 1 else "No Dropout"
        st.warning(f"Prediction: {behavioral_result}")
    
    # Probability breakdown
    st.subheader("📈 Risk Probability Breakdown")
    
    prob_df = pd.DataFrame({
        'Outcome': ['No Dropout', 'Dropout'],
        'Probability': [prediction_proba[0] * 100, prediction_proba[1] * 100]
    })
    
    fig = px.bar(prob_df, x='Outcome', y='Probability',
                color='Outcome',
                color_discrete_map={'No Dropout': '#2E8B57', 'Dropout': '#FF4500'},
                text='Probability',
                title='Dropout Risk Probability Distribution')
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)
    
    # Recommendations
    st.subheader("💡 Recommendations")
    
    if dropout_prediction == 1:
        st.error("""
        **🚨 URGENT ACTION REQUIRED:**
        
        **Immediate Steps:**
        1. **Behavioral Intervention:** Address motivation, adaptability, and social engagement issues
        2. **Academic Support:** Provide additional tutoring and personalized learning resources
        3. **Social Integration:** Facilitate peer interactions and group activities
        4. **Adaptability Training:** Offer course orientation and study skills workshops
        5. **Wellness Support:** Improve sleep habits and stress management
        6. **Mentorship:** Assign a mentor for regular check-ins and guidance
        """)
        
        # Show personalized learning paths for high-risk students
        st.markdown("---")
        st.subheader("🎯 Personalized Career Guidance")
        
        # Get student interest with button
        selected_interest = get_student_interests()
        
        if selected_interest:
            # Display personalized recommendations
            provide_personalized_recommendations(selected_interest, dropout_prediction)
        
    else:
        st.success("""
        **✅ STUDENT IS ON TRACK:**
        
        **Maintenance Actions:**
        1. **Continue Current Support:** Maintain existing academic and behavioral support
        2. **Regular Check-ins:** Schedule monthly progress reviews
        3. **Encourage Engagement:** Promote participation in advanced activities and social events
        4. **Skill Development:** Provide opportunities for continuous skill enhancement
        5. **Wellness Monitoring:** Continue monitoring behavioral factors regularly
        """)

def analyze_enhanced_behavioral_factors(input_data):
    """Analyze enhanced behavioral factors and provide insights"""
    insights = {}
    
    # Motivation analysis
    motivation_level = input_data.get('Motivation_Level', 1)
    if motivation_level == 0:  # Low
        insights['Motivation'] = {
            'risk': True,
            'message': 'Low motivation detected. This is the primary risk factor for dropout.'
        }
    elif motivation_level == 1:  # Medium
        insights['Motivation'] = {
            'risk': False,
            'message': 'Moderate motivation level. Maintain engagement to prevent decline.'
        }
    else:  # High
        insights['Motivation'] = {
            'risk': False,
            'message': 'High motivation level. Strong protective factor against dropout.'
        }
    
    # Stress analysis
    stress_level = input_data.get('Stress_Level', 1)
    if stress_level == 0:  # Low
        insights['Stress'] = {
            'risk': True,
            'message': 'Low stress may indicate lack of engagement or care about course outcomes.'
        }
    elif stress_level == 1:  # Medium
        insights['Stress'] = {
            'risk': False,
            'message': 'Moderate stress level. Healthy engagement with course material.'
        }
    else:  # High
        insights['Stress'] = {
            'risk': False,
            'message': 'High stress level. Monitor for burnout but indicates active engagement.'
        }
    
    # Sleep analysis
    sleep_hours = input_data.get('Sleep_Hours', 7)
    if sleep_hours < 4:
        insights['Sleep'] = {
            'risk': True,
            'message': f'Only {sleep_hours} hours sleep. Severe sleep deprivation affects cognitive function.'
        }
    elif sleep_hours < 6:
        insights['Sleep'] = {
            'risk': True,
            'message': f'{sleep_hours} hours sleep. Insufficient rest may impact learning capacity.'
        }
    else:
        insights['Sleep'] = {
            'risk': False,
            'message': f'{sleep_hours} hours sleep. Adequate rest for optimal learning performance.'
        }
    
    # Adaptability analysis
    adaptability = input_data.get('Adaptability', 1)
    if adaptability == 0:  # No
        insights['Adaptability'] = {
            'risk': True,
            'message': 'Poor adaptability to course structure. May struggle with learning pace and methods.'
        }
    else:  # Yes
        insights['Adaptability'] = {
            'risk': False,
            'message': 'Good adaptability to course. Can adjust to learning demands effectively.'
        }
    
    # Social interaction analysis
    social_interaction = input_data.get('Social_Interaction', 1)
    if social_interaction == 0:  # Low
        insights['Social Interaction'] = {
            'risk': True,
            'message': 'Low social interaction. Limited peer support and collaborative learning opportunities.'
        }
    elif social_interaction == 1:  # Medium
        insights['Social Interaction'] = {
            'risk': False,
            'message': 'Moderate social interaction. Balanced engagement with peers and independent study.'
        }
    else:  # High
        insights['Social Interaction'] = {
            'risk': False,
            'message': 'High social interaction. Strong peer network supports learning and motivation.'
        }
    
    return insights

def analyze_risk_factors(input_data):
    """Analyze input data to identify potential risk factors"""
    risk_factors = {}
    
    # Academic performance risks
    if input_data.get('Quiz_Scores', 100) < 50:
        risk_factors['Low Quiz Scores'] = f"Quiz score of {input_data['Quiz_Scores']} is below passing threshold"
    
    if input_data.get('Final_Exam_Score', 100) < 50:
        risk_factors['Low Exam Score'] = f"Final exam score of {input_data['Final_Exam_Score']} indicates academic struggle"
    
    if input_data.get('Assignment_Completion_Rate', 100) < 60:
        risk_factors['Low Assignment Completion'] = f"Only {input_data['Assignment_Completion_Rate']}% assignments completed"
    
    # Engagement risks
    if input_data.get('Time_Spent_on_Videos', 600) < 100:
        risk_factors['Low Video Engagement'] = "Minimal time spent on course videos"
    
    if input_data.get('Forum_Participation', 50) < 5:
        risk_factors['Low Forum Participation'] = "Limited interaction with peers and instructors"
    
    if input_data.get('Engagement_Level', 2) == 0:  # Low engagement
        risk_factors['Low Overall Engagement'] = "Student shows minimal course engagement"
    
    if input_data.get('Quiz_Attempts', 5) < 2:
        risk_factors['Limited Quiz Attempts'] = "Few attempts made on quizzes"
    
    # Demographic risks (based on historical patterns)
    if input_data.get('Age', 25) < 18:
        risk_factors['Young Age'] = "Younger students may need additional support"
    
    if input_data.get('Feedback_Score', 5) < 2:
        risk_factors['Low Satisfaction'] = "Low course feedback indicates potential issues"
    
    return risk_factors

def predict_new_student(df, preprocessor):
    """Handle new student prediction with enhanced behavioral analysis"""
    st.header("🎯 New Student Dropout Risk Prediction")
    
    # Check if models exist
    model_files = ['random_forest.pkl', 'logistic_regression.pkl', 'support_vector_machine.pkl', 
                   'gradient_boosting.pkl', 'decision_tree.pkl']
    model_exists = any(os.path.exists(f'models/{model}') for model in model_files)
    
    if not model_exists:
        st.error("""
        ❌ **Models not trained yet!**
        
        Before making predictions, please:
        1. Go to the **🔮 Classification Models** section
        2. Click the **'Train All Models'** button
        3. Wait for model training to complete
        4. Return here to make predictions
        """)
        
        if st.button("🚀 Go to Model Training", use_container_width=True):
            st.session_state.navigation = "Classification Models"
            st.rerun()
        
        return None

    # Display the enhanced decision tree structure first
    display_behavioral_decision_tree()
    
    # Create comprehensive input form
    with st.form("student_input_form"):
        st.subheader("📝 Student Profile Information")
        
        input_data = {}
        
        # Personal Information
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Personal Details**")
            input_data['Age'] = st.slider("Age", 15, 60, 25)
            
            gender = st.selectbox("Gender", ["Male", "Female"])
            input_data['Gender'] = 1 if gender == "Male" else 0
            
            education_level = st.selectbox("Education Level", 
                                         ["High School", "Undergraduate", "Postgraduate"])
            input_data['Education_Level'] = ["High School", "Undergraduate", "Postgraduate"].index(education_level)
        
        with col2:
            st.write("**Course Information**")
            course_name = st.selectbox("Course Name", 
                                     ["Machine Learning", "Python Basics", "Data Science", 
                                      "Web Development", "Cybersecurity"])
            input_data['Course_Name'] = ["Machine Learning", "Python Basics", "Data Science", 
                                       "Web Development", "Cybersecurity"].index(course_name)
            
            learning_style = st.selectbox("Learning Style", 
                                        ["Visual", "Reading/Writing", "Auditory", "Kinesthetic"])
            input_data['Learning_Style'] = ["Visual", "Reading/Writing", "Auditory", "Kinesthetic"].index(learning_style)
        
        # Academic Performance
        st.subheader("📊 Academic Performance")
        
        col3, col4 = st.columns(2)
        
        with col3:
            input_data['Time_Spent_on_Videos'] = st.slider("Time Spent on Videos (minutes)", 0, 600, 200)
            input_data['Quiz_Attempts'] = st.slider("Quiz Attempts", 0, 5, 2)
            input_data['Quiz_Scores'] = st.slider("Quiz Scores", 0, 100, 70)
        
        with col4:
            input_data['Forum_Participation'] = st.slider("Forum Participation", 0, 50, 10)
            input_data['Assignment_Completion_Rate'] = st.slider("Assignment Completion Rate (%)", 0, 100, 80)
            input_data['Final_Exam_Score'] = st.slider("Final Exam Score", 0, 100, 65)
        
        # Engagement and Feedback
        st.subheader("💡 Engagement & Feedback")
        
        col5, col6 = st.columns(2)
        
        with col5:
            engagement_level = st.selectbox("Engagement Level", ["Low", "Medium", "High"])
            input_data['Engagement_Level'] = ["Low", "Medium", "High"].index(engagement_level)
            
        with col6:
            input_data['Feedback_Score'] = st.slider("Course Feedback Score (1-5)", 1, 5, 3)
        
        # Enhanced Behavioral Analysis Section
        st.subheader("🧠 Enhanced Behavioral Analysis - Decision Tree Nodes")
        
        # Get behavioral inputs using the enhanced decision tree structure
        behavioral_inputs = get_behavioral_tree_input()
        input_data.update(behavioral_inputs)
        
        submitted = st.form_submit_button("🔮 Predict Dropout Risk", use_container_width=True)
        
        if submitted:
            # Apply enhanced behavioral analysis decision tree
            behavioral_prediction = apply_behavioral_decision_tree(input_data)
            
            # Display decision tree pathway
            st.subheader("🛣️ Decision Tree Pathway Analysis")
            pathway = analyze_behavioral_pathway(input_data)
            for step in pathway:
                st.write(step)
            
            # Prepare input data for ML prediction
            input_df = pd.DataFrame([input_data])
            
            try:
                # Load the best model
                best_model = None
                model_name = ""
                
                # Try to load models in order of preference
                if os.path.exists('models/random_forest.pkl'):
                    best_model = joblib.load('models/random_forest.pkl')
                    model_name = "Random Forest"
                elif os.path.exists('models/gradient_boosting.pkl'):
                    best_model = joblib.load('models/gradient_boosting.pkl')
                    model_name = "Gradient Boosting"
                elif os.path.exists('models/logistic_regression.pkl'):
                    best_model = joblib.load('models/logistic_regression.pkl')
                    model_name = "Logistic Regression"
                else:
                    # Load any available model
                    for model_file in model_files:
                        if os.path.exists(f'models/{model_file}'):
                            best_model = joblib.load(f'models/{model_file}')
                            model_name = model_file.replace('.pkl', '').replace('_', ' ').title()
                            break
                
                if best_model is None:
                    st.error("No trained models found. Please train models first.")
                    return None
                
                # Load scaler if available
                scaler = None
                if os.path.exists('models/scaler.pkl'):
                    scaler = joblib.load('models/scaler.pkl')
                
                # Feature alignment
                try:
                    # Get the features the model was trained on
                    if hasattr(best_model, 'feature_names_in_'):
                        expected_features = best_model.feature_names_in_
                    else:
                        # Remove features that might not be in the training data
                        features_to_drop = ['Dropout_Likelihood', 'Dropout_encoded', 'Student_ID', 'Cluster']
                        expected_features = [col for col in input_df.columns if col not in features_to_drop]
                    
                    # Check if behavioral features are in expected features
                    behavioral_features = ['Learning_Style', 'Motivation_Level', 'Stress_Level', 'Sleep_Hours', 'Adaptability', 'Social_Interaction']
                    unexpected_features = []
                    for feature in behavioral_features:
                        if feature not in expected_features and feature in input_df.columns:
                            unexpected_features.append(feature)
                    
                    if unexpected_features:
                        st.warning(f"⚠️ Features {unexpected_features} not used in model training. They will be excluded from ML prediction.")
                        input_df = input_df.drop(unexpected_features, axis=1)
                    
                    # Ensure we only use features that the model expects
                    available_features = [col for col in expected_features if col in input_df.columns]
                    input_df_aligned = input_df[available_features]
                    
                    # Make ML prediction
                    if scaler is not None:
                        input_scaled = scaler.transform(input_df_aligned)
                        ml_dropout_prediction = best_model.predict(input_scaled)[0]
                        ml_prediction_proba = best_model.predict_proba(input_scaled)[0]
                    else:
                        ml_dropout_prediction = best_model.predict(input_df_aligned)[0]
                        ml_prediction_proba = best_model.predict_proba(input_df_aligned)[0]
                    
                    # Combine ML prediction with behavioral analysis
                    final_prediction, final_confidence = combine_predictions(
                        ml_dropout_prediction, ml_prediction_proba, behavioral_prediction, input_data
                    )
                    
                    # Display results
                    display_prediction_results(
                        final_prediction, 
                        ml_prediction_proba, 
                        input_data, 
                        model_name, 
                        behavioral_prediction,
                        ml_dropout_prediction
                    )
                    
                    # Return results for career recommendation
                    return {
                        'dropout_prediction': 'Dropout' if final_prediction == 1 else 'Not Dropout',
                        'confidence': final_confidence,
                        'input_data': input_data,
                        'probabilities': ml_prediction_proba,
                        'model_used': model_name,
                        'behavioral_prediction': behavioral_prediction,
                        'ml_prediction': ml_dropout_prediction
                    }
                    
                except Exception as e:
                    st.error(f"❌ Feature alignment error: {str(e)}")
                    st.info("""
                    **Troubleshooting steps:**
                    1. Ensure models are trained with the correct features
                    2. Check if the training data included all expected features
                    3. Try retraining the models with the current feature set
                    """)
                    return None
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.info("Please ensure all models are properly trained and try again.")
                return None
    
    return None

def get_expected_features(model):
    """Get the features expected by the model"""
    if hasattr(model, 'feature_names_in_'):
        return model.feature_names_in_
    else:
        # Return a safe default set of features (adjust based on your dataset)
        return ['Age', 'Gender', 'Education_Level', 'Course_Name', 'Time_Spent_on_Videos', 
                'Quiz_Attempts', 'Quiz_Scores', 'Forum_Participation', 
                'Assignment_Completion_Rate', 'Final_Exam_Score', 'Engagement_Level', 
                'Feedback_Score', 'Motivation_Level', 'Stress_Level', 'Sleep_Hours',
                'Adaptability', 'Social_Interaction']
