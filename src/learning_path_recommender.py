import streamlit as st
import pandas as pd
import numpy as np

def ai_learning_path_recommendation():
    """AI Learning Path Recommendation System"""
    st.header("🤖 AI Learning Path Recommendation Engine")
    st.info("Get personalized AI learning paths based on your skills, goals, and background")
    
    # Initialize session state
    if 'learning_path_submitted' not in st.session_state:
        st.session_state.learning_path_submitted = False
    if 'learning_path_results' not in st.session_state:
        st.session_state.learning_path_results = None
    
    # Display results first if they exist
    if st.session_state.learning_path_submitted and st.session_state.learning_path_results:
        display_learning_path_recommendation(st.session_state.learning_path_results)
        return
    
    # Learning path questionnaire - only show if no results
    with st.form("learning_path_form"):
        st.subheader("🎯 Learning Profile Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Skills & Background**")
            
            current_skills = st.multiselect(
                "Current Technical Skills",
                ["Python Basics", "Statistics", "Linear Algebra", "Calculus",
                 "SQL", "Data Analysis", "Programming Fundamentals", "Mathematics",
                 "None - Starting from scratch"],
                help="Select skills you already have"
            )
            
            programming_experience = st.selectbox(
                "Programming Experience Level",
                ["Beginner (0-6 months)", "Intermediate (6 months - 2 years)", 
                 "Advanced (2+ years)", "Expert (5+ years)"]
            )
            
            math_background = st.selectbox(
                "Mathematics Background",
                ["Basic High School Math", "College-level Calculus", 
                 "Statistics & Probability", "Linear Algebra", "Advanced Mathematics"]
            )
        
        with col2:
            st.write("**Learning Goals & Interests**")
            
            ai_interests = st.multiselect(
                "AI Areas of Interest",
                ["Machine Learning", "Deep Learning", "Natural Language Processing",
                 "Computer Vision", "Data Science", "AI Ethics", "Reinforcement Learning",
                 "Generative AI", "AI Research", "ML Engineering"],
                help="Select AI fields that interest you"
            )
            
            learning_goals = st.selectbox(
                "Primary Learning Goal",
                ["Career Transition", "Skill Enhancement", "Academic Research",
                 "Personal Projects", "Entrepreneurship", "Competitions"]
            )
            
            time_commitment = st.selectbox(
                "Weekly Time Commitment",
                ["5-10 hours", "10-20 hours", "20-30 hours", "30+ hours"]
            )
        
        # Additional preferences
        st.write("**Learning Preferences**")
        col3, col4 = st.columns(2)
        
        with col3:
            preferred_format = st.multiselect(
                "Preferred Learning Format",
                ["Video Courses", "Interactive Coding", "Text-based Tutorials",
                 "Project-based", "Research Papers", "Books"]
            )
            
        with col4:
            budget = st.selectbox(
                "Learning Budget",
                ["Free resources only", "Under $50", "$50-$200", "$200-$500", "No budget limit"]
            )
        
        submitted = st.form_submit_button("🚀 Generate AI Learning Path", use_container_width=True)
        
        if submitted:
            # Validate that required fields are filled
            if not ai_interests:
                st.error("❌ Please select at least one AI area of interest")
                return
                
            if not current_skills:
                st.warning("⚠️ No skills selected. Assuming you're starting from scratch.")
                current_skills = ["None - Starting from scratch"]
            
            # Generate learning path recommendation
            with st.spinner("🤖 Generating your personalized AI learning path..."):
                learning_path_results = generate_ai_learning_path(
                    current_skills, programming_experience, math_background,
                    ai_interests, learning_goals, time_commitment,
                    preferred_format, budget
                )
                
                st.session_state.learning_path_submitted = True
                st.session_state.learning_path_results = learning_path_results
                
                # Rerun to display results
                st.rerun()

def generate_ai_learning_path(skills, programming_exp, math_bg, interests, goals, time_commitment, format_pref, budget):
    """Generate personalized AI learning path"""
    
    # Determine starting level
    if "None - Starting from scratch" in skills or programming_exp == "Beginner (0-6 months)":
        starting_level = "Beginner"
    elif programming_exp in ["Intermediate (6 months - 2 years)", "Advanced (2+ years)"]:
        starting_level = "Intermediate"
    else:
        starting_level = "Advanced"
    
    # Generate learning path based on level and interests
    learning_path = create_structured_learning_path(starting_level, interests, goals)
    
    # Calculate estimated timeline
    timeline = estimate_timeline(time_commitment, starting_level, len(interests))
    
    # Recommend resources
    resources = recommend_resources(starting_level, interests, format_pref, budget)
    
    # Generate projects
    projects = generate_projects(starting_level, interests)
    
    return {
        "starting_level": starting_level,
        "learning_path": learning_path,
        "timeline": timeline,
        "resources": resources,
        "projects": projects,
        "assessment": {
            "current_skills": skills,
            "programming_level": programming_exp,
            "math_level": math_bg,
            "interests": interests,
            "goals": goals,
            "time_commitment": time_commitment,
            "preferred_format": format_pref,
            "budget": budget
        }
    }

def create_structured_learning_path(level, interests, goals):
    """Create structured learning path based on level and interests"""
    
    base_path = {
        "Beginner": [
            {"phase": "Foundation", "duration": "4-6 weeks", "topics": [
                "Python Programming Basics",
                "Mathematics for AI (Linear Algebra, Calculus, Statistics)",
                "Data Manipulation with Pandas and NumPy",
                "Data Visualization with Matplotlib/Seaborn"
            ]},
            {"phase": "ML Fundamentals", "duration": "6-8 weeks", "topics": [
                "Introduction to Machine Learning",
                "Supervised Learning (Regression, Classification)",
                "Unsupervised Learning (Clustering, Dimensionality Reduction)",
                "Model Evaluation and Validation"
            ]}
        ],
        "Intermediate": [
            {"phase": "Advanced ML", "duration": "8-10 weeks", "topics": [
                "Ensemble Methods",
                "Feature Engineering",
                "Hyperparameter Tuning",
                "ML Pipelines"
            ]},
            {"phase": "Deep Learning Basics", "duration": "6-8 weeks", "topics": [
                "Neural Networks Fundamentals",
                "TensorFlow/PyTorch Basics",
                "Convolutional Neural Networks",
                "Recurrent Neural Networks"
            ]}
        ],
        "Advanced": [
            {"phase": "Specialization", "duration": "10-12 weeks", "topics": [
                "Advanced Deep Learning Architectures",
                "Transfer Learning",
                "Generative Models",
                "Reinforcement Learning"
            ]},
            {"phase": "Production & Deployment", "duration": "8-10 weeks", "topics": [
                "MLOps Fundamentals",
                "Model Deployment",
                "Cloud Platforms (AWS, GCP, Azure)",
                "Docker and Kubernetes"
            ]}
        ]
    }
    
    # Add specialization based on interests
    specializations = []
    for interest in interests:
        if interest == "Natural Language Processing":
            specializations.append({
                "phase": "NLP Specialization",
                "duration": "8-10 weeks",
                "topics": ["Text Processing", "Word Embeddings", "Transformers", "BERT/GPT Models"]
            })
        elif interest == "Computer Vision":
            specializations.append({
                "phase": "Computer Vision Specialization", 
                "duration": "8-10 weeks",
                "topics": ["Image Processing", "Object Detection", "Image Segmentation", "GANs"]
            })
        elif interest == "Generative AI":
            specializations.append({
                "phase": "Generative AI Specialization",
                "duration": "10-12 weeks", 
                "topics": ["VAEs", "GANs", "Diffusion Models", "LLM Fine-tuning"]
            })
        elif interest == "Machine Learning":
            specializations.append({
                "phase": "Advanced ML Techniques",
                "duration": "8-10 weeks",
                "topics": ["Bayesian Methods", "Time Series Analysis", "Anomaly Detection", "Recommendation Systems"]
            })
    
    # Combine base path with specializations
    full_path = base_path.get(level, base_path["Beginner"])
    full_path.extend(specializations)
    
    return full_path

def estimate_timeline(time_commitment, level, num_interests):
    """Estimate learning timeline based on commitment and complexity"""
    
    base_hours = {
        "Beginner": 300,
        "Intermediate": 400, 
        "Advanced": 500
    }
    
    hours_per_week = {
        "5-10 hours": 7.5,
        "10-20 hours": 15,
        "20-30 hours": 25,
        "30+ hours": 35
    }
    
    total_hours = base_hours.get(level, 300) + (num_interests * 80)
    weekly_hours = hours_per_week.get(time_commitment, 15)
    
    weeks = max(8, round(total_hours / weekly_hours))
    months = round(weeks / 4)
    
    return {
        "total_hours": total_hours,
        "weekly_hours": weekly_hours,
        "estimated_weeks": weeks,
        "estimated_months": months
    }

def recommend_resources(level, interests, format_pref, budget):
    """Recommend learning resources based on preferences"""
    
    resources = {
        "free": {
            "courses": [
                "Coursera: Machine Learning by Andrew Ng",
                "Fast.ai: Practical Deep Learning",
                "Kaggle Learn: Micro-courses",
                "Google Machine Learning Crash Course"
            ],
            "platforms": [
                "YouTube: Sentdex, 3Blue1Brown",
                "GitHub: Open-source projects",
                "Hugging Face: NLP models and datasets",
                "Papers With Code: Research implementations"
            ]
        },
        "paid": {
            "courses": [
                "Udacity: AI Nanodegree",
                "Coursera: Deep Learning Specialization", 
                "edX: MIT MicroMasters in Statistics and Data Science",
                "Udemy: Complete Machine Learning Bootcamp"
            ],
            "platforms": [
                "DataCamp: Interactive learning",
                "Pluralsight: Skill paths",
                "LinkedIn Learning: Professional courses",
                "Brilliant: Math and science"
            ]
        }
    }
    
    # Specialized resources based on interests
    specialized = []
    for interest in interests:
        if interest == "Natural Language Processing":
            specialized.extend([
                "Hugging Face Transformers Course",
                "Stanford CS224N: Natural Language Processing",
                "spaCy Course: Advanced NLP"
            ])
        elif interest == "Computer Vision":
            specialized.extend([
                "CS231n: Convolutional Neural Networks for Visual Recognition",
                "PyImageSearch: Computer Vision Blog",
                "OpenCV Courses"
            ])
        elif interest == "Generative AI":
            specialized.extend([
                "Hugging Face Diffusion Models Course",
                "Stanford CS25: Transformers United",
                "Anthropic: LLM Safety and Alignment"
            ])
    
    selected_resources = resources["free"] if budget == "Free resources only" else {
        "courses": resources["free"]["courses"] + resources["paid"]["courses"],
        "platforms": resources["free"]["platforms"] + resources["paid"]["platforms"]
    }
    
    selected_resources["specialized"] = specialized
    
    return selected_resources

def generate_projects(level, interests):
    """Generate project ideas based on level and interests"""
    
    projects = {
        "Beginner": [
            "Predict House Prices using Linear Regression",
            "Customer Segmentation with K-Means Clustering",
            "Spam Email Classification",
            "Movie Recommendation System"
        ],
        "Intermediate": [
            "Image Classification with CNN",
            "Sentiment Analysis on Social Media Data",
            "Time Series Forecasting",
            "Customer Churn Prediction"
        ],
        "Advanced": [
            "Object Detection System",
            "Text Generation with Transformers",
            "Style Transfer with GANs",
            "Reinforcement Learning Agent for Games"
        ]
    }
    
    base_projects = projects.get(level, projects["Beginner"])
    
    # Add interest-specific projects
    interest_projects = []
    for interest in interests:
        if interest == "Natural Language Processing":
            interest_projects.extend([
                "Chatbot with Transformer Architecture",
                "Text Summarization System",
                "Named Entity Recognition Model"
            ])
        elif interest == "Computer Vision":
            interest_projects.extend([
                "Facial Recognition System",
                "Image Captioning Model", 
                "Real-time Object Detection"
            ])
        elif interest == "Generative AI":
            interest_projects.extend([
                "AI Art Generator with Stable Diffusion",
                "Text-to-Image Synthesis",
                "AI-powered Story Writer"
            ])
    
    return base_projects + interest_projects

def display_learning_path_recommendation(results):
    """Display comprehensive learning path recommendation"""
    
    st.markdown("---")
    st.header("🎯 Your Personalized AI Learning Path")
    
    # Display assessment summary
    st.subheader("📊 Learning Profile Summary")
    assessment = results["assessment"]
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Starting Level:** {results['starting_level']}")
        st.write(f"**Programming Experience:** {assessment['programming_level']}")
        st.write(f"**Mathematics Background:** {assessment['math_level']}")
        st.write(f"**Learning Goal:** {assessment['goals']}")
    
    with col2:
        st.write(f"**AI Interests:** {', '.join(assessment['interests'])}")
        st.write(f"**Current Skills:** {', '.join(assessment['current_skills'])}")
        st.write(f"**Time Commitment:** {assessment['time_commitment']}")
        st.write(f"**Budget:** {assessment['budget']}")
    
    # Display timeline
    st.subheader("⏱️ Estimated Timeline")
    timeline = results["timeline"]
    
    timeline_cols = st.columns(4)
    with timeline_cols[0]:
        st.metric("Total Hours", timeline["total_hours"])
    with timeline_cols[1]:
        st.metric("Weekly Commitment", f"{timeline['weekly_hours']} hrs")
    with timeline_cols[2]:
        st.metric("Estimated Weeks", timeline["estimated_weeks"])
    with timeline_cols[3]:
        st.metric("Estimated Months", timeline["estimated_months"])
    
    # Display learning path
    st.subheader("🛣️ Structured Learning Path")
    
    learning_path = results["learning_path"]
    for i, phase in enumerate(learning_path, 1):
        with st.expander(f"Phase {i}: {phase['phase']} ({phase['duration']})", expanded=i==1):
            st.write("**Topics to Master:**")
            for topic in phase["topics"]:
                st.write(f"• {topic}")
    
    # Display recommended resources
    st.subheader("📚 Recommended Learning Resources")
    resources = results["resources"]
    
    tab1, tab2, tab3 = st.tabs(["🎓 Courses", "🖥️ Platforms", "🎯 Specialized Resources"])
    
    with tab1:
        st.write("**Core Courses:**")
        for course in resources["courses"]:
            st.write(f"• {course}")
    
    with tab2:
        st.write("**Learning Platforms:**")
        for platform in resources["platforms"]:
            st.write(f"• {platform}")
    
    with tab3:
        if resources["specialized"]:
            st.write("**Interest-Specific Resources:**")
            for resource in resources["specialized"]:
                st.write(f"• {resource}")
        else:
            st.info("No specialized resources for your selected interests")
    
    # Display project ideas
    st.subheader("💡 Hands-on Project Ideas")
    projects = results["projects"]
    
    # Display projects in a nice grid
    cols = st.columns(2)
    for i, project in enumerate(projects):
        with cols[i % 2]:
            with st.container():
                st.info(f"**Project {i+1}:** {project}")
    
    # Learning tips
    st.subheader("🌟 Success Tips")
    tips = [
        "**Consistency is key** - Regular practice beats occasional intensive study",
        "**Build projects** - Apply concepts through hands-on implementation", 
        "**Join communities** - Engage with AI communities for support and networking",
        "**Read research papers** - Stay updated with latest developments",
        "**Participate in competitions** - Test your skills on platforms like Kaggle",
        "**Document your learning** - Maintain a portfolio of your projects and progress"
    ]
    
    for tip in tips:
        st.write(f"• {tip}")
    
    # Download learning plan
    st.subheader("💾 Download Your Learning Plan")
    learning_plan_text = generate_learning_plan_text(results)
    
    st.download_button(
        label="📥 Download Complete Learning Plan",
        data=learning_plan_text,
        file_name="ai_learning_path_plan.txt",
        mime="text/plain",
        use_container_width=True
    )
    
    # Reset button to start over
    st.markdown("---")
    if st.button("🔄 Start Over with New Assessment", use_container_width=True):
        st.session_state.learning_path_submitted = False
        st.session_state.learning_path_results = None
        st.rerun()

def generate_learning_plan_text(results):
    """Generate downloadable learning plan text"""
    
    plan = f"""
    AI LEARNING PATH RECOMMENDATION
    ================================
    
    Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
    
    LEARNING PROFILE:
    -----------------
    Starting Level: {results['starting_level']}
    Programming Experience: {results['assessment']['programming_level']}
    Mathematics Background: {results['assessment']['math_level']}
    AI Interests: {', '.join(results['assessment']['interests'])}
    Learning Goal: {results['assessment']['goals']}
    
    TIMELINE ESTIMATE:
    ------------------
    Total Learning Hours: {results['timeline']['total_hours']} hours
    Weekly Commitment: {results['timeline']['weekly_hours']} hours/week  
    Estimated Duration: {results['timeline']['estimated_weeks']} weeks ({results['timeline']['estimated_months']} months)
    
    STRUCTURED LEARNING PATH:
    -------------------------
    """
    
    for phase in results['learning_path']:
        plan += f"\n{phase['phase']} ({phase['duration']}):\n"
        for topic in phase['topics']:
            plan += f"  • {topic}\n"
    
    plan += """
    RECOMMENDED RESOURCES:
    ----------------------
    Courses:
    """
    
    for course in results['resources']['courses']:
        plan += f"  • {course}\n"
    
    plan += "\nPlatforms:\n"
    for platform in results['resources']['platforms']:
        plan += f"  • {platform}\n"
    
    if results['resources']['specialized']:
        plan += "\nSpecialized Resources:\n"
        for resource in results['resources']['specialized']:
            plan += f"  • {resource}\n"
    
    plan += """
    PROJECT IDEAS:
    --------------
    """
    
    for i, project in enumerate(results['projects'], 1):
        plan += f"{i}. {project}\n"
    
    plan += """
    SUCCESS TIPS:
    -------------
    1. Consistency is key - Regular practice beats occasional intensive study
    2. Build projects - Apply concepts through hands-on implementation
    3. Join communities - Engage with AI communities for support and networking  
    4. Read research papers - Stay updated with latest developments
    5. Participate in competitions - Test your skills on platforms like Kaggle
    6. Document your learning - Maintain a portfolio of your projects and progress
    
    Good luck on your AI learning journey!
    """
    
    return plan

# Make sure to call the function in your main app
if __name__ == "__main__":
    ai_learning_path_recommendation()
