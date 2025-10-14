import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing import load_and_preprocess_data
from src.visualization import (
    display_dataset_statistics, 
    plot_correlation_heatmap, 
    plot_histograms, 
    plot_pca_scatter, 
    plot_feature_importance, 
    plot_student_analytics
)
from src.clustering import (
    perform_clustering, 
    plot_clustering_results, 
    print_clustering_details
)
from src.classification import (
    train_and_evaluate_models, 
    display_model_comparison,
    run_dropout_prediction,
    DropoutPredictor,
    NEURAL_NETWORKS_AVAILABLE
)
from src.prediction import predict_new_student

# Configure the Streamlit app
st.set_page_config(
    page_title="Student Analytics System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2e86ab;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .neural-network-badge {
        background-color: #ff6b6b;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .learning-path-button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        margin: 10px 0;
    }
    .learning-path-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

    # Main header
    st.markdown('<h1 class="main-header">🎓 Student Dropout Prediction & Career Guidance System</h1>', unsafe_allow_html=True)
    st.markdown("### Comprehensive ML system for student analytics and career recommendations")
    
    # Sidebar navigation with enhanced styling
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧭 Navigation")
    app_mode = st.sidebar.selectbox("Choose Section", [
        "🏠 Dashboard Overview",
        "📁 Dataset Overview",
        "🎓 Student Analytics", 
        "📊 Data Visualization",
        "👥 Clustering Analysis",
        "🔮 Classification Models",
        "🎯 New Student Prediction",
        "🤖 AI Learning Paths",
        "ℹ️ About"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 System Info")
    
    # Load data with error handling and caching
    @st.cache_data(show_spinner="Loading and preprocessing data...")
    def load_data():
        try:
            return load_and_preprocess_data()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame(), {}
    
    # Load data
    with st.spinner('🔄 Loading dataset and preprocessing...'):
        df, preprocessor = load_data()
    
    # Display system info in sidebar
    if not df.empty:
        st.sidebar.success(f"✅ Data loaded successfully")
        st.sidebar.metric("Total Students", len(df))
        st.sidebar.metric("Number of Features", len(df.columns))
        if 'Dropout_encoded' in df.columns:
            dropout_rate = df['Dropout_encoded'].mean() * 100
            st.sidebar.metric("Dropout Rate", f"{dropout_rate:.1f}%")
    else:
        st.sidebar.error("❌ No data loaded")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛠️ Quick Actions")
    
    # Quick actions in sidebar
    if st.sidebar.button("🔄 Reload Data"):
        st.cache_data.clear()
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with ❤️ using Streamlit")
    
    try:
        # Handle different app modes
        if app_mode == "🏠 Dashboard Overview":
            display_dashboard_overview(df, preprocessor)
            
        elif app_mode == "📁 Dataset Overview":
            display_dataset_overview(df, preprocessor)
            
        elif app_mode == "🎓 Student Analytics":
            display_student_analytics(df, preprocessor)
            
        elif app_mode == "📊 Data Visualization":
            display_data_visualization(df)
            
        elif app_mode == "👥 Clustering Analysis":
            display_clustering_analysis(df)
            
        elif app_mode == "🔮 Classification Models":
            display_classification_models(df)
            
        elif app_mode == "🎯 New Student Prediction":
            display_prediction_section(df, preprocessor)
            
        elif app_mode == "🤖 AI Learning Paths":
            display_learning_paths_section()
            
        elif app_mode == "ℹ️ About":
            display_about_section()
            
    except Exception as e:
        st.error(f"🚨 An error occurred in {app_mode}: {str(e)}")
        st.info("💡 Please check your data and try again. If the problem persists, try reloading the data.")
        
        # Debug information
        with st.expander("🔧 Debug Information"):
            st.write("**Error Details:**", str(e))
            if not df.empty:
                st.write("**DataFrame Shape:**", df.shape)
                st.write("**DataFrame Columns:**", list(df.columns))
            else:
                st.write("**DataFrame:** Empty")

def display_dashboard_overview(df, preprocessor):
    """Display dashboard overview with key metrics and insights"""
    st.markdown('<h2 class="section-header">🏠 Dashboard Overview</h2>', unsafe_allow_html=True)
    
    if df.empty:
        st.error("❌ No data available for dashboard")
        return
    
    # Key metrics row
    st.subheader("📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_students = len(df)
        st.metric("Total Students", total_students)
    
    with col2:
        if 'Dropout_encoded' in df.columns:
            dropout_rate = df['Dropout_encoded'].mean() * 100
            st.metric("Dropout Rate", f"{dropout_rate:.1f}%")
        else:
            st.metric("Dropout Rate", "N/A")
    
    with col3:
        numeric_features = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("Numeric Features", numeric_features)
    
    with col4:
        categorical_features = len(df.select_dtypes(include=['object']).columns)
        st.metric("Categorical Features", categorical_features)
    
    # Quick insights
    st.subheader("💡 Quick Insights")
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.write("**📊 Data Overview**")
        if not df.empty:
            st.write(f"- Dataset shape: {df.shape}")
            st.write(f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Check for missing values
            missing_values = df.isnull().sum().sum()
            if missing_values > 0:
                st.warning(f"- Missing values: {missing_values}")
            else:
                st.success("- No missing values")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col6:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.write("**🎯 Model Readiness**")
        if 'Dropout_encoded' in df.columns:
            st.success("- Target variable available")
            # Check class balance
            class_balance = df['Dropout_encoded'].value_counts(normalize=True)
            if abs(class_balance[0] - class_balance[1]) > 0.7:
                st.warning("- Imbalanced classes detected")
            else:
                st.success("- Balanced classes")
        else:
            st.error("- Target variable missing")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent activity or sample data
    st.subheader("📋 Data Sample")
    if not df.empty:
        st.dataframe(df.head(10), use_container_width=True)

def display_dataset_overview(df, preprocessor):
    """Display dataset overview and statistics"""
    st.markdown('<h2 class="section-header">📁 Dataset Overview</h2>', unsafe_allow_html=True)
    
    if df.empty:
        st.error("❌ No data available for analysis")
        return
    
    display_dataset_statistics(df)

def display_student_analytics(df, preprocessor):
    """Display student analytics"""
    st.markdown('<h2 class="section-header">🎓 Student Analytics</h2>', unsafe_allow_html=True)
    
    if df.empty:
        st.error("❌ No data available for analytics")
        return
    
    # Use original data for analytics to see actual values
    if preprocessor and 'original_data' in preprocessor:
        plot_student_analytics(preprocessor['original_data'])
    else:
        plot_student_analytics(df)

def display_data_visualization(df):
    """Display data visualization section"""
    st.markdown('<h2 class="section-header">📊 Data Visualization & Analysis</h2>', unsafe_allow_html=True)
    
    if df.empty:
        st.error("❌ No data available for visualization")
        return
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Correlation Analysis", 
        "📊 Distributions", 
        "🔍 PCA Analysis", 
        "🎯 Feature Importance"
    ])
    
    with tab1:
        plot_correlation_heatmap(df)
    
    with tab2:
        plot_histograms(df)
    
    with tab3:
        plot_pca_scatter(df)
    
    with tab4:
        plot_feature_importance(df)

def display_clustering_analysis(df):
    """Display clustering analysis section"""
    st.markdown('<h2 class="section-header">👥 Student Clustering Analysis</h2>', unsafe_allow_html=True)
    
    if df.empty:
        st.error("❌ No data available for clustering")
        return
    
    # Perform clustering
    with st.spinner('🔄 Performing student clustering...'):
        cluster_results = perform_clustering(df)
    
    if cluster_results:
        cluster_labels, pca_result = cluster_results
        
        # Display clustering results
        with st.spinner('📊 Generating cluster visualizations...'):
            plot_clustering_results(df, cluster_labels, pca_result)
        
        # Display clustering details
        with st.spinner('🔍 Analyzing cluster details...'):
            print_clustering_details(df, cluster_labels)

def display_classification_models(df):
    """Display classification models section"""
    st.markdown('<h2 class="section-header">🔮 Dropout Prediction Models</h2>', unsafe_allow_html=True)
    
    if df.empty:
        st.error("❌ No data available for model training")
        return
    
    if 'Dropout_encoded' not in df.columns:
        st.error("❌ Target variable 'Dropout_encoded' not found in dataset")
        return
    
    # Display neural network availability
    if NEURAL_NETWORKS_AVAILABLE:
        st.success("🧠 Neural Networks: Available (TensorFlow)")
    
    # Model training section
    st.markdown("### 🚀 Model Training")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("""
        **Models to be trained:**
        - Logistic Regression
        - Random Forest
        - SVM
        - K-Nearest Neighbors
        - Decision Tree
        - XGBoost
        - Gradient Boosting
        
        Each model will be evaluated using Accuracy, Precision, Recall, F1-Score, AUC, and Cross-Validation.
        """)
    
    with col2:
        if st.button("🎯 Train Models", type="primary", use_container_width=True):
            with st.spinner('Training and evaluating models... This may take a few minutes.'):
                try:
                    predictor, results_df = train_and_evaluate_models()
                    
                    if predictor is not None and results_df is not None and not results_df.empty:
                        st.session_state.predictor = predictor
                        st.session_state.results_df = results_df
                        st.success("✅ Model training completed successfully!")
                    else:
                        st.error("❌ Model training failed. Please check the data and try again.")
                        
                except Exception as e:
                    st.error(f"❌ Error during model training: {str(e)}")
                    st.info("💡 Try reloading the data or checking your dataset format.")
    
    # Display results if available
    if 'predictor' in st.session_state and 'results_df' in st.session_state:
        if not st.session_state.results_df.empty:
            display_detailed_model_results(st.session_state.predictor, st.session_state.results_df)
        else:
            st.error("❌ No model results available. Please try training again.")
    else:
        st.warning("👆 Click 'Train Models' to start model training and evaluation.")

def display_detailed_model_results(predictor, results_df):
    """Display detailed results for all trained models"""
    st.markdown("### 📊 Detailed Model Performance")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Summary Table", 
        "📈 Performance Charts", 
        "🏆 Best Model", 
        "🔍 Model Details"
    ])
    
    with tab1:
        display_model_summary_table(results_df)
    
    with tab2:
        display_performance_charts(results_df)
    
    with tab3:
        display_best_model_details(results_df, predictor)
    
    with tab4:
        display_individual_model_details(results_df, predictor)

def display_model_summary_table(results_df):
    """Display comprehensive model comparison table"""
    st.subheader("📋 Model Performance Summary")
    
    # Create a formatted dataframe for display
    display_df = results_df.copy()
    
    # Convert all numeric columns to proper format
    numeric_columns = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC_Score', 'CV_Score', 'CV_Std', 'Fit_Time']
    
    for col in numeric_columns:
        if col in display_df.columns:
            if col == 'Fit_Time':
                display_df[col] = display_df[col].apply(
                    lambda x: f'{float(x):.3f}s' if isinstance(x, (int, float)) else str(x)
                )
            elif col == 'CV_Std':
                display_df[col] = display_df[col].apply(
                    lambda x: f'±{float(x):.4f}' if isinstance(x, (int, float)) else str(x)
                )
            else:
                display_df[col] = display_df[col].apply(
                    lambda x: f'{float(x):.4f}' if isinstance(x, (int, float)) else str(x)
                )
    
    # Highlight Neural Network
    def highlight_neural_network(row):
        if row['Model'] == 'Neural Network':
            return ['background-color: #ffebee'] * len(row)
        return [''] * len(row)
    
    styled_df = display_df.style.apply(highlight_neural_network, axis=1)
    
    # Display the table
    st.dataframe(styled_df, use_container_width=True)
    
    # Add some statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'Accuracy' in results_df.columns:
            avg_accuracy = results_df['Accuracy'].mean()
            st.metric("Average Accuracy", f"{avg_accuracy:.4f}")
    
    with col2:
        if 'F1_Score' in results_df.columns:
            avg_f1 = results_df['F1_Score'].mean()
            st.metric("Average F1-Score", f"{avg_f1:.4f}")
    
    with col3:
        if 'CV_Score' in results_df.columns:
            avg_cv = results_df['CV_Score'].mean()
            st.metric("Average CV Score", f"{avg_cv:.4f}")
    
    with col4:
        if 'AUC_Score' in results_df.columns:
            valid_auc = results_df[results_df['AUC_Score'] != 'N/A']
            if not valid_auc.empty:
                avg_auc = valid_auc['AUC_Score'].mean()
                st.metric("Average AUC", f"{avg_auc:.4f}")

def display_performance_charts(results_df):
    """Display performance comparison charts"""
    st.subheader("📈 Performance Visualization")
    
    # Convert to numeric for plotting
    plot_df = results_df.copy()
    numeric_columns = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC_Score', 'CV_Score', 'Fit_Time']
    
    for col in numeric_columns:
        if col in plot_df.columns:
            plot_df[col] = plot_df[col].apply(
                lambda x: float(x) if isinstance(x, (int, float)) else 0 if x == 'N/A' else 0
            )
    
    # Create charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Accuracy & F1-Score Comparison**")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(plot_df))
        width = 0.35
        
        # Use different colors for Neural Network
        colors = ['#2E86AB' if model != 'Neural Network' else '#FF6B6B' for model in plot_df['Model']]
        
        bars1 = ax.bar(x - width/2, plot_df['Accuracy'], width, label='Accuracy', alpha=0.8, color=colors)
        bars2 = ax.bar(x + width/2, plot_df['F1_Score'], width, label='F1-Score', alpha=0.8, color=colors)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Scores')
        ax.set_title('Accuracy vs F1-Score Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Precision & Recall Comparison**")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(plot_df))
        width = 0.35
        
        colors = ['#2E86AB' if model != 'Neural Network' else '#FF6B6B' for model in plot_df['Model']]
        
        bars1 = ax.bar(x - width/2, plot_df['Precision'], width, label='Precision', alpha=0.8, color=colors)
        bars2 = ax.bar(x + width/2, plot_df['Recall'], width, label='Recall', alpha=0.8, color=colors)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Scores')
        ax.set_title('Precision vs Recall Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)

def display_best_model_details(results_df, predictor):
    """Display details about the best performing model"""
    st.subheader("🏆 Best Performing Model")
    
    # Find the best model based on accuracy
    if 'Accuracy' in results_df.columns:
        best_model_idx = results_df['Accuracy'].idxmax()
        best_model = results_df.loc[best_model_idx]
        
        # Display best model card
        st.markdown(f"### 🎯 {best_model['Model']}")
        
        if best_model['Model'] == 'Neural Network':
            st.markdown('<span class="neural-network-badge">NEURAL NETWORK</span>', unsafe_allow_html=True)
        
        # Create metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Accuracy", f"{best_model['Accuracy']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("F1-Score", f"{best_model['F1_Score']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Precision", f"{best_model['Precision']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Recall", f"{best_model['Recall']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional metrics
        col5, col6, col7 = st.columns(3)
        
        with col5:
            cv_score = best_model['CV_Score']
            if isinstance(cv_score, (int, float)):
                st.metric("CV Score", f"{cv_score:.4f}")
        
        with col6:
            auc_score = best_model.get('AUC_Score', 'N/A')
            if isinstance(auc_score, (int, float)):
                st.metric("AUC Score", f"{auc_score:.4f}")
        
        with col7:
            fit_time = best_model.get('Fit_Time', 'N/A')
            if isinstance(fit_time, (int, float)):
                st.metric("Training Time", f"{fit_time:.3f}s")
        
        # Best model insights
        st.markdown("### 💡 Best Model Insights")
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.markdown("""
            **🎯 Performance Highlights:**
            - Highest accuracy among all models
            - Best generalization capability
            - Recommended for predictions
            - Balanced precision and recall
            """)
        
        with insight_col2:
            st.markdown("""
            **🚀 Recommended Usage:**
            - Use for new student predictions
            - Most reliable for dropout risk assessment
            - Suitable for production deployment
            - Well-balanced performance metrics
            """)

def display_individual_model_details(results_df, predictor):
    """Display detailed information for each individual model"""
    st.subheader("🔍 Individual Model Details")
    
    # Model selector
    selected_model = st.selectbox(
        "Select a model to view detailed information:",
        results_df['Model'].tolist()
    )
    
    if selected_model:
        model_data = results_df[results_df['Model'] == selected_model].iloc[0]
        
        # Display model card
        st.markdown(f"### {selected_model}")
        
        if selected_model == 'Neural Network':
            st.markdown('<span class="neural-network-badge">DEEP LEARNING</span>', unsafe_allow_html=True)
        
        # Performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Performance Metrics:**")
            st.write(f"- **Accuracy:** {model_data['Accuracy']:.4f}")
            st.write(f"- **Precision:** {model_data['Precision']:.4f}")
            st.write(f"- **Recall:** {model_data['Recall']:.4f}")
            st.write(f"- **F1-Score:** {model_data['F1_Score']:.4f}")
        
        with col2:
            st.markdown("**📈 Additional Metrics:**")
            
            cv_score = model_data['CV_Score']
            if isinstance(cv_score, (int, float)):
                st.write(f"- **Cross-Validation:** {cv_score:.4f}")
            else:
                st.write(f"- **Cross-Validation:** {cv_score}")
            
            auc_score = model_data.get('AUC_Score', 'N/A')
            if isinstance(auc_score, (int, float)):
                st.write(f"- **AUC Score:** {auc_score:.4f}")
            else:
                st.write(f"- **AUC Score:** {auc_score}")
            
            fit_time = model_data.get('Fit_Time', 'N/A')
            if isinstance(fit_time, (int, float)):
                st.write(f"- **Training Time:** {fit_time:.3f}s")
            else:
                st.write(f"- **Training Time:** {fit_time}")
        
        # Model ranking
        st.markdown("**📈 Model Ranking:**")
        
        # Calculate rankings
        metrics = ['Accuracy', 'F1_Score', 'Precision', 'Recall', 'CV_Score']
        rankings = {}
        
        for metric in metrics:
            if metric in results_df.columns:
                rank = (results_df[metric].rank(ascending=False, method='min')[results_df['Model'] == selected_model]).iloc[0]
                rankings[metric] = int(rank)
        
        rank_cols = st.columns(len(rankings))
        
        for i, (metric, rank) in enumerate(rankings.items()):
            with rank_cols[i]:
                st.metric(f"{metric} Rank", f"#{rank}")

def display_prediction_section(df, preprocessor):
    """Display new student prediction section"""
    st.markdown('<h2 class="section-header">🎯 New Student Prediction</h2>', unsafe_allow_html=True)
    
    if df.empty:
        st.error("❌ No data available for prediction")
        return
    
    if 'Dropout_encoded' not in df.columns:
        st.error("❌ Target variable not found. Please train models first.")
        return
    
    # Check if models are trained
    if 'predictor' not in st.session_state:
        st.warning("""
        ⚠️ **Models not trained yet!**
        
        Before making predictions, please:
        1. Go to the **🔮 Classification Models** section
        2. Click the **'Train Models'** button
        3. Return here to make predictions
        """)
        
        if st.button("🚀 Go to Model Training"):
            st.session_state.navigation = "Classification Models"
            st.rerun()
        
        return
    
    # Make prediction - this will now automatically show learning paths for high-risk students
    st.info("🔍 Fill out the form below to predict dropout risk for a new student.")
    prediction_result = predict_new_student(df, preprocessor)

def display_learning_paths_section():
    """Display dedicated section for AI learning paths"""
    st.markdown('<h2 class="section-header">🤖 AI Learning Paths</h2>', unsafe_allow_html=True)
    
    st.info("""
    **Explore personalized learning paths for different career domains.**
    Select your area of interest and generate a customized learning roadmap.
    """)
    
    # Get student interest with button
    selected_interest = get_student_interests()
    
    if selected_interest:
        # Display personalized recommendations (assuming high dropout risk for dedicated section)
        provide_personalized_recommendations(selected_interest, 1)
    
    # Show all available learning paths
    st.markdown("---")
    st.subheader("📚 Available Career Paths")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🌐 Web Development**
        - Frontend & Backend Development
        - Modern frameworks and tools
        - Full project lifecycle
        
        **🔒 Cybersecurity**
        - Network security & ethical hacking
        - Security tools and frameworks
        - Threat detection and prevention
        """)
    
    with col2:
        st.markdown("""
        **📊 Data Analytics**
        - Data processing & visualization
        - Statistical analysis & ML basics
        - Business intelligence tools
        
        **🚀 Full Stack Development**
        - End-to-end web development
        - Database management & APIs
        - Deployment & DevOps basics
        """)

def display_about_section():
    """Display about section"""
    st.markdown('<h2 class="section-header">ℹ️ About This Project</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>🎯 Project Overview</h3>
    <p>This comprehensive Machine Learning system provides intelligent analytics and predictions for student dropout risk, 
    learning style clustering, and career guidance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚀 Key Features
        
        - **📊 Data Analytics**: Comprehensive EDA and visualization
        - **👥 Student Clustering**: Group students by learning styles
        - **🔮 Dropout Prediction**: ML models to predict at-risk students
        - **🧠 Neural Networks**: Deep learning for advanced predictions
        - **🎯 Career Guidance**: Personalized recommendations for dropouts
        - **📈 Interactive Dashboard**: User-friendly Streamlit interface
        
        ### 🛠️ Technical Stack
        
        - **Framework**: Streamlit
        - **ML Library**: Scikit-learn, TensorFlow/Keras
        - **Data Processing**: Pandas, NumPy
        - **Visualization**: Plotly, Matplotlib, Seaborn
        - **Model Persistence**: Joblib
        """)
    
    with col2:
        st.markdown("""
        ### 📋 Dataset Features
        
        **Demographic Information:**
        - Age, Gender, Education Level
        
        **Academic Performance:**
        - Quiz Scores, Exam Scores, Assignment Completion
        - Time Spent on Videos, Forum Participation
        
        **Engagement Metrics:**
        - Engagement Level, Feedback Scores
        - Learning Style Preferences
        
        **Target Variable:**
        - Dropout Likelihood (Yes/No)
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    </div>
    """, unsafe_allow_html=True)

# Initialize session state variables
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None


if __name__ == "__main__":
    main()
