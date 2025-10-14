import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import streamlit as st
import os

def load_and_preprocess_data():
    """Load and preprocess the dataset with proper error handling and analysis"""
    # Initialize variables with default values
    scaler = None
    label_encoders = {}
    target_encoder = None
    original_data = None
    
    try:
        # Try multiple possible file paths
        possible_paths = [
            'data/personalized_learning_dataset.csv',
            './data/personalized_learning_dataset.csv',
            'personalized_learning_dataset.csv',
            '../data/personalized_learning_dataset.csv'
        ]
        
        df = None
        file_path_used = None
        
        for file_path in possible_paths:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                file_path_used = file_path
                st.success(f"✅ Dataset loaded from: {file_path}")
                break
        
        if df is None:
            st.error("❌ Dataset file not found. Please ensure 'personalized_learning_dataset.csv' is in the data/ folder.")
            return pd.DataFrame(), {}
        
        # Store original data
        original_data = df.copy()
        
        # Display comprehensive dataset analysis
        display_comprehensive_analysis(df)
        
        # Handle missing values
        st.subheader("🔧 Data Preprocessing Steps")
        
        # Create a copy for preprocessing
        df_processed = df.copy()
        
        # Drop Student_ID as it's not useful for ML
        if 'Student_ID' in df_processed.columns:
            df_processed = df_processed.drop('Student_ID', axis=1)
            st.info("✅ Dropped 'Student_ID' column as it's not useful for ML")
        
        # Handle missing values
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        missing_before = df_processed.isnull().sum().sum()
        
        if len(numeric_cols) > 0:
            imputer_num = SimpleImputer(strategy='median')
            df_processed[numeric_cols] = imputer_num.fit_transform(df_processed[numeric_cols])
        
        if len(categorical_cols) > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            df_processed[categorical_cols] = imputer_cat.fit_transform(df_processed[categorical_cols])
        
        missing_after = df_processed.isnull().sum().sum()
        st.info(f"✅ Handled missing values: {missing_before} → {missing_after}")
        
        # Encode categorical variables
        if len(categorical_cols) > 0:
            encoding_info = {}
            for col in categorical_cols:
                if col != 'Dropout_Likelihood':  # Don't encode target yet
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    label_encoders[col] = le
                    encoding_info[col] = dict(zip(le.classes_, le.transform(le.classes_)))
            
            # Display encoding information
            with st.expander("🔤 Categorical Variable Encoding"):
                for col, mapping in encoding_info.items():
                    st.write(f"**{col}:**")
                    for category, code in mapping.items():
                        st.write(f"  - {category} → {code}")
        
        # Encode target variable
        if 'Dropout_Likelihood' in df_processed.columns:
            target_encoder = LabelEncoder()
            df_processed['Dropout_encoded'] = target_encoder.fit_transform(df_processed['Dropout_Likelihood'])
            dropout_count = df_processed['Dropout_encoded'].value_counts()
            dropout_mapping = dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))
            st.info(f"✅ Encoded target: {dropout_mapping} - Distribution: {dropout_count[0]} No, {dropout_count[1]} Yes")
        else:
            st.error("❌ 'Dropout_Likelihood' column not found in dataset!")
            return pd.DataFrame(), {}
        
        # Scale numerical features if any exist
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df_processed[numeric_cols])
            df_processed[numeric_cols] = scaled_features
            st.info(f"✅ Scaled {len(numeric_cols)} numeric features (StandardScaler)")
        
        st.success("🎉 Data preprocessing completed successfully!")
        
        # Show processed data sample
        st.subheader("📋 Processed Data (First 5 rows)")
        st.dataframe(df_processed.head())
        
        return df_processed, {
            'scaler': scaler, 
            'label_encoders': label_encoders, 
            'target_encoder': target_encoder,
            'original_data': original_data
        }
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return pd.DataFrame(), {}

def display_comprehensive_analysis(df):
    """Display comprehensive analysis of the dataset"""
    st.header("📊 Comprehensive Dataset Analysis")
    
    # Basic Information
    st.subheader("📋 Basic Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        st.metric("Number of Features", len(df.columns))
    with col3:
        if 'Dropout_Likelihood' in df.columns:
            dropout_rate = (df['Dropout_Likelihood'] == 'Yes').mean() * 100
            st.metric("Dropout Rate", f"{dropout_rate:.1f}%")
        else:
            st.metric("Dropout Rate", "N/A")
    with col4:
        numeric_features = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("Numeric Features", numeric_features)
    
    # Dataset Overview
    st.subheader("🔍 Dataset Overview")
    st.write(f"**Shape:** {df.shape} (rows × columns)")
    st.write(f"**Columns:** {list(df.columns)}")
    
    # Display first few rows
    st.write("**First 5 rows of original data:**")
    st.dataframe(df.head())
    
    # Data Types and Missing Values
    st.subheader("📈 Data Types & Quality")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Data Types Summary:**")
        dtype_summary = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum()
        })
        st.dataframe(dtype_summary)
    
    with col2:
        st.write("**Missing Values Analysis:**")
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Values': missing_data.values,
                'Percentage': (missing_data.values / len(df)) * 100
            })
            missing_df = missing_df[missing_df['Missing Values'] > 0]
            st.dataframe(missing_df)
        else:
            st.success("✅ No missing values found")
    
    # Statistical Analysis - Show original values (not scaled)
    st.subheader("📊 Statistical Analysis (Original Values)")
    
    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.write("**Numerical Features Summary:**")
        numeric_summary = df[numeric_cols].describe().T
        numeric_summary['variance'] = df[numeric_cols].var()
        numeric_summary['skewness'] = df[numeric_cols].skew()
        st.dataframe(numeric_summary.style.format("{:.2f}"))
    
    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.write("**Categorical Features Summary:**")
        for col in categorical_cols:
            with st.expander(f"📊 {col} Distribution"):
                value_counts = df[col].value_counts()
                st.write(f"**Unique values:** {df[col].nunique()}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Value Counts:")
                    st.dataframe(value_counts)
                with col2:
                    if len(value_counts) <= 15:  # Only plot if not too many categories
                        import plotly.express as px
                        fig = px.bar(x=value_counts.index, y=value_counts.values,
                                    title=f'Distribution of {col}',
                                    labels={'x': col, 'y': 'Count'})
                        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Analysis (for numeric columns)
    if len(numeric_cols) > 1:
        st.subheader("📈 Correlation Analysis")
        correlation_matrix = df[numeric_cols].corr()
        
        # Create heatmap
        import plotly.express as px
        fig = px.imshow(correlation_matrix,
                       text_auto=True,
                       aspect="auto",
                       color_continuous_scale='RdBu_r',
                       title='Feature Correlation Matrix',
                       width=800, height=600)
        st.plotly_chart(fig)
    
    # Target Variable Analysis
    if 'Dropout_Likelihood' in df.columns:
        st.subheader("🎯 Target Variable Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution
            dropout_counts = df['Dropout_Likelihood'].value_counts()
            fig = px.pie(values=dropout_counts.values,
                        names=dropout_counts.index,
                        title='Dropout Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Dropout rates by categorical variables
            st.write("**Dropout Rates by Category:**")
            
            # By Gender
            if 'Gender' in df.columns:
                gender_dropout = df.groupby('Gender')['Dropout_Likelihood'].apply(
                    lambda x: (x == 'Yes').mean() * 100
                ).round(1)
                st.write("**By Gender:**")
                for gender, rate in gender_dropout.items():
                    st.write(f"- {gender}: {rate}%")
            
            # By Course
            if 'Course_Name' in df.columns:
                course_dropout = df.groupby('Course_Name')['Dropout_Likelihood'].apply(
                    lambda x: (x == 'Yes').mean() * 100
                ).round(1)
                st.write("**By Course:**")
                for course, rate in course_dropout.items():
                    st.write(f"- {course}: {rate}%")
            
            # By Education Level
            if 'Education_Level' in df.columns:
                education_dropout = df.groupby('Education_Level')['Dropout_Likelihood'].apply(
                    lambda x: (x == 'Yes').mean() * 100
                ).round(1)
                st.write("**By Education Level:**")
                for education, rate in education_dropout.items():
                    st.write(f"- {education}: {rate}%")
    
    # Data Quality Assessment
    st.subheader("🔍 Data Quality Assessment")
    
    quality_metrics = {
        "Completeness": f"{(1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.1f}%",
        "Duplicate Rows": f"{df.duplicated().sum()} ({df.duplicated().sum() / len(df) * 100:.1f}%)",
        "Constant Columns": f"{len([col for col in df.columns if df[col].nunique() == 1])}",
        "High Cardinality Features": f"{len([col for col in categorical_cols if df[col].nunique() > 50])}"
    }
    
    for metric, value in quality_metrics.items():
        st.write(f"**{metric}:** {value}")
    
    # Feature Insights
    st.subheader("💡 Feature Insights")
    
    if 'Age' in df.columns:
        st.write(f"**Age Range:** {df['Age'].min()} - {df['Age'].max()} years")
    
    if 'Quiz_Scores' in df.columns:
        st.write(f"**Quiz Scores:** {df['Quiz_Scores'].min()} - {df['Quiz_Scores'].max()} points")
    
    if 'Final_Exam_Score' in df.columns:
        st.write(f"**Final Exam Scores:** {df['Final_Exam_Score'].min()} - {df['Final_Exam_Score'].max()} points")
    
    if 'Time_Spent_on_Videos' in df.columns:
        st.write(f"**Time Spent on Videos:** {df['Time_Spent_on_Videos'].min()} - {df['Time_Spent_on_Videos'].max()} minutes")
