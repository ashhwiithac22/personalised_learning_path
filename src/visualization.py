import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np

def display_dataset_statistics(df):
    """Display basic dataset statistics with error handling"""
    st.header("📈 Dataset Statistics")
    
    if df.empty or len(df.columns) == 0:
        st.error("❌ No data available to display statistics")
        return
    
    try:
        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Students", len(df))
        with col2:
            st.metric("Number of Features", len(df.columns))
        with col3:
            if 'Dropout_encoded' in df.columns:
                dropout_rate = df['Dropout_encoded'].mean() * 100
                st.metric("Dropout Rate", f"{dropout_rate:.1f}%")
            else:
                st.metric("Dropout Rate", "N/A")
        with col4:
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Features", numeric_cols)
        
        # Dataset overview
        st.subheader("📊 Numerical Features Summary")
        st.dataframe(df.describe())
        
        # Categorical features summary
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            st.subheader("📊 Categorical Features Summary")
            for col in categorical_cols:
                st.write(f"**{col}:**")
                st.write(df[col].value_counts())
        
    except Exception as e:
        st.error(f"Error displaying statistics: {str(e)}")

def plot_correlation_heatmap(df):
    """Plot correlation heatmap with error handling"""
    st.subheader("📊 Correlation Heatmap")
    
    try:
        # Select only numeric columns for correlation
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            st.warning("Need at least 2 numeric columns for correlation heatmap")
            return
        
        # Calculate correlation matrix
        correlation_matrix = numeric_df.corr()
        
        # Create plotly figure for better interactivity
        fig = px.imshow(correlation_matrix,
                       text_auto=True,
                       aspect="auto",
                       color_continuous_scale='RdBu_r',
                       title='Feature Correlation Heatmap')
        
        fig.update_layout(height=600)
        st.plotly_chart(fig)
        
        # Show high correlations with target
        if 'Dropout_encoded' in correlation_matrix.columns:
            st.subheader("🔍 Top Correlations with Dropout")
            dropout_corr = correlation_matrix['Dropout_encoded'].drop('Dropout_encoded').sort_values(key=abs, ascending=False)
            top_correlations = dropout_corr.head(10)
            
            fig_bar = px.bar(x=top_correlations.values, 
                           y=top_correlations.index,
                           orientation='h',
                           title='Top Features Correlated with Dropout',
                           labels={'x': 'Correlation', 'y': 'Feature'})
            st.plotly_chart(fig_bar)
        
    except Exception as e:
        st.error(f"Error creating correlation heatmap: {str(e)}")

def plot_histograms(df):
    """Plot histograms for numeric features"""
    st.subheader("📋 Feature Distributions")
    
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            st.warning("No numeric columns available for histograms")
            return
        
        # Plot distributions for all numeric features
        for col in numeric_cols[:6]:  # Show first 6 features to avoid clutter
            if col != 'Dropout_encoded':
                fig, ax = plt.subplots(figsize=(8, 4))
                df[col].hist(bins=30, ax=ax, alpha=0.7, color='skyblue')
                ax.set_title(f'Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error creating histogram: {str(e)}")

def plot_pca_scatter(df):
    """Plot PCA scatter plot with error handling"""
    st.subheader("🔍 PCA Analysis")
    
    try:
        # Exclude target variable for PCA
        features_df = df.select_dtypes(include=[np.number])
        if 'Dropout_encoded' in features_df.columns:
            features_df = features_df.drop('Dropout_encoded', axis=1)
        
        if len(features_df.columns) < 2:
            st.warning("Need at least 2 numeric features for PCA")
            return
        
        # Perform PCA
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(features_df)
        
        # Create DataFrame for plotting
        pca_df = pd.DataFrame({
            'PC1': pca_components[:, 0],
            'PC2': pca_components[:, 1]
        })
        
        # Add target variable for coloring if available
        if 'Dropout_encoded' in df.columns:
            pca_df['Dropout'] = df['Dropout_encoded'].map({0: 'No', 1: 'Yes'})
            color_col = 'Dropout'
            title = 'PCA Scatter Plot (Colored by Dropout Status)'
        else:
            color_col = None
            title = 'PCA Scatter Plot'
        
        # Create plot
        fig = px.scatter(pca_df, x='PC1', y='PC2', color=color_col,
                        title=title,
                        labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                               'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'})
        
        st.plotly_chart(fig)
        
        # Display PCA explained variance
        st.write(f"**Explained Variance:**")
        st.write(f"- PC1: {pca.explained_variance_ratio_[0]:.2%}")
        st.write(f"- PC2: {pca.explained_variance_ratio_[1]:.2%}")
        st.write(f"- **Total:** {pca.explained_variance_ratio_.sum():.2%}")
        
    except Exception as e:
        st.error(f"Error performing PCA: {str(e)}")

def plot_feature_importance(df):
    """Plot feature importance based on correlation with target"""
    st.subheader("🎯 Feature Importance (Correlation with Dropout)")
    
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        
        if 'Dropout_encoded' not in numeric_df.columns:
            st.warning("Dropout target not available for feature importance")
            return
        
        # Calculate correlation with target
        correlations = numeric_df.corr()['Dropout_encoded'].drop('Dropout_encoded')
        correlations = correlations.sort_values(key=abs, ascending=False)
        
        # Create horizontal bar chart
        fig = px.bar(x=correlations.values, 
                    y=correlations.index,
                    orientation='h',
                    title='Feature Correlation with Dropout',
                    color=correlations.values,
                    color_continuous_scale='RdBu_r',
                    labels={'x': 'Correlation Coefficient', 'y': 'Features'})
        
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error plotting feature importance: {str(e)}")

def plot_student_analytics(df):
    """Create comprehensive analytics for student data"""
    st.header("🎓 Student Performance Analytics")
    
    if df.empty:
        st.error("No data available for analytics")
        return
    
    try:
        # Dropout distribution
        if 'Dropout_Likelihood' in df.columns:
            st.subheader("📊 Dropout Distribution")
            dropout_counts = df['Dropout_Likelihood'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                fig_pie = px.pie(values=dropout_counts.values, 
                               names=dropout_counts.index,
                               title='Dropout Distribution')
                st.plotly_chart(fig_pie)
            
            with col2:
                # Calculate dropout rate by different categories
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
        
        # Performance metrics by course
        if 'Course_Name' in df.columns and 'Quiz_Scores' in df.columns:
            st.subheader("📈 Performance by Course")
            
            performance_metrics = df.groupby('Course_Name').agg({
                'Quiz_Scores': 'mean',
                'Final_Exam_Score': 'mean',
                'Assignment_Completion_Rate': 'mean',
                'Time_Spent_on_Videos': 'mean'
            }).round(1)
            
            st.dataframe(performance_metrics.style.background_gradient(cmap='Blues'))
            
            # Create comparison chart
            fig = go.Figure()
            courses = performance_metrics.index
            
            fig.add_trace(go.Bar(name='Avg Quiz Scores', x=courses, y=performance_metrics['Quiz_Scores']))
            fig.add_trace(go.Bar(name='Avg Exam Scores', x=courses, y=performance_metrics['Final_Exam_Score']))
            fig.add_trace(go.Bar(name='Avg Assignment Completion', x=courses, y=performance_metrics['Assignment_Completion_Rate']))
            
            fig.update_layout(barmode='group', title='Average Performance Metrics by Course')
            st.plotly_chart(fig)
        
        # Learning style analysis
        if 'Learning_Style' in df.columns:
            st.subheader("🎯 Learning Style Analysis")
            
            learning_style_dist = df['Learning_Style'].value_counts()
            fig_learning = px.bar(x=learning_style_dist.index, y=learning_style_dist.values,
                                title='Distribution of Learning Styles',
                                labels={'x': 'Learning Style', 'y': 'Count'})
            st.plotly_chart(fig_learning)
            
            # Performance by learning style
            if 'Quiz_Scores' in df.columns:
                learning_performance = df.groupby('Learning_Style').agg({
                    'Quiz_Scores': 'mean',
                    'Final_Exam_Score': 'mean'
                }).round(1)
                
                st.write("**Average Performance by Learning Style:**")
                st.dataframe(learning_performance.style.background_gradient(cmap='Greens'))
        
        # Engagement analysis
        if 'Engagement_Level' in df.columns:
            st.subheader("🔥 Engagement Analysis")
            
            engagement_dist = df['Engagement_Level'].value_counts()
            fig_engagement = px.pie(values=engagement_dist.values, 
                                  names=engagement_dist.index,
                                  title='Student Engagement Levels')
            st.plotly_chart(fig_engagement)
            
            # Engagement vs Performance
            if 'Quiz_Scores' in df.columns:
                engagement_performance = df.groupby('Engagement_Level').agg({
                    'Quiz_Scores': 'mean',
                    'Final_Exam_Score': 'mean',
                    'Time_Spent_on_Videos': 'mean'
                }).round(1)
                
                st.write("**Performance by Engagement Level:**")
                st.dataframe(engagement_performance.style.background_gradient(cmap='Reds'))
                
    except Exception as e:
        st.error(f"Error in student analytics: {str(e)}")
