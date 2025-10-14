import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def perform_clustering(df):
    """Perform KMeans clustering on student data with 2 clusters"""
    st.header("👥 Student Clustering Analysis")
    
    try:
        # Exclude target variable and non-numeric columns for clustering
        features_to_drop = ['Dropout_Likelihood', 'Dropout_encoded', 'Student_ID']
        features_to_drop = [col for col in features_to_drop if col in df.columns]
        
        # Remove any existing cluster columns
        if 'Cluster' in df.columns:
            features_to_drop.append('Cluster')
        if 'Learning_Style' in df.columns:
            features_to_drop.append('Learning_Style')
            
        features_df = df.drop(columns=features_to_drop)
        numeric_df = features_df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            st.error("No numeric features available for clustering.")
            return None, None
        
        st.subheader("📊 Features Used for Clustering")
        st.write(f"**Number of features:** {len(numeric_df.columns)}")
        st.write(f"**Features:** {list(numeric_df.columns)}")
        
        # Display original feature statistics
        st.write("**Original Feature Statistics:**")
        original_stats = numeric_df.agg(['mean', 'std', 'min', 'max']).round(2)
        st.dataframe(original_stats)
        
        # Improved preprocessing for better clustering
        st.subheader("🔧 Advanced Preprocessing for Better Clustering")
        
        # 1. Handle outliers using robust scaling
        robust_scaler = RobustScaler()
        robust_features = robust_scaler.fit_transform(numeric_df)
        
        # 2. Apply power transformation for skewed data
        power_transformer = PowerTransformer(method='yeo-johnson')
        transformed_features = power_transformer.fit_transform(robust_features)
        
        # 3. Final standardization
        final_scaler = StandardScaler()
        scaled_features = final_scaler.fit_transform(transformed_features)
        scaled_df = pd.DataFrame(scaled_features, columns=numeric_df.columns)
        
        st.info("✅ Applied: Robust Scaling → Power Transformation → Standardization")
        
        # Determine optimal number of clusters with improved method
        st.subheader("🔍 Finding Optimal Number of Clusters")
        
        range_n_clusters = [2, 3, 4, 5]
        wcss = []
        silhouette_scores = []
        
        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, algorithm='elkan')
            cluster_labels = kmeans.fit_predict(scaled_df)
            wcss.append(kmeans.inertia_)
            silhouette_avg = silhouette_score(scaled_df, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Find the best number of clusters (elbow + silhouette)
        best_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
        st.info(f"🎯 Recommended clusters based on silhouette score: {best_n_clusters}")
        
        # Plot WCSS (Elbow Method)
        fig_elbow = px.line(x=range_n_clusters, y=wcss, 
                           title='Elbow Method for Optimal Clusters',
                           labels={'x': 'Number of Clusters', 'y': 'Within-Cluster Sum of Squares'})
        fig_elbow.update_traces(mode='lines+markers')
        st.plotly_chart(fig_elbow)
        
        # Plot Silhouette Scores
        fig_silhouette = px.line(x=range_n_clusters, y=silhouette_scores,
                                title='Silhouette Scores for Different Cluster Numbers',
                                labels={'x': 'Number of Clusters', 'y': 'Silhouette Score'})
        fig_silhouette.update_traces(mode='lines+markers')
        st.plotly_chart(fig_silhouette)
        
        # Use 2 clusters as requested
        optimal_clusters = 2
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=50, 
                       algorithm='elkan', max_iter=500, tol=1e-6)
        cluster_labels = kmeans.fit_predict(scaled_df)
        
        # Calculate silhouette score for final clustering
        final_silhouette = silhouette_score(scaled_df, cluster_labels)
        
        st.success(f"✅ Clustering completed with {optimal_clusters} clusters")
        st.info(f"**Improved Silhouette Score:** {final_silhouette:.3f}")
        
        # Use t-SNE for better visualization (reduces overlap) - FIXED PARAMETERS
        st.subheader("🔄 Applying t-SNE for Better Visualization")
        try:
            # Try with updated parameter names
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, 
                       n_iter=1000, learning_rate=200)
        except TypeError:
            # Fallback for different scikit-learn versions
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=30, 
                           max_iter=1000, learning_rate=200)
            except TypeError:
                # Most compatible version
                tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        
        tsne_components = tsne.fit_transform(scaled_df)
        
        # Also keep PCA for comparison
        pca = PCA(n_components=2, random_state=42)
        pca_components = pca.fit_transform(scaled_df)
        
        # Map clusters to learning styles based on performance metrics
        st.subheader("🎯 Cluster Interpretation")
        
        # Create a copy for analysis with original values
        cluster_df = numeric_df.copy()
        cluster_df['Cluster'] = cluster_labels
        
        # Analyze cluster characteristics to assign learning styles
        cluster_profiles = {}
        cluster_stats = {}
        
        # Calculate overall averages for comparison
        overall_metrics = {}
        performance_cols = ['Quiz_Scores', 'Final_Exam_Score', 'Assignment_Completion_Rate']
        available_performance_cols = [col for col in performance_cols if col in cluster_df.columns]
        
        for col in available_performance_cols:
            overall_metrics[col] = cluster_df[col].mean()
        
        # Sort clusters by performance to assign labels correctly
        cluster_performance = []
        
        for cluster in range(optimal_clusters):
            cluster_data = cluster_df[cluster_df['Cluster'] == cluster]
            
            # Calculate average performance metrics
            performance_metrics = {}
            avg_performance = 0
            metric_count = 0
            
            for col in available_performance_cols:
                if col in cluster_data.columns:
                    performance_metrics[f'Avg_{col}'] = cluster_data[col].mean()
                    # Calculate normalized performance score
                    normalized_score = (cluster_data[col].mean() - cluster_df[col].min()) / (cluster_df[col].max() - cluster_df[col].min())
                    avg_performance += normalized_score
                    metric_count += 1
            
            if metric_count > 0:
                avg_performance /= metric_count
            
            cluster_stats[cluster] = performance_metrics
            cluster_performance.append((cluster, avg_performance))
        
        # Sort clusters by performance (highest to lowest)
        cluster_performance.sort(key=lambda x: x[1], reverse=True)
        
        # Assign learning styles based on performance ranking for 2 clusters
        learning_styles = ["Good or Above Average Learner", "Weak or Below Average Learner"]
        for idx, (cluster, performance) in enumerate(cluster_performance):
            cluster_profiles[cluster] = learning_styles[idx]
        
        # Apply learning styles to original dataframe
        df['Cluster'] = cluster_labels
        df['Learning_Style'] = df['Cluster'].map(cluster_profiles)
        
        # Display detailed cluster information
        st.subheader("📈 Cluster Distribution")
        cluster_counts = df['Learning_Style'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(values=cluster_counts.values, 
                           names=cluster_counts.index,
                           title='Learning Style Distribution',
                           color=cluster_counts.index,
                           color_discrete_map={
                               'Good or Above Average Learner': '#2E8B57',  # Green
                               'Weak or Below Average Learner': '#FF4500'   # Red
                           })
            st.plotly_chart(fig_pie)
        
        with col2:
            fig_bar = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                           title='Learning Style Counts',
                           labels={'x': 'Learning Style', 'y': 'Number of Students'},
                           color=cluster_counts.index,
                           color_discrete_map={
                               'Good or Above Average Learner': '#2E8B57',
                               'Weak or Below Average Learner': '#FF4500'
                           })
            st.plotly_chart(fig_bar)
        
        # Display cluster statistics
        st.subheader("📋 Detailed Cluster Statistics")
        
        # Create a comprehensive performance comparison
        performance_comparison = []
        for cluster, learning_style in cluster_profiles.items():
            cluster_data = cluster_df[cluster_df['Cluster'] == cluster]
            cluster_info = {
                'Learning Style': learning_style,
                'Cluster Label': cluster,
                'Student Count': len(cluster_data),
                'Percentage': (len(cluster_data) / len(df)) * 100
            }
            
            # Add performance metrics
            for col in available_performance_cols:
                if col in cluster_data.columns:
                    cluster_info[f'Avg {col}'] = cluster_data[col].mean()
                    cluster_info[f'Std {col}'] = cluster_data[col].std()
            
            performance_comparison.append(cluster_info)
        
        performance_df = pd.DataFrame(performance_comparison)
        st.dataframe(performance_df.style.format({
            'Percentage': '{:.1f}%',
            'Avg Quiz_Scores': '{:.1f}',
            'Avg Final_Exam_Score': '{:.1f}',
            'Avg Assignment_Completion_Rate': '{:.1f}'
        }).background_gradient(cmap='Blues'))
        
        # Show performance comparison
        st.subheader("🎯 Performance Comparison")
        
        if available_performance_cols:
            comparison_data = []
            for learning_style in learning_styles:
                cluster_data = df[df['Learning_Style'] == learning_style]
                row = {'Learning Style': learning_style}
                for col in available_performance_cols:
                    row[col] = cluster_data[col].mean()
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Create performance comparison chart
            fig_comparison = go.Figure()
            
            colors = ['#2E8B57', '#FF4500']  # Green, Red
            
            for i, metric in enumerate(available_performance_cols):
                fig_comparison.add_trace(go.Bar(
                    name=metric,
                    x=comparison_df['Learning Style'],
                    y=comparison_df[metric],
                    marker_color=colors[i % len(colors)],
                    opacity=0.8
                ))
            
            fig_comparison.update_layout(
                title='Performance Metrics Comparison',
                xaxis_title='Learning Style',
                yaxis_title='Average Score',
                barmode='group',
                showlegend=True
            )
            
            st.plotly_chart(fig_comparison)
        
        return cluster_labels, tsne_components  # Return t-SNE components for better visualization
        
    except Exception as e:
        st.error(f"Error in clustering: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None, None

def plot_clustering_results(df, cluster_labels, tsne_result):
    """Visualize clustering results with reduced overlap"""
    st.subheader("📈 Cluster Visualization (t-SNE - Reduced Overlap)")
    
    try:
        if tsne_result is None or cluster_labels is None:
            st.warning("No clustering results to visualize")
            return
        
        # Ensure cluster labels exist in dataframe
        if 'Cluster' not in df.columns or 'Learning_Style' not in df.columns:
            st.error("Cluster information not found in dataframe")
            return
        
        # Create visualization DataFrame with t-SNE
        viz_df = pd.DataFrame({
            'X': tsne_result[:, 0],
            'Y': tsne_result[:, 1],
            'Cluster': df['Cluster'],
            'Learning_Style': df['Learning_Style'],
            'Dropout_Risk': df['Dropout_Likelihood'] if 'Dropout_Likelihood' in df.columns else 'Unknown'
        })
        
        # Add jitter to reduce overlap
        jitter_strength = 0.1  # Increased jitter for better separation
        np.random.seed(42)  # For reproducible jitter
        viz_df['X_jittered'] = viz_df['X'] + np.random.normal(0, jitter_strength, len(viz_df))
        viz_df['Y_jittered'] = viz_df['Y'] + np.random.normal(0, jitter_strength, len(viz_df))
        
        # Create interactive scatter plot with reduced overlap
        fig = px.scatter(viz_df, x='X_jittered', y='Y_jittered', color='Learning_Style',
                        title='Student Clusters (t-SNE Visualization with Reduced Overlap)',
                        hover_data=['Cluster', 'Learning_Style', 'Dropout_Risk'],
                        labels={'X_jittered': 't-SNE Component 1',
                               'Y_jittered': 't-SNE Component 2',
                               'Learning_Style': 'Learning Style'},
                        color_discrete_map={
                            'Good or Above Average Learner': '#2E8B57',  # Green
                            'Weak or Below Average Learner': '#FF4500'   # Red
                        },
                        size_max=6,  # Smaller points
                        opacity=0.7)  # Some transparency
        
        # Improve layout to reduce overlap
        fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color='DarkSlateGrey')),
                         selector=dict(mode='markers'))
        
        fig.update_layout(
            width=800,
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig)
        
        # Also show PCA for comparison
        st.subheader("📊 Alternative View (PCA Visualization)")
        
        # Perform PCA for comparison
        numeric_df = df.select_dtypes(include=[np.number])
        features_to_exclude = ['Dropout_encoded', 'Cluster']
        numeric_df = numeric_df.drop(columns=[col for col in features_to_exclude if col in numeric_df.columns])
        
        if not numeric_df.empty:
            pca = PCA(n_components=2, random_state=42)
            pca_components = pca.fit_transform(numeric_df)
            
            pca_df = pd.DataFrame({
                'PC1': pca_components[:, 0],
                'PC2': pca_components[:, 1],
                'Learning_Style': df['Learning_Style']
            })
            
            # Add jitter to PCA as well
            pca_df['PC1_jittered'] = pca_df['PC1'] + np.random.normal(0, 0.05, len(pca_df))
            pca_df['PC2_jittered'] = pca_df['PC2'] + np.random.normal(0, 0.05, len(pca_df))
            
            fig_pca = px.scatter(pca_df, x='PC1_jittered', y='PC2_jittered', color='Learning_Style',
                               title='Student Clusters (PCA Visualization)',
                               color_discrete_map={
                                   'Good or Above Average Learner': '#2E8B57',
                                   'Weak or Below Average Learner': '#FF4500'
                               },
                               opacity=0.7)
            
            fig_pca.update_traces(marker=dict(size=5))
            st.plotly_chart(fig_pca)
        
        # Display cluster characteristics
        st.subheader("📊 Cluster Characteristics Summary")
        
        # Select relevant numeric columns for summary
        summary_columns = ['Age', 'Time_Spent_on_Videos', 'Quiz_Attempts', 'Quiz_Scores', 
                          'Forum_Participation', 'Assignment_Completion_Rate', 'Final_Exam_Score', 
                          'Feedback_Score']
        summary_columns = [col for col in summary_columns if col in df.columns]
        
        if summary_columns:
            cluster_summary = df.groupby('Learning_Style')[summary_columns].mean().round(2)
            
            # Display with styling
            styled_summary = cluster_summary.style.background_gradient(cmap='Blues', axis=0)
            st.dataframe(styled_summary)
            
            # Create a bar chart for performance comparison
            st.subheader("📈 Performance Comparison - Bar Chart")
            create_performance_barchart(cluster_summary)
        
        # Dropout analysis by cluster
        if 'Dropout_Likelihood' in df.columns:
            st.subheader("🎯 Dropout Analysis by Learning Style")
            
            dropout_analysis = df.groupby('Learning_Style').agg({
                'Dropout_Likelihood': lambda x: (x == 'Yes').mean() * 100,
                'Cluster': 'count'
            }).round(2)
            dropout_analysis.columns = ['Dropout_Rate (%)', 'Student_Count']
            
            st.dataframe(dropout_analysis.style.background_gradient(cmap='Reds', subset=['Dropout_Rate (%)']))
            
            # Visualize dropout rates
            fig_dropout = px.bar(dropout_analysis.reset_index(), 
                               x='Learning_Style', y='Dropout_Rate (%)',
                               title='Dropout Rates by Learning Style',
                               color='Learning_Style',
                               color_discrete_map={
                                   'Good or Above Average Learner': '#2E8B57',
                                   'Weak or Below Average Learner': '#FF4500'
                               })
            st.plotly_chart(fig_dropout)
        
        # Show sample students from each cluster
        st.subheader("👥 Sample Students from Each Cluster")
        for learning_style in ['Good or Above Average Learner', 'Weak or Below Average Learner']:
            if learning_style in df['Learning_Style'].unique():
                with st.expander(f"🎓 {learning_style} Students"):
                    cluster_students = df[df['Learning_Style'] == learning_style].head(5)
                    display_columns = ['Age', 'Course_Name', 'Quiz_Scores', 'Final_Exam_Score', 
                                     'Assignment_Completion_Rate', 'Learning_Style']
                    display_columns = [col for col in display_columns if col in cluster_students.columns]
                    st.dataframe(cluster_students[display_columns])
        
    except Exception as e:
        st.error(f"Error visualizing clusters: {str(e)}")

def create_performance_barchart(cluster_summary):
    """Create a bar chart for performance comparison"""
    try:
        # Select key performance metrics
        performance_metrics = ['Quiz_Scores', 'Final_Exam_Score', 'Assignment_Completion_Rate']
        available_metrics = [metric for metric in performance_metrics if metric in cluster_summary.columns]
        
        if len(available_metrics) == 0:
            st.warning("No performance metrics available for comparison")
            return
        
        # Create grouped bar chart
        fig = go.Figure()
        
        colors = ['#2E8B57', '#FF4500']  # Green, Red
        
        for i, learning_style in enumerate(cluster_summary.index):
            values = [cluster_summary.loc[learning_style, metric] for metric in available_metrics]
            fig.add_trace(go.Bar(
                name=learning_style,
                x=available_metrics,
                y=values,
                marker_color=colors[i % len(colors)],
                opacity=0.8
            ))
        
        fig.update_layout(
            title='Performance Metrics by Learning Style',
            xaxis_title='Metrics',
            yaxis_title='Scores',
            barmode='group',
            showlegend=True,
            width=600,
            height=500
        )
        
        st.plotly_chart(fig)
        
    except Exception as e:
        st.warning(f"Could not create performance chart: {str(e)}")

def print_clustering_details(df, cluster_labels):
    """Print detailed clustering information"""
    st.subheader("🔍 Clustering Technical Details")
    
    try:
        # Check if cluster columns exist
        if 'Cluster' not in df.columns or 'Learning_Style' not in df.columns:
            st.error("Cluster information not available in dataframe")
            return
        
        # Cluster sizes and proportions
        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
        st.write("**Cluster Sizes:**")
        for cluster, size in cluster_sizes.items():
            percentage = (size / len(df)) * 100
            learning_style = df[df['Cluster'] == cluster]['Learning_Style'].iloc[0]
            st.write(f"**{learning_style}** (Cluster {cluster}): {size} students ({percentage:.1f}%)")
        
        # Learning style mapping
        st.write("**Learning Style Mapping:**")
        learning_mapping = df[['Cluster', 'Learning_Style']].drop_duplicates().sort_values('Cluster')
        st.dataframe(learning_mapping)
        
        # Performance comparison
        st.write("**Performance Comparison Across Clusters:**")
        performance_cols = ['Quiz_Scores', 'Final_Exam_Score', 'Assignment_Completion_Rate']
        available_performance_cols = [col for col in performance_cols if col in df.columns]
        
        if available_performance_cols:
            performance_comparison = df.groupby('Learning_Style')[available_performance_cols].mean().round(2)
            st.dataframe(performance_comparison.style.background_gradient(cmap='RdYlGn'))
            
            # Calculate performance difference
            if len(performance_comparison) == 2:
                good_learner = performance_comparison.iloc[0]
                weak_learner = performance_comparison.iloc[1]
                
                st.write("**Performance Gap Analysis:**")
                for col in available_performance_cols:
                    difference = good_learner[col] - weak_learner[col]
                    percentage_diff = (difference / weak_learner[col]) * 100
                    st.write(f"- **{col}:** {difference:.1f} points difference ({percentage_diff:.1f}% higher)")
        
        # Cluster separation quality
        st.write("**Cluster Quality Metrics:**")
        numeric_df = df.select_dtypes(include=[np.number])
        features_to_exclude = ['Dropout_encoded', 'Cluster']
        numeric_df = numeric_df.drop(columns=[col for col in features_to_exclude if col in numeric_df.columns])
        
        if not numeric_df.empty and len(numeric_df.columns) > 0:
            try:
                ch_score = calinski_harabasz_score(numeric_df, cluster_labels)
                db_score = davies_bouldin_score(numeric_df, cluster_labels)
                
                st.write(f"- **Calinski-Harabasz Score:** {ch_score:.3f} (Higher is better)")
                st.write(f"- **Davies-Bouldin Score:** {db_score:.3f} (Lower is better)")
                
                # Interpretation
                if ch_score > 300:
                    st.success("✅ Excellent cluster separation")
                elif ch_score > 200:
                    st.info("ℹ️ Good cluster separation")
                else:
                    st.warning("⚠️ Moderate cluster separation")
                    
            except Exception as e:
                st.write(f"- Could not calculate quality metrics: {e}")
        
    except Exception as e:
        st.error(f"Error printing clustering details: {str(e)}")
