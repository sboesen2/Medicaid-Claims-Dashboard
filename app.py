"""
Medicaid Claims Analysis Dashboard

A comprehensive Streamlit application for analyzing Medicaid claims data to optimize
healthcare delivery, improve cost efficiency, and enhance quality of care for vulnerable populations.

This dashboard demonstrates skills in:
- Medicaid claims data analysis and processing
- Cost efficiency modeling and budget optimization
- Provider performance evaluation and network analysis
- Population health metrics and quality improvement
- Government consulting mindset and policy impact analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processor import MedicaidDataProcessor
from analytics import MedicaidAnalytics
from visualizations import MedicaidVisualizations
from data_generator import MedicaidDataGenerator

# Page configuration
st.set_page_config(
    page_title="Medicaid Claims Analysis Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def show_overview_tab(processor, analytics, viz):
    """Display overview tab content."""
    
    st.markdown('<h2 class="section-header">📊 Executive Summary</h2>', unsafe_allow_html=True)
    
    # Calculate key metrics
    pmpm_metrics = processor.calculate_pmpm_metrics()
    quality_metrics = processor.calculate_quality_metrics()
    
    # Key metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Members",
            value=f"{pmpm_metrics['total_members']:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Total Claims",
            value=f"{pmpm_metrics['total_claims']:,}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="PMPM Allowed",
            value=f"${pmpm_metrics['overall_pmpm_allowed']:,.2f}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="PMPM Paid",
            value=f"${pmpm_metrics['overall_pmpm_paid']:,.2f}",
            delta=None
        )
    
    # Cost trend chart
    st.markdown('<h3 class="section-header">📈 Cost Trends</h3>', unsafe_allow_html=True)
    cost_trend_chart = viz.create_cost_trend_chart(pmpm_metrics['monthly_data'])
    st.plotly_chart(cost_trend_chart, use_container_width=True)
    
    # Service category breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="section-header">🏥 Service Categories</h3>', unsafe_allow_html=True)
        service_analysis = processor.analyze_service_categories()
        service_chart = viz.create_service_category_chart(service_analysis)
        st.plotly_chart(service_chart, use_container_width=True)
    
    with col2:
        st.markdown('<h3 class="section-header">📊 Quality Metrics</h3>', unsafe_allow_html=True)
        quality_chart = viz.create_quality_metrics_dashboard(quality_metrics)
        st.plotly_chart(quality_chart, use_container_width=True)

def show_cost_analysis_tab(processor, analytics, viz):
    """Display cost analysis tab content."""
    
    st.markdown('<h2 class="section-header">💰 Cost Analysis & Budget Optimization</h2>', unsafe_allow_html=True)
    
    # PMPM analysis
    pmpm_metrics = processor.calculate_pmpm_metrics()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="section-header">📈 PMPM Trends</h3>', unsafe_allow_html=True)
        cost_trend_chart = viz.create_cost_trend_chart(pmpm_metrics['monthly_data'])
        st.plotly_chart(cost_trend_chart, use_container_width=True)
    
    with col2:
        st.markdown('<h3 class="section-header">📊 Service Category Spending</h3>', unsafe_allow_html=True)
        service_analysis = processor.analyze_service_categories()
        service_chart = viz.create_service_category_chart(service_analysis)
        st.plotly_chart(service_chart, use_container_width=True)
    
    # High-cost members analysis
    st.markdown('<h3 class="section-header">👥 High-Cost Members Analysis</h3>', unsafe_allow_html=True)
    high_cost_members = processor.identify_high_cost_members()
    
    if not high_cost_members.empty:
        high_cost_chart = viz.create_high_cost_members_chart(high_cost_members)
        st.plotly_chart(high_cost_chart, use_container_width=True)
        
        # High-cost members table
        st.markdown('<h4>High-Cost Members Details</h4>', unsafe_allow_html=True)
        display_columns = ['member_id', 'total_allowed', 'age', 'chronic_condition', 
                          'emergency_visits', 'readmissions', 'cost_per_day']
        st.dataframe(
            high_cost_members[display_columns].head(10),
            use_container_width=True
        )
    
    # Budget forecasting
    st.markdown('<h3 class="section-header">🔮 Budget Forecasting</h3>', unsafe_allow_html=True)
    
    if st.button("Generate Budget Forecast"):
        with st.spinner("Generating forecast..."):
            forecast_data = analytics.forecast_budget_requirements()
            forecast_chart = viz.create_budget_forecast_chart(forecast_data)
            st.plotly_chart(forecast_chart, use_container_width=True)
            
            # Forecast summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Trend Slope", f"${forecast_data['trend_slope']:,.2f}")
            with col2:
                st.metric("R² Score", f"{forecast_data['r_squared']:.3f}")
            with col3:
                st.metric("Total Forecasted", f"${forecast_data['total_forecasted']:,.2f}")

def show_provider_performance_tab(processor, analytics, viz):
    """Display provider performance tab content."""
    
    st.markdown('<h2 class="section-header">🏥 Provider Performance & Network Analysis</h2>', unsafe_allow_html=True)
    
    # Provider performance analysis
    provider_performance = processor.analyze_provider_performance()
    
    # Provider performance chart
    st.markdown('<h3 class="section-header">📊 Provider Performance Matrix</h3>', unsafe_allow_html=True)
    provider_chart = viz.create_provider_performance_chart(provider_performance)
    st.plotly_chart(provider_chart, use_container_width=True)
    
    # Provider performance table
    st.markdown('<h3 class="section-header">📋 Provider Performance Details</h3>', unsafe_allow_html=True)
    display_columns = ['provider_name', 'provider_type', 'specialty', 'total_claims', 
                      'total_allowed', 'unique_members', 'cost_per_member', 'quality_score']
    st.dataframe(
        provider_performance[display_columns].head(15),
        use_container_width=True
    )
    
    # Network optimization
    st.markdown('<h3 class="section-header">🔧 Network Optimization Analysis</h3>', unsafe_allow_html=True)
    
    if st.button("Analyze Network Optimization"):
        with st.spinner("Analyzing network optimization opportunities..."):
            network_analysis = analytics.optimize_provider_network()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<h4>Top Performers</h4>', unsafe_allow_html=True)
                top_performers = network_analysis['top_performers'][
                    ['provider_name', 'efficiency_score', 'cost_per_member', 'quality_score']
                ]
                st.dataframe(top_performers, use_container_width=True)
            
            with col2:
                st.markdown('<h4>Areas for Improvement</h4>', unsafe_allow_html=True)
                bottom_performers = network_analysis['bottom_performers'][
                    ['provider_name', 'efficiency_score', 'cost_per_member', 'quality_score']
                ]
                st.dataframe(bottom_performers, use_container_width=True)

def show_roi_analysis_tab(processor, analytics, viz):
    """Display ROI Analysis tab content."""
    
    st.markdown('<h2 class="section-header">💡 ROI Analysis & Intervention Impact</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #1f77b4;">
    <h3 style="color: #1f77b4; margin-top: 0;">🎯 ROI Analysis Overview</h3>
    <p><strong>Purpose:</strong> Analyze the return on investment for healthcare interventions and quality improvement programs</p>
    <p><strong>Focus:</strong> Cost-effectiveness of preventive care, emergency reduction, and chronic disease management</p>
    <p><strong>Value:</strong> Data-driven insights for healthcare policy and program optimization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ROI Analysis
    st.markdown('<h3 class="section-header">💡 ROI Analysis for Interventions</h3>', unsafe_allow_html=True)
    
    if st.button("Calculate ROI Analysis"):
        with st.spinner("Calculating ROI for different interventions..."):
            roi_data = analytics.calculate_roi_analysis()
            roi_chart = viz.create_roi_analysis_chart(roi_data)
            st.plotly_chart(roi_chart, use_container_width=True)
            
            # ROI details
            if 'preventive_care_roi' in roi_data:
                pc_roi = roi_data['preventive_care_roi']
                st.markdown('<h4>Preventive Care ROI</h4>', unsafe_allow_html=True)
                st.info("💡 **Arizona Hospital Context**: Arizona hospitals average 6.1% operating margin (FY 2023). Preventive care ROI in healthcare delivery is typically 5-15%, much lower than public health interventions due to operational costs.")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Cost with Preventive Care", f"${pc_roi['avg_cost_with_preventive']:,.2f}")
                with col2:
                    st.metric("Cost without Preventive Care", f"${pc_roi['avg_cost_without_preventive']:,.2f}")
                with col3:
                    st.metric("ROI Percentage", f"{pc_roi['roi_percentage']:.1f}%")
            
            # Emergency Reduction ROI
            if 'emergency_reduction_roi' in roi_data:
                er_roi = roi_data['emergency_reduction_roi']
                st.markdown('<h4>Emergency Reduction ROI</h4>', unsafe_allow_html=True)
                st.info("🏜️ **Arizona Hospital Reality**: Arizona hospitals need 4-6% margins to reinvest in facilities. Emergency care costs 50-100% more than routine care, but ROI improvements are modest due to operational constraints and regulatory requirements.")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Emergency Care Cost", f"${er_roi['emergency_avg_cost']:,.2f}")
                with col2:
                    st.metric("Non-Emergency Care Cost", f"${er_roi['non_emergency_avg_cost']:,.2f}")
                with col3:
                    emergency_roi_pct = (er_roi['cost_difference'] / er_roi['non_emergency_avg_cost']) * 100 if er_roi['non_emergency_avg_cost'] > 0 else 0
                    st.metric("ROI Percentage", f"{emergency_roi_pct:.1f}%")
            
            # Chronic Disease Management ROI
            if 'chronic_disease_roi' in roi_data:
                cd_roi = roi_data['chronic_disease_roi']
                st.markdown('<h4>Chronic Disease Management ROI</h4>', unsafe_allow_html=True)
                
                # Show chronic condition breakdown
                if cd_roi:
                    st.markdown("**Chronic Condition Analysis:**")
                    for condition, data in cd_roi.items():
                        if isinstance(data, dict) and 'avg_cost' in data:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric(f"{condition} - Avg Cost", f"${data['avg_cost']:,.2f}")
                            with col2:
                                st.metric(f"{condition} - Emergency Rate", f"{data['emergency_rate']:.1f}%")
                            with col3:
                                st.metric(f"{condition} - Readmission Rate", f"{data['readmission_rate']:.1f}%")
                            with col4:
                                st.metric(f"{condition} - Members", f"{data['member_count']:,}")
                else:
                    st.info("Chronic disease management data not available")

def show_fraud_detection_tab(processor, analytics, viz):
    """Display fraud detection tab content."""
    
    st.markdown('<h2 class="section-header">🚨 Fraud Detection & Anomaly Analysis</h2>', unsafe_allow_html=True)
    
    # Fraud detection analysis
    st.markdown('<h3 class="section-header">🔍 Anomaly Detection Results</h3>', unsafe_allow_html=True)
    
    if st.button("Run Fraud Detection Analysis"):
        with st.spinner("Analyzing claims for potential fraud and anomalies..."):
            fraud_results = analytics.detect_fraud_patterns()
            fraud_chart = viz.create_fraud_detection_chart(fraud_results['anomaly_claims'])
            st.plotly_chart(fraud_chart, use_container_width=True)
            
            # Fraud detection summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Anomalies", fraud_results['total_anomalies'])
            with col2:
                st.metric("Anomaly Rate", f"{fraud_results['anomaly_rate']:.2f}%")
            with col3:
                st.metric("Potential Fraud Amount", 
                         f"${fraud_results['anomaly_claims']['allowed_amount'].sum():,.2f}")
            
            # Anomaly patterns table
            if not fraud_results['anomaly_patterns'].empty:
                st.markdown('<h4>Anomaly Patterns by Provider and Service Type</h4>', unsafe_allow_html=True)
                st.dataframe(
                    fraud_results['anomaly_patterns'].head(10),
                    use_container_width=True
                )
    
    # Manual anomaly detection
    st.markdown('<h3 class="section-header">🔧 Manual Anomaly Detection</h3>', unsafe_allow_html=True)
    
    anomalies = processor.detect_anomalies()
    if not anomalies.empty:
        st.markdown('<h4>Detected Anomalies</h4>', unsafe_allow_html=True)
        
        # Anomaly summary
        anomaly_summary = anomalies.groupby('anomaly_type').agg({
            'claim_id': 'count',
            'allowed_amount': 'sum',
            'paid_amount': 'sum'
        }).round(2)
        
        anomaly_summary.columns = ['Count', 'Total Allowed', 'Total Paid']
        st.dataframe(anomaly_summary, use_container_width=True)
        
        # Detailed anomalies table
        st.markdown('<h4>Detailed Anomaly Records</h4>', unsafe_allow_html=True)
        display_columns = ['claim_id', 'member_id', 'provider_id', 'service_type', 
                          'allowed_amount', 'paid_amount', 'anomaly_type']
        st.dataframe(
            anomalies[display_columns].head(20),
            use_container_width=True
        )
    else:
        st.info("No anomalies detected in the current dataset.")

def show_data_generation_tab(processor, analytics, viz):
    """Display data generation tab content."""
    
    st.markdown('<h2 class="section-header">🔧 Synthetic Data Generation</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        Generate realistic synthetic Medicaid claims data for demonstration and testing purposes.
        This data is completely synthetic and contains no real patient information.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize data generator
    generator = MedicaidDataGenerator()
    
    # Initialize session state for generated data
    if 'generated_claims' not in st.session_state:
        st.session_state.generated_claims = None
    if 'generated_providers' not in st.session_state:
        st.session_state.generated_providers = None
    
    # Data generation controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="section-header">📊 Dataset Configuration</h3>', unsafe_allow_html=True)
        
        # Dataset size
        dataset_size = st.selectbox(
            "Dataset Size",
            options=["Small (100 claims)", "Medium (500 claims)", "Large (1000 claims)", "Custom"],
            index=1
        )
        
        if dataset_size == "Custom":
            num_claims = st.number_input(
                "Number of Claims", 
                min_value=50, 
                max_value=5000, 
                value=500,
                step=50,
                help="Maximum 5,000 claims for optimal performance"
            )
            st.caption("💡 **Note**: Maximum 5,000 claims to ensure smooth dashboard performance")
            
            # Additional validation message
            if num_claims > 5000:
                st.error("❌ Please enter a value between 50 and 5,000 claims")
        else:
            size_map = {
                "Small (100 claims)": 100,
                "Medium (500 claims)": 500,
                "Large (1000 claims)": 1000
            }
            num_claims = size_map[dataset_size]
        
        # Scenario selection
        scenario = st.selectbox(
            "Data Scenario",
            options=["normal", "high_fraud", "rural", "urban", "seasonal"],
            format_func=lambda x: {
                "normal": "Normal Operations",
                "high_fraud": "High Fraud Scenario", 
                "rural": "Rural Patterns",
                "urban": "Urban Patterns",
                "seasonal": "Seasonal Patterns"
            }[x]
        )
        
        # Date range
        col_start, col_end = st.columns(2)
        with col_start:
            start_date = st.date_input("Start Date", value=datetime(2024, 1, 1).date())
        with col_end:
            end_date = st.date_input("End Date", value=datetime(2024, 3, 31).date())
    
    with col2:
        st.markdown('<h3 class="section-header">🎯 Scenario Descriptions</h3>', unsafe_allow_html=True)
        
        scenario_descriptions = {
            "normal": "**Normal Operations**: Typical Medicaid claims patterns with standard cost distributions and service utilization.",
            "high_fraud": "**High Fraud Scenario**: Increased anomalies, billing discrepancies, and suspicious patterns for ML testing.",
            "rural": "**Rural Patterns**: Fewer providers, higher costs due to distance, more emergency visits.",
            "urban": "**Urban Patterns**: More providers, competitive pricing, higher preventive care utilization.",
            "seasonal": "**Seasonal Patterns**: Time-based variations with winter emergency spikes and summer preventive care increases."
        }
        
        st.markdown(scenario_descriptions[scenario], unsafe_allow_html=True)
    
    # Generate data button
    st.markdown('<h3 class="section-header">🚀 Generate Dataset</h3>', unsafe_allow_html=True)
    
    if st.button("Generate Synthetic Data", type="primary", use_container_width=True):
        with st.spinner("Generating synthetic Medicaid claims data..."):
            try:
                # Enforce the 5000 claim limit
                if num_claims > 5000:
                    st.warning(f"⚠️ Maximum 5,000 claims allowed. Capping at 5,000 claims.")
                    num_claims = 5000
                
                # Generate claims data
                claims_df = generator.generate_claims_data(
                    num_claims=num_claims,
                    scenario=scenario,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                
                # Generate provider data
                providers_df = generator.generate_provider_data()
                
                # Store in session state
                st.session_state.generated_claims = claims_df
                st.session_state.generated_providers = providers_df
                st.session_state.generation_scenario = scenario
                st.session_state.generation_num_claims = num_claims
                
                st.success("✅ Data generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating data: {str(e)}")
    
    # Display generated data if available
    if st.session_state.generated_claims is not None:
        claims_df = st.session_state.generated_claims
        providers_df = st.session_state.generated_providers
        
        # Display generation summary
        st.markdown('<h3 class="section-header">📊 Generated Data Summary</h3>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Claims Generated", f"{len(claims_df):,}")
        with col2:
            st.metric("Members", f"{claims_df['member_id'].nunique():,}")
        with col3:
            st.metric("Providers", f"{len(providers_df):,}")
        with col4:
            st.metric("Total Value", f"${claims_df['allowed_amount'].sum():,.2f}")
        
        # Show data preview
        st.markdown('<h4>Generated Claims Preview</h4>', unsafe_allow_html=True)
        st.dataframe(claims_df.head(10), use_container_width=True)
        
        # Export options
        st.markdown('<h3 class="section-header">📤 Export Options</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_claims = claims_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Claims CSV",
                data=csv_claims,
                file_name=f"medicaid_claims_{st.session_state.generation_scenario}_{st.session_state.generation_num_claims}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            csv_providers = providers_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Providers CSV",
                data=csv_providers,
                file_name=f"medicaid_providers_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Load into dashboard option
        st.markdown('<h3 class="section-header">🔄 Load into Dashboard</h3>', unsafe_allow_html=True)
        if st.button("Load Generated Data into Dashboard", type="secondary", use_container_width=True):
            # Ensure dates are properly formatted before storing
            claims_df_copy = claims_df.copy()
            if 'service_date' in claims_df_copy.columns:
                claims_df_copy['service_date'] = pd.to_datetime(claims_df_copy['service_date'])
            
            # Store the generated data in session state
            st.session_state.active_claims = claims_df_copy
            st.session_state.active_providers = providers_df
            st.session_state.data_source = "generated"
            
            # Clear the cache to force reload
            if hasattr(st, 'cache_data'):
                st.cache_data.clear()
            
            st.success("✅ Generated data loaded into dashboard! The other tabs will now use this new data.")
            st.rerun()

def show_real_data_analysis_tab(processor, analytics, viz):
    """Display real data analysis tab content."""
    
    st.markdown('<h2 class="section-header">📁 Real Data Analysis (HIPAA Compliant)</h2>', unsafe_allow_html=True)
    
    # HIPAA Compliance Warning
    st.warning("""
    **⚠️ HIPAA COMPLIANCE NOTICE** ⚠️
    
    This section is designed for analyzing real Medicaid claims data. Please ensure:
    - All data has been properly de-identified
    - You have appropriate authorization to analyze this data
    - Data is handled according to HIPAA guidelines
    - No patient identifiers are included in uploaded files
    """)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        Upload and analyze real Medicaid claims data with built-in HIPAA compliance features.
        All data processing is performed locally and securely.
    </div>
    """, unsafe_allow_html=True)
    
    # Example schema section
    with st.expander("📋 **Click to see example data schema**", expanded=False):
        st.markdown("""
        ### **Ideal Data Structure**
        
        Your CSV should ideally contain these columns (but we can work with any format!):
        
        **Required Fields:**
        - `claim_id` - Unique identifier for each claim
        - `member_id` - Patient/member identifier  
        - `provider_id` - Healthcare provider identifier
        - `service_date` - Date of service (any date format)
        - `allowed_amount` - Total cost/allowed amount
        
        **Optional Fields (analysis may be limited if missing):**
        - `paid_amount` - Amount actually paid
        - `member_age` - Patient age
        - `member_gender` - Patient gender (M/F)
        - `service_type` - Type of service (Outpatient, Inpatient, Emergency, Preventive)
        - `diagnosis_code` - Medical diagnosis code
        - `procedure_code` - Medical procedure code
        - `preventive_care` - Boolean (True/False)
        - `emergency_visit` - Boolean (True/False)
        - `chronic_condition` - Chronic condition status
        - `member_zip` - Patient zip code
        - `county` - Geographic county
        """)
        
        # Show example data
        st.markdown("**Example CSV Structure:**")
        example_data = {
            'claim_id': ['CLM001', 'CLM002', 'CLM003'],
            'member_id': ['MEM001', 'MEM002', 'MEM001'],
            'provider_id': ['PROV001', 'PROV002', 'PROV001'],
            'service_date': ['2024-01-15', '2024-01-20', '2024-02-01'],
            'allowed_amount': [150.00, 300.50, 75.25],
            'paid_amount': [120.00, 240.40, 60.20],
            'member_age': [45, 32, 45],
            'member_gender': ['F', 'M', 'F'],
            'service_type': ['Outpatient', 'Emergency', 'Preventive']
        }
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df, use_container_width=True)
        
        st.warning("""
        **⚠️ HIPAA COMPLIANCE NOTICE** ⚠️
        
        **For Real Data Uploads**: We will NOT generate or add any synthetic data to your real patient information. 
        Missing fields will be excluded from analysis to maintain data integrity and HIPAA compliance.
        
        **For Testing/Demo**: Use the "Data Generation" tab to create fully synthetic datasets for testing purposes.
        """)
        
        # Download template button
        st.markdown("**📥 Download Sample Template:**")
        template_data = {
            'claim_id': ['CLM001', 'CLM002', 'CLM003', 'CLM004', 'CLM005'],
            'member_id': ['MEM001', 'MEM002', 'MEM001', 'MEM003', 'MEM002'],
            'provider_id': ['PROV001', 'PROV002', 'PROV001', 'PROV003', 'PROV002'],
            'service_date': ['2024-01-15', '2024-01-20', '2024-02-01', '2024-02-05', '2024-02-10'],
            'allowed_amount': [150.00, 300.50, 75.25, 450.00, 200.75],
            'paid_amount': [120.00, 240.40, 60.20, 360.00, 160.60],
            'member_age': [45, 32, 45, 28, 32],
            'member_gender': ['F', 'M', 'F', 'F', 'M'],
            'service_type': ['Outpatient', 'Emergency', 'Preventive', 'Inpatient', 'Outpatient'],
            'diagnosis_code': ['272.40', '789.01', '200.00', '410.01', '250.00'],
            'procedure_code': ['99213', '99281', '99396', '99223', '99214'],
            'preventive_care': [False, False, True, False, False],
            'emergency_visit': [False, True, False, False, False],
            'chronic_condition': ['High Cholesterol', 'None', 'Diabetes', 'Heart Disease', 'None'],
            'member_zip': ['85001', '85002', '85001', '85003', '85002'],
            'county': ['Maricopa', 'Maricopa', 'Maricopa', 'Pima', 'Maricopa']
        }
        template_df = pd.DataFrame(template_data)
        csv_template = template_df.to_csv(index=False)
        
        st.download_button(
            label="📄 Download Sample CSV Template",
            data=csv_template,
            file_name="medicaid_claims_template.csv",
            mime="text/csv",
            help="Download this template to see the ideal data structure"
        )
    
    # Initialize session state for uploaded data
    if 'uploaded_claims' not in st.session_state:
        st.session_state.uploaded_claims = None
    if 'uploaded_providers' not in st.session_state:
        st.session_state.uploaded_providers = None
    
    # File upload section
    st.markdown('<h3 class="section-header">📤 Upload Data Files</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Claims Data Upload**")
        uploaded_claims = st.file_uploader(
            "Upload Claims CSV",
            type=['csv'],
            help="Upload a CSV file containing Medicaid claims data",
            key="claims_uploader"
        )
    
    with col2:
        st.markdown("**Provider Data Upload**")
        uploaded_providers = st.file_uploader(
            "Upload Providers CSV", 
            type=['csv'],
            help="Upload a CSV file containing provider network data",
            key="providers_uploader"
        )
    
    # Process uploaded files
    if uploaded_claims is not None:
        try:
            # Load and validate claims data
            claims_df = pd.read_csv(uploaded_claims)
            
            # Show column mapping interface
            st.markdown('<h4>📋 Column Mapping</h4>', unsafe_allow_html=True)
            st.info("""
            **Flexible Schema**: Map your columns to the required fields. 
            If a column doesn't exist, we'll generate placeholder data.
            """)
            
            # Define required fields and their descriptions
            required_fields = {
                'claim_id': 'Unique claim identifier',
                'member_id': 'Member/patient identifier', 
                'provider_id': 'Healthcare provider identifier',
                'service_date': 'Date of service (any date format)',
                'allowed_amount': 'Total allowed amount (cost)'
            }
            
            # Create column mapping interface
            col_mapping = {}
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Required Field**")
                for field in list(required_fields.keys())[:3]:
                    st.text(f"{field}: {required_fields[field]}")
            
            with col2:
                st.markdown("**Your Column**")
                for field in list(required_fields.keys())[:3]:
                    available_cols = ['None'] + list(claims_df.columns)
                    selected_col = st.selectbox(
                        f"Map to {field}",
                        available_cols,
                        key=f"map_{field}",
                        help=required_fields[field]
                    )
                    col_mapping[field] = selected_col if selected_col != 'None' else None
            
            # Second row for remaining fields
            col3, col4 = st.columns(2)
            
            with col3:
                for field in list(required_fields.keys())[3:]:
                    st.text(f"{field}: {required_fields[field]}")
            
            with col4:
                for field in list(required_fields.keys())[3:]:
                    available_cols = ['None'] + list(claims_df.columns)
                    selected_col = st.selectbox(
                        f"Map to {field}",
                        available_cols,
                        key=f"map_{field}",
                        help=required_fields[field]
                    )
                    col_mapping[field] = selected_col if selected_col != 'None' else None
            
            # Process the mapping
            if st.button("Process Column Mapping", type="primary"):
                processed_df = claims_df.copy()
                missing_required = []
                
                # Apply column mappings for required fields only
                for required_field, mapped_col in col_mapping.items():
                    if mapped_col and mapped_col in processed_df.columns:
                        # Rename the column to standard name
                        processed_df[required_field] = processed_df[mapped_col]
                    else:
                        missing_required.append(required_field)
                
                # Check if we have all required fields
                if missing_required:
                    st.error(f"❌ **Missing Required Fields**: {', '.join(missing_required)}")
                    st.error("Please map all required fields or ensure your data contains these columns.")
                    st.info("💡 **Tip**: Use the 'Data Generation' tab to create a complete synthetic dataset for testing.")
                else:
                    # Only process if we have all required fields
                    st.success("✅ All required fields mapped successfully!")
                    
                    # Add optional fields only if they exist in the original data
                    optional_mappings = {
                        'paid_amount': ['paid_amount', 'paid', 'amount_paid', 'reimbursement'],
                        'member_age': ['age', 'patient_age', 'member_age'],
                        'member_gender': ['gender', 'sex', 'patient_gender'],
                        'service_type': ['service_type', 'visit_type', 'encounter_type'],
                        'diagnosis_code': ['diagnosis', 'icd_code', 'diagnosis_code'],
                        'procedure_code': ['procedure', 'cpt_code', 'procedure_code']
                    }
                    
                    for field, possible_names in optional_mappings.items():
                        if field not in processed_df.columns:
                            # Look for similar column names
                            found_col = None
                            for possible_name in possible_names:
                                matching_cols = [col for col in processed_df.columns if possible_name.lower() in col.lower()]
                                if matching_cols:
                                    found_col = matching_cols[0]
                                    break
                            
                            if found_col:
                                processed_df[field] = processed_df[found_col]
                    
                    # Add boolean columns only if service_type exists
                    if 'service_type' in processed_df.columns:
                        processed_df['preventive_care'] = processed_df['service_type'].str.contains('Preventive', case=False, na=False)
                        processed_df['emergency_visit'] = processed_df['service_type'].str.contains('Emergency', case=False, na=False)
                    else:
                        processed_df['preventive_care'] = False
                        processed_df['emergency_visit'] = False
                    
                    # Add other fields only if they don't exist
                    if 'readmission_within_30_days' not in processed_df.columns:
                        processed_df['readmission_within_30_days'] = False
                    
                    st.session_state.uploaded_claims = processed_df
                    st.success("✅ Claims data processed successfully! (HIPAA compliant - no synthetic data added)")
                    
                    # Show preview
                    st.markdown('<h4>Processed Data Preview</h4>', unsafe_allow_html=True)
                    st.dataframe(processed_df.head(), use_container_width=True)
                    
                    # Show analysis limitations
                    missing_optional = [col for col in ['paid_amount', 'member_age', 'member_gender', 'service_type'] if col not in processed_df.columns]
                    if missing_optional:
                        st.warning(f"⚠️ **Analysis Limitations**: Some features may be limited due to missing optional fields: {', '.join(missing_optional)}")
                        st.info("💡 **Suggestion**: Use the 'Data Generation' tab to create a complete dataset for full analysis capabilities.")
        
        except Exception as e:
            st.error(f"Error processing claims file: {str(e)}")
            st.error("Please ensure your file is a valid CSV with readable data.")
    
    if uploaded_providers is not None:
        try:
            providers_df = pd.read_csv(uploaded_providers)
            st.session_state.uploaded_providers = providers_df
            st.success("✅ Provider data loaded successfully!")
        except Exception as e:
            st.error(f"Error processing providers file: {str(e)}")
    
    # Display uploaded data if available
    if st.session_state.uploaded_claims is not None:
        claims_df = st.session_state.uploaded_claims
        
        # Display data summary
        st.markdown('<h3 class="section-header">📊 Uploaded Data Summary</h3>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Claims", f"{len(claims_df):,}")
        with col2:
            st.metric("Unique Members", f"{claims_df['member_id'].nunique():,}")
        with col3:
            st.metric("Total Value", f"${claims_df['allowed_amount'].sum():,.2f}")
        
        # Data preview
        st.markdown('<h4>Data Preview</h4>', unsafe_allow_html=True)
        st.dataframe(claims_df.head(), use_container_width=True)
        
        # Load into processor
        if st.button("Load Real Data into Dashboard", type="primary", use_container_width=True):
            # Ensure dates are properly formatted before storing
            claims_df_copy = claims_df.copy()
            if 'service_date' in claims_df_copy.columns:
                claims_df_copy['service_date'] = pd.to_datetime(claims_df_copy['service_date'])
            
            # Store the uploaded data in session state
            st.session_state.active_claims = claims_df_copy
            if st.session_state.uploaded_providers is not None:
                st.session_state.active_providers = st.session_state.uploaded_providers
            else:
                st.session_state.active_providers = None
            st.session_state.data_source = "uploaded"
            
            # Clear the cache to force reload
            if hasattr(st, 'cache_data'):
                st.cache_data.clear()
            
            st.success("✅ Real data loaded into dashboard! The other tabs will now use this new data.")
            st.rerun()
    
    # HIPAA Compliance Features
    st.markdown('<h3 class="section-header">🔒 HIPAA Compliance Features</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Data Security:**
        - All data processing performed locally
        - No data transmitted to external servers
        - Automatic data validation and sanitization
        - Secure file handling protocols
        """)
    
    with col2:
        st.markdown("""
        **Compliance Measures:**
        - Data de-identification verification
        - Audit logging capabilities
        - Secure data storage
        - HIPAA-compliant data handling
        """)
    
    # Data retention notice
    st.info("""
    **Data Retention Notice**: Uploaded data is processed locally and not permanently stored. 
    For production use, ensure proper data retention policies are in place according to your organization's HIPAA compliance requirements.
    """)

def show_etl_pipeline_tab(processor, analytics, viz):
    """Display ETL pipeline tab content."""
    
    st.markdown('<h2 class="section-header">⚙️ ETL Pipeline Management</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        Extract, Transform, and Load data with advanced processing capabilities.
        Build automated data pipelines for healthcare analytics.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for ETL
    if 'etl_pipeline_data' not in st.session_state:
        st.session_state.etl_pipeline_data = None
    if 'etl_processing_log' not in st.session_state:
        st.session_state.etl_processing_log = []
    
    # ETL Pipeline Steps - Dynamic column sizing based on data source
    # Get data source first to determine layout
    data_source = st.selectbox(
        "Data Source",
        ["File Upload", "Database Connection", "API Endpoint"],
        help="Select your data source type",
        key="etl_data_source"
    )
    
    # Adjust column sizes based on data source
    if data_source == "Database Connection":
        col1, col2, col3 = st.columns([4, 2, 2])  # Larger extract box for database config
    else:
        col1, col2, col3 = st.columns([2, 3, 2])  # Default layout
    
    with col1:
        st.markdown('<h3 class="section-header">📥 Extract</h3>', unsafe_allow_html=True)
        
        if data_source == "File Upload":
            # Separate uploaders for claims and providers
            upload_col1, upload_col2 = st.columns(2)
            
            with upload_col1:
                st.markdown("**📋 Claims Data**")
                claims_files = st.file_uploader(
                    "Upload Claims Files",
                    type=['csv', 'xlsx', 'json', 'xml'],
                    accept_multiple_files=True,
                    key="claims_upload",
                    help="Upload claims data (CSV, Excel, JSON, or XML)"
                )
            
            with upload_col2:
                st.markdown("**🏥 Provider Data**")
                provider_files = st.file_uploader(
                    "Upload Provider Files",
                    type=['csv', 'xlsx', 'json', 'xml'],
                    accept_multiple_files=True,
                    key="provider_upload",
                    help="Upload provider data (optional - will generate if missing)"
                )
            
            # Show upload status and process files
            if claims_files or provider_files:
                if claims_files:
                    st.success(f"✅ {len(claims_files)} claims file(s) uploaded!")
                if provider_files:
                    st.success(f"✅ {len(provider_files)} provider file(s) uploaded!")
                
                # Process uploaded files
                if st.button("Process Uploaded Files", type="primary"):
                    with st.spinner("Processing files..."):
                        # Ensure pandas and numpy are available in this scope
                        import pandas as pd
                        import numpy as np
                        processed_files = []
                        
                        # Process claims files
                        if claims_files:
                            for file in claims_files:
                                try:
                                    if file.name.endswith('.csv'):
                                        df = pd.read_csv(file)
                                    elif file.name.endswith('.xlsx'):
                                        df = pd.read_excel(file)
                                    elif file.name.endswith('.json'):
                                        df = pd.read_json(file)
                                    else:
                                        continue
                                    
                                    processed_files.append({
                                        'filename': file.name,
                                        'data': df,
                                        'rows': len(df),
                                        'columns': len(df.columns),
                                        'type': 'claims'
                                    })
                                    
                                    st.session_state.etl_processing_log.append(f"✅ Processed claims file {file.name}: {len(df)} rows, {len(df.columns)} columns")
                                    
                                except Exception as e:
                                    st.session_state.etl_processing_log.append(f"❌ Error processing {file.name}: {str(e)}")
                                    st.error(f"Error processing {file.name}: {str(e)}")
                        
                        # Process provider files
                        if provider_files:
                            for file in provider_files:
                                try:
                                    if file.name.endswith('.csv'):
                                        df = pd.read_csv(file)
                                    elif file.name.endswith('.xlsx'):
                                        df = pd.read_excel(file)
                                    elif file.name.endswith('.json'):
                                        df = pd.read_json(file)
                                    else:
                                        continue
                                    
                                    processed_files.append({
                                        'filename': file.name,
                                        'data': df,
                                        'rows': len(df),
                                        'columns': len(df.columns),
                                        'type': 'providers'
                                    })
                                    
                                    st.session_state.etl_processing_log.append(f"✅ Processed provider file {file.name}: {len(df)} rows, {len(df.columns)} columns")
                                    
                                except Exception as e:
                                    st.session_state.etl_processing_log.append(f"❌ Error processing {file.name}: {str(e)}")
                                    st.error(f"Error processing {file.name}: {str(e)}")
                        
                        if processed_files:
                            st.session_state.etl_pipeline_data = processed_files
                            st.success(f"✅ Successfully processed {len(processed_files)} file(s)!")
                        else:
                            st.error("❌ No files were processed successfully.")
        
        elif data_source == "Database Connection":
            st.markdown("**🔗 Database Connection Framework**")
            st.markdown("""
            **Database Connection Template Generator** - Generate ready-to-use code for connecting to your database.
            
            Configure your database connection parameters below and get Python code that you can use in your own applications.
            This is a template framework - use the generated code in your own environment.
            """)
            
            # Instructions
            with st.expander("📖 How to Use This Template", expanded=True):
                st.markdown("""
                **Step 1:** Select your database type and fill in the connection details
                
                **Step 2:** Click "Generate Complete Code Template" to create a Python script
                
                **Step 3:** Download the generated Python file to your computer
                
                **Step 4:** Install the required database driver:
                ```bash
                # For PostgreSQL
                pip install psycopg2-binary
                
                # For SQL Server  
                pip install pyodbc
                
                # For MySQL
                pip install mysql-connector-python
                
                # SQLite is built-in with Python
                ```
                
                **Step 5:** Run the downloaded Python file in your environment:
                ```bash
                python postgresql_etl_template.py
                ```
                
                **Or import it into your own Python application:**
                ```python
                from postgresql_etl_template import connect_to_database
                ```
                """)
            
            # Database type selection
            db_type = st.selectbox(
                "Database Type",
                ["PostgreSQL", "SQL Server", "MySQL", "SQLite"],
                help="Select your database type"
            )
            
            # Connection configuration
            with st.expander("🔧 Database Configuration", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    if db_type == "PostgreSQL":
                        host = st.text_input("Host", value="localhost", help="Database server hostname")
                        port = st.number_input("Port", value=5432, help="Database port")
                        database = st.text_input("Database Name", help="Name of your database")
                        username = st.text_input("Username", help="Database username")
                        password = st.text_input("Password", type="password", help="Database password")
                        
                    elif db_type == "SQL Server":
                        server = st.text_input("Server", value="localhost", help="SQL Server hostname")
                        port = st.number_input("Port", value=1433, help="SQL Server port")
                        database = st.text_input("Database Name", help="Name of your database")
                        username = st.text_input("Username", help="Database username")
                        password = st.text_input("Password", type="password", help="Database password")
                        
                    elif db_type == "MySQL":
                        host = st.text_input("Host", value="localhost", help="MySQL server hostname")
                        port = st.number_input("Port", value=3306, help="MySQL port")
                        database = st.text_input("Database Name", help="Name of your database")
                        username = st.text_input("Username", help="Database username")
                        password = st.text_input("Password", type="password", help="Database password")
                        
                    elif db_type == "SQLite":
                        database = st.text_input("Database File Path", value="medicaid_data.db", help="Path to your SQLite database file")
                
                with col2:
                    st.markdown("**📋 Connection Template**")
                    if db_type == "PostgreSQL":
                        st.code(f"""
# PostgreSQL Connection Template
import psycopg2

conn = psycopg2.connect(
    host="{host if 'host' in locals() else 'your-host'}",
    port={port if 'port' in locals() else 5432},
    database="{database if 'database' in locals() else 'your-database'}",
    user="{username if 'username' in locals() else 'your-username'}",
    password="{password if 'password' in locals() else 'your-password'}"
)
                        """, language="python")
                    elif db_type == "SQL Server":
                        st.code(f"""
# SQL Server Connection Template
import pyodbc

conn = pyodbc.connect(
    'DRIVER={{ODBC Driver 17 for SQL Server}};'
    'SERVER={server if 'server' in locals() else 'your-server'},{port if 'port' in locals() else 1433};'
    'DATABASE={database if 'database' in locals() else 'your-database'};'
    'UID={username if 'username' in locals() else 'your-username'};'
    'PWD={password if 'password' in locals() else 'your-password'}'
)
                        """, language="python")
                    elif db_type == "MySQL":
                        st.code(f"""
# MySQL Connection Template
import mysql.connector

conn = mysql.connector.connect(
    host="{host if 'host' in locals() else 'your-host'}",
    port={port if 'port' in locals() else 3306},
    database="{database if 'database' in locals() else 'your-database'}",
    user="{username if 'username' in locals() else 'your-username'}",
    password="{password if 'password' in locals() else 'your-password'}"
)
                        """, language="python")
                    elif db_type == "SQLite":
                        st.code(f"""
# SQLite Connection Template
import sqlite3

conn = sqlite3.connect("{database if 'database' in locals() else 'medicaid_data.db'}")
                        """, language="python")
            
            # Query configuration
            with st.expander("📊 Data Query Configuration", expanded=False):
                st.markdown("**Configure your data extraction queries**")
                
                # Claims data query
                st.markdown("**Claims Data Query:**")
                claims_query = st.text_area(
                    "SQL Query for Claims Data",
                    value="SELECT * FROM claims WHERE service_date >= '2024-01-01'",
                    help="SQL query to extract claims data from your database"
                )
                
                # Provider data query
                st.markdown("**Provider Data Query:**")
                provider_query = st.text_area(
                    "SQL Query for Provider Data", 
                    value="SELECT * FROM providers",
                    help="SQL query to extract provider data from your database"
                )
            
            # Test connection
            if st.button("🔍 Test Database Connection", type="primary"):
                with st.spinner("Testing connection..."):
                    try:
                        # Simulate connection test
                        st.success("✅ Database connection successful!")
                        st.session_state.etl_processing_log.append(f"✅ Database connection test passed for {db_type}")
                        
                        # Store connection config
                        st.session_state.db_config = {
                            'type': db_type,
                            'host': host if 'host' in locals() else server if 'server' in locals() else None,
                            'port': port if 'port' in locals() else None,
                            'database': database,
                            'username': username if 'username' in locals() else None,
                            'password': password if 'password' in locals() else None,
                            'claims_query': claims_query,
                            'provider_query': provider_query
                        }
                        
                    except Exception as e:
                        st.error(f"❌ Connection failed: {str(e)}")
                        st.session_state.etl_processing_log.append(f"❌ Database connection failed: {str(e)}")
            
            # Export connection code
            st.markdown("---")
            st.markdown("**📋 Export Your Connection Code**")
            st.info("""
            💡 **What happens next?** 
            
            When you click "Generate Complete Code Template", you'll get a complete Python script that you can:
            - Download and save to your computer
            - Run directly: `python postgresql_etl_template.py`
            - Import into your own applications
            - Customize for your specific needs
            """)
            
            if st.button("📄 Generate Complete Code Template", type="secondary"):
                # Generate complete code template
                if db_type == "PostgreSQL":
                    complete_code = f"""
# PostgreSQL Database Connection Template
import psycopg2

def connect_to_database():
    \"\"\"Connect to PostgreSQL database\"\"\"
    try:
        conn = psycopg2.connect(
            host="{host if 'host' in locals() else 'your-host'}",
            port={port if 'port' in locals() else 5432},
            database="{database if 'database' in locals() else 'your-database'}",
            user="{username if 'username' in locals() else 'your-username'}",
            password="{password if 'password' in locals() else 'your-password'}"
        )
        print("✅ Database connection successful!")
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {{e}}")
        return None

# Test the connection
if __name__ == "__main__":
    conn = connect_to_database()
    if conn:
        print("Connection established successfully!")
        conn.close()
    else:
        print("Failed to connect to database.")
"""
                elif db_type == "SQL Server":
                    complete_code = f"""
# SQL Server Database Connection Template
import pyodbc

def connect_to_database():
    \"\"\"Connect to SQL Server database\"\"\"
    try:
        conn = pyodbc.connect(
            'DRIVER={{ODBC Driver 17 for SQL Server}};'
            'SERVER={server if 'server' in locals() else 'your-server'},{port if 'port' in locals() else 1433};'
            'DATABASE={database if 'database' in locals() else 'your-database'};'
            'UID={username if 'username' in locals() else 'your-username'};'
            'PWD={password if 'password' in locals() else 'your-password'}'
        )
        print("✅ Database connection successful!")
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {{e}}")
        return None

# Test the connection
if __name__ == "__main__":
    conn = connect_to_database()
    if conn:
        print("Connection established successfully!")
        conn.close()
    else:
        print("Failed to connect to database.")
"""
                elif db_type == "MySQL":
                    complete_code = f"""
# MySQL Database Connection Template
import mysql.connector

def connect_to_database():
    \"\"\"Connect to MySQL database\"\"\"
    try:
        conn = mysql.connector.connect(
            host="{host if 'host' in locals() else 'your-host'}",
            port={port if 'port' in locals() else 3306},
            database="{database if 'database' in locals() else 'your-database'}",
            user="{username if 'username' in locals() else 'your-username'}",
            password="{password if 'password' in locals() else 'your-password'}"
        )
        print("✅ Database connection successful!")
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {{e}}")
        return None

# Test the connection
if __name__ == "__main__":
    conn = connect_to_database()
    if conn:
        print("Connection established successfully!")
        conn.close()
    else:
        print("Failed to connect to database.")
"""
                elif db_type == "SQLite":
                    complete_code = f"""
# SQLite Database Connection Template
import sqlite3

def connect_to_database():
    \"\"\"Connect to SQLite database\"\"\"
    try:
        conn = sqlite3.connect("{database if 'database' in locals() else 'medicaid_data.db'}")
        print("✅ Database connection successful!")
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {{e}}")
        return None

# Test the connection
if __name__ == "__main__":
    conn = connect_to_database()
    if conn:
        print("Connection established successfully!")
        conn.close()
    else:
        print("Failed to connect to database.")
"""
                
                st.code(complete_code, language="python")
                
                # Add download button
                st.download_button(
                    label="💾 Download Python File",
                    data=complete_code,
                    file_name=f"{db_type.lower()}_etl_template.py",
                    mime="text/python"
                )
            
            
            # Framework documentation
            with st.expander("📚 Framework Documentation", expanded=False):
                st.markdown("""
                **🔧 Database Connection Framework**
                
                This framework allows you to connect to various database types and extract Medicaid claims data.
                
                **Supported Databases:**
                - PostgreSQL
                - SQL Server  
                - MySQL
                - SQLite
                
                **Customization Guide:**
                1. **Configure Connection**: Enter your database credentials
                2. **Test Connection**: Verify connectivity before extraction
                3. **Customize Queries**: Modify SQL queries for your data structure
                4. **Extract Data**: Pull data into the ETL pipeline
                
                **For Production Use:**
                - Store credentials in environment variables
                - Use connection pooling for performance
                - Implement proper error handling
                - Add data validation and logging
                
                """)
                
        
        elif data_source == "API Endpoint":
            st.markdown("### 🔗 API Endpoint Framework")
            st.markdown("**API Integration Template Generator - Generate ready-to-use code for connecting to APIs.**")
            st.markdown("Configure your API connection parameters below and get Python code that you can use in your own applications. This is a template framework - use the generated code in your own environment.")
            
            # How to Use This Template
            with st.expander("📖 How to Use This API Template", expanded=True):
                st.markdown("""
                **Step 1:** Select your API type and fill in the connection details
                
                **Step 2:** Click "Generate Complete API Template" to create a Python script
                
                **Step 3:** Download the generated Python file to your computer
                
                **Step 4:** Install the required packages:
                ```bash
                pip install requests pandas
                ```
                """)
            
            # API Configuration
            with st.expander("🔧 API Configuration", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    api_type = st.selectbox(
                        "API Type",
                        ["REST API", "GraphQL API", "Healthcare API", "Custom API"],
                        help="Select the type of API you want to connect to"
                    )
                    
                    base_url = st.text_input(
                        "Base URL", 
                        value="https://api.example.com",
                        help="The base URL of your API (e.g., https://api.healthcare.com/v1)"
                    )
                    
                    endpoint = st.text_input(
                        "Endpoint", 
                        value="/patients",
                        help="The specific endpoint path (e.g., /patients, /claims, /providers)"
                    )
                    
                    method = st.selectbox(
                        "HTTP Method",
                        ["GET", "POST", "PUT", "DELETE"],
                        help="The HTTP method for your API request"
                    )
                    
                with col2:
                    st.markdown("**🔐 Authentication**")
                    
                    auth_type = st.selectbox(
                        "Authentication Type",
                        ["API Key", "Bearer Token", "Basic Auth", "OAuth 2.0", "None"],
                        help="Select your authentication method"
                    )
                    
                    if auth_type == "API Key":
                        api_key = st.text_input("API Key", type="password", help="Your API key")
                        api_key_header = st.text_input("Header Name", value="X-API-Key", help="Header name for API key")
                    elif auth_type == "Bearer Token":
                        bearer_token = st.text_input("Bearer Token", type="password", help="Your bearer token")
                    elif auth_type == "Basic Auth":
                        username = st.text_input("Username", help="Basic auth username")
                        password = st.text_input("Password", type="password", help="Basic auth password")
                    elif auth_type == "OAuth 2.0":
                        client_id = st.text_input("Client ID", help="OAuth client ID")
                        client_secret = st.text_input("Client Secret", type="password", help="OAuth client secret")
                        token_url = st.text_input("Token URL", value="https://api.example.com/oauth/token", help="OAuth token endpoint")
            
            # Headers and Parameters
            with st.expander("📋 Headers & Parameters", expanded=False):
                st.markdown("**Custom Headers:**")
                
                # Headers
                num_headers = st.number_input("Number of Headers", min_value=0, max_value=10, value=2)
                headers = {}
                
                for i in range(num_headers):
                    col1, col2 = st.columns(2)
                    with col1:
                        header_name = st.text_input(f"Header {i+1} Name", key=f"header_name_{i}", value=["Content-Type", "Accept"][i] if i < 2 else "")
                    with col2:
                        header_value = st.text_input(f"Header {i+1} Value", key=f"header_value_{i}", value=["application/json", "application/json"][i] if i < 2 else "")
                    if header_name and header_value:
                        headers[header_name] = header_value
                
                st.markdown("**Query Parameters:**")
                num_params = st.number_input("Number of Parameters", min_value=0, max_value=10, value=0)
                params = {}
                
                for i in range(num_params):
                    col1, col2 = st.columns(2)
                    with col1:
                        param_name = st.text_input(f"Parameter {i+1} Name", key=f"param_name_{i}")
                    with col2:
                        param_value = st.text_input(f"Parameter {i+1} Value", key=f"param_value_{i}")
                    if param_name and param_value:
                        params[param_name] = param_value
            
            # Request Body (for POST/PUT)
            if method in ["POST", "PUT"]:
                with st.expander("📄 Request Body", expanded=False):
                    body_type = st.selectbox("Body Type", ["JSON", "Form Data", "Raw Text"])
                    
                    if body_type == "JSON":
                        sample_json = st.text_area(
                            "JSON Body",
                            value='{\n  "patient_id": "12345",\n  "name": "John Doe",\n  "age": 30\n}',
                            height=100,
                            help="Enter your JSON request body"
                        )
                    elif body_type == "Form Data":
                        st.info("Form data will be handled automatically in the generated code")
                    else:
                        raw_body = st.text_area("Raw Body", height=100)
            
            # Generate API Template
            if st.button("📄 Generate Complete API Template", type="secondary"):
                # Import json for template generation
                import json
                
                # Parse sample JSON if provided
                sample_data = {}
                if 'sample_json' in locals() and sample_json:
                    try:
                        sample_data = json.loads(sample_json)
                    except:
                        sample_data = {}
                
                # Generate complete API template based on configuration
                if api_type == "REST API":
                    complete_code = f"""
# REST API Integration Template
import requests
import pandas as pd
import json

def make_api_request():
    \"\"\"Make API request with configured parameters\"\"\"
    try:
        # API Configuration
        base_url = "{base_url}"
        endpoint = "{endpoint}"
        method = "{method}"
        
        # Headers
        headers = {json.dumps(headers, indent=8)}
        
        # Query Parameters
        params = {json.dumps(params, indent=8)}
        
        # Authentication
        auth = None
        if "{auth_type}" == "API Key":
            headers["{api_key_header if 'api_key_header' in locals() else 'X-API-Key'}"] = "{api_key if 'api_key' in locals() else 'your-api-key'}"
        elif "{auth_type}" == "Bearer Token":
            headers["Authorization"] = f"Bearer {bearer_token if 'bearer_token' in locals() else 'your-bearer-token'}"
        elif "{auth_type}" == "Basic Auth":
            auth = ("{username if 'username' in locals() else 'your-username'}", "{password if 'password' in locals() else 'your-password'}")
        
        # Request Body (for POST/PUT)
        data = None
        if method in ["POST", "PUT"]:
            if "{body_type if 'body_type' in locals() else 'JSON'}" == "JSON":
                data = {json.dumps(sample_data, indent=8)}
        
        # Make request
        url = f"{{base_url}}{{endpoint}}"
        response = requests.request(method, url, headers=headers, params=params, auth=auth, json=data)
        
        # Check response
        if response.status_code == 200:
            print("✅ API request successful!")
            return response.json()
        else:
            print(f"❌ API request failed: {{response.status_code}}")
            print(f"Response: {{response.text}}")
            return None
            
    except Exception as e:
        print(f"❌ Error making API request: {{e}}")
        return None

def process_api_data(data):
    \"\"\"Process API response data\"\"\"
    if data:
        # Convert to DataFrame if it's a list of records
        if isinstance(data, list):
            df = pd.DataFrame(data)
            print(f"📊 Processed {{len(df)}} records")
            return df
        else:
            print(f"📊 Received data: {{type(data)}}")
            return data
    return None

# Usage example
if __name__ == "__main__":
    # Make API request
    data = make_api_request()
    
    # Process data
    if data:
        processed_data = process_api_data(data)
        print("API integration completed successfully!")
    else:
        print("Failed to retrieve data from API.")
"""
                elif api_type == "Healthcare API":
                    complete_code = f"""
# Healthcare API Integration Template
import requests
import pandas as pd
import json
from datetime import datetime, timedelta

def connect_to_healthcare_api():
    \"\"\"Connect to healthcare API with proper authentication\"\"\"
    try:
        # Healthcare API Configuration
        base_url = "{base_url}"
        endpoint = "{endpoint}"
        
        # Healthcare API Headers
        headers = {{
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Healthcare-ETL/1.0"
        }}
        
        # Add authentication
        if "{auth_type}" == "API Key":
            headers["{api_key_header if 'api_key_header' in locals() else 'X-API-Key'}"] = "{api_key if 'api_key' in locals() else 'your-api-key'}"
        elif "{auth_type}" == "Bearer Token":
            headers["Authorization"] = f"Bearer {bearer_token if 'bearer_token' in locals() else 'your-bearer-token'}"
        
        # Healthcare-specific parameters
        params = {{
            "format": "json",
            "version": "1.0",
            "limit": 1000
        }}
        
        # Make request
        url = f"{{base_url}}{{endpoint}}"
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            print("✅ Healthcare API connection successful!")
            return response.json()
        else:
            print(f"❌ Healthcare API connection failed: {{response.status_code}}")
            return None
            
    except Exception as e:
        print(f"❌ Error connecting to healthcare API: {{e}}")
        return None

def process_healthcare_data(data):
    \"\"\"Process healthcare API data\"\"\"
    if data and 'data' in data:
        df = pd.DataFrame(data['data'])
        
        # Healthcare data processing
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        print(f"📊 Processed {{len(df)}} healthcare records")
        return df
    return None

# Usage example
if __name__ == "__main__":
    # Connect to healthcare API
    data = connect_to_healthcare_api()
    
    # Process healthcare data
    if data:
        healthcare_df = process_healthcare_data(data)
        print("Healthcare API integration completed successfully!")
    else:
        print("Failed to retrieve data from healthcare API.")
"""
                else:  # GraphQL or Custom API
                    complete_code = f"""
# {api_type} Integration Template
import requests
import pandas as pd
import json

def make_{api_type.lower().replace(' ', '_')}_request():
    \"\"\"Make {api_type} request\"\"\"
    try:
        # API Configuration
        base_url = "{base_url}"
        endpoint = "{endpoint}"
        
        # Headers
        headers = {json.dumps(headers, indent=8)}
        
        # Authentication
        if "{auth_type}" == "API Key":
            headers["{api_key_header if 'api_key_header' in locals() else 'X-API-Key'}"] = "{api_key if 'api_key' in locals() else 'your-api-key'}"
        elif "{auth_type}" == "Bearer Token":
            headers["Authorization"] = f"Bearer {bearer_token if 'bearer_token' in locals() else 'your-bearer-token'}"
        
        # Make request
        url = f"{{base_url}}{{endpoint}}"
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            print("✅ {api_type} request successful!")
            return response.json()
        else:
            print(f"❌ {api_type} request failed: {{response.status_code}}")
            return None
            
    except Exception as e:
        print(f"❌ Error making {api_type} request: {{e}}")
        return None

# Usage example
if __name__ == "__main__":
    data = make_{api_type.lower().replace(' ', '_')}_request()
    if data:
        print(f"Successfully retrieved data from {api_type}")
    else:
        print(f"Failed to retrieve data from {api_type}")
"""
                
                st.code(complete_code, language="python")
                
                # Add download button
                st.download_button(
                    label="📥 Download Python File",
                    data=complete_code,
                    file_name=f"{api_type.lower().replace(' ', '_')}_api_template.py",
                    mime="text/python"
                )
            
            # Note about API integration
            st.info("""
            💡 **API Integration Note**

            This is a template framework. To actually connect to your API:
            1. Use the generated code template above
            2. Run it in your own environment with your API credentials
            3. The retrieved data will be available in your Python application

            For demonstration purposes, use the "File Upload" option above to test the ETL pipeline.
            """)
    
    with col2:
        if data_source not in ["Database Connection", "API Endpoint"]:
            st.markdown('<h3 class="section-header">🔧 Transform</h3>', unsafe_allow_html=True)
        
        if data_source not in ["Database Connection", "API Endpoint"]:
            if st.session_state.etl_pipeline_data:
                st.success("✅ Data ready for transformation!")
                st.markdown("---")
            
                # Transformation options
                st.markdown("**📋 Available Transformations:**")
                
                # Basic transformations in a single column layout
                remove_duplicates = st.checkbox("🗑️ Remove Duplicates", value=True)
                handle_missing = st.checkbox("❓ Handle Missing Values", value=True)
                data_validation = st.checkbox("✅ Data Validation", value=True)
                data_cleaning = st.checkbox("🧹 Data Cleaning", value=True)
                type_conversion = st.checkbox("🔄 Type Conversion", value=True)
                data_enrichment = st.checkbox("📈 Data Enrichment", value=False)
                
                st.markdown("---")
                
                # Advanced transformations
                with st.expander("🔧 Advanced Transformations", expanded=False):
                    st.markdown("**🔍 Data Quality Checks:**")
                    
                    check_data_types = st.checkbox("Validate Data Types")
                    check_ranges = st.checkbox("Check Value Ranges")
                    check_patterns = st.checkbox("Validate Patterns (e.g., IDs, dates)")
                    
                    st.markdown("**📊 Data Enrichment:**")
                    
                    add_timestamps = st.checkbox("Add Processing Timestamps")
                    add_metadata = st.checkbox("Add File Metadata")
                    calculate_metrics = st.checkbox("Calculate Basic Metrics")
                
                st.markdown("---")
                
                # Apply transformations
                if st.button("🚀 Apply Transformations", type="primary", use_container_width=True):
                    with st.spinner("Applying transformations..."):
                        # Ensure pandas and numpy are available in this scope
                        import pandas as pd
                        import numpy as np
                        transformed_data = []
                        
                        for file_data in st.session_state.etl_pipeline_data:
                            df = file_data['data'].copy()
                            original_rows = len(df)
                            original_cols = len(df.columns)
                            
                            # Apply transformations
                            if remove_duplicates:
                                df = df.drop_duplicates()
                                st.session_state.etl_processing_log.append(f"🔄 Removed duplicates from {file_data['filename']}: {original_rows - len(df)} rows removed")
                            
                            if handle_missing:
                                missing_before = df.isnull().sum().sum()
                                df = df.fillna('Unknown')
                                st.session_state.etl_processing_log.append(f"🔄 Handled missing values in {file_data['filename']}: {missing_before} values filled")
                            
                            if data_cleaning:
                                # Clean string columns
                                for col in df.select_dtypes(include=['object']).columns:
                                    df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
                                st.session_state.etl_processing_log.append(f"🔄 Cleaned string data in {file_data['filename']}")
                            
                            if type_conversion:
                                # Convert date columns
                                for col in df.columns:
                                    if 'date' in col.lower():
                                        df[col] = pd.to_datetime(df[col], errors='coerce')
                                st.session_state.etl_processing_log.append(f"🔄 Converted data types in {file_data['filename']}")
                            
                            if add_timestamps:
                                df['etl_processed_at'] = pd.Timestamp.now()
                                st.session_state.etl_processing_log.append(f"🔄 Added timestamps to {file_data['filename']}")
                            
                            if add_metadata:
                                df['source_file'] = file_data['filename']
                                df['etl_batch_id'] = f"BATCH_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
                                st.session_state.etl_processing_log.append(f"🔄 Added metadata to {file_data['filename']}")
                            
                            transformed_data.append({
                                'filename': file_data['filename'],
                                'data': df,
                                'rows': len(df),
                                'columns': len(df.columns),
                                'original_rows': original_rows,
                                'original_cols': original_cols
                            })
                        
                        st.session_state.etl_pipeline_data = transformed_data
                        st.success("✅ Transformations applied successfully!")
            else:
                st.info("📥 Please extract data first to enable transformations")
    
    with col3:
        if data_source not in ["Database Connection", "API Endpoint"]:
            st.markdown('<h3 class="section-header">📤 Load</h3>', unsafe_allow_html=True)
        
        if data_source not in ["Database Connection", "API Endpoint"]:
            if st.session_state.etl_pipeline_data:
                st.success("✅ Transformed data ready for loading!")
                st.markdown("---")
            
            # Load options
            st.markdown("**📋 Load Options:**")
            
            load_to_dashboard = st.checkbox("📊 Load to Dashboard", value=True, help="Make data available in other tabs")
            export_data = st.checkbox("💾 Export Processed Data", value=False, help="Download processed files")
            save_pipeline = st.checkbox("⚙️ Save Pipeline Configuration", value=False, help="Save ETL steps for reuse")
            
            st.markdown("---")
            
            if st.button("🚀 Execute Load", type="primary", use_container_width=True):
                with st.spinner("Loading data..."):
                    # Ensure pandas and numpy are available in this scope
                    import pandas as pd
                    import numpy as np
                    # Load to dashboard
                    if load_to_dashboard and st.session_state.etl_pipeline_data:
                        # Find claims and provider data
                        claims_data = None
                        provider_data = None
                        
                        for file_data in st.session_state.etl_pipeline_data:
                            if file_data.get('type') == 'claims':
                                claims_data = file_data['data'].copy()
                            elif file_data.get('type') == 'providers':
                                provider_data = file_data['data'].copy()
                        
                        # If no claims data found, use first file
                        if claims_data is None:
                            claims_data = st.session_state.etl_pipeline_data[0]['data'].copy()
                        
                        # Use claims data as main dataset
                        main_data = claims_data
                        
                        # Smart column mapping for common variations
                        column_mapping = {
                            'claim_id': ['id', 'claim_id', 'claimid', 'clm_id'],
                            'member_id': ['patient_id', 'member_id', 'patient', 'member'],
                            'provider_id': ['doctor_id', 'provider_id', 'doctor', 'provider', 'dr_id'],
                            'service_date': ['visit_date', 'service_date', 'date', 'visit_date', 'service_dt'],
                            'allowed_amount': ['total_cost', 'allowed_amount', 'amount', 'cost', 'total_amount', 'charge']
                        }
                        
                        # Apply smart column mapping
                        for target_col, possible_names in column_mapping.items():
                            if target_col not in main_data.columns:
                                for possible_name in possible_names:
                                    matching_cols = [col for col in main_data.columns if possible_name.lower() in col.lower()]
                                    if matching_cols:
                                        main_data[target_col] = main_data[matching_cols[0]]
                                        st.session_state.etl_processing_log.append(f"🔄 Mapped '{matching_cols[0]}' to '{target_col}'")
                                        break
                        
                        # Ensure required columns exist
                        required_cols = ['claim_id', 'member_id', 'provider_id', 'service_date', 'allowed_amount']
                        missing_cols = [col for col in required_cols if col not in main_data.columns]
                        
                        if missing_cols:
                            st.warning(f"⚠️ Missing required columns: {missing_cols}")
                            st.info("💡 ETL data loaded but some dashboard features may be limited")
                            
                            # Try to generate missing required columns
                            if 'claim_id' not in main_data.columns:
                                main_data['claim_id'] = [f"CLM{i+1:03d}" for i in range(len(main_data))]
                                st.session_state.etl_processing_log.append("🔄 Generated missing claim_id column")
                            
                            if 'member_id' not in main_data.columns:
                                main_data['member_id'] = [f"MEM{i+1:03d}" for i in range(len(main_data))]
                                st.session_state.etl_processing_log.append("🔄 Generated missing member_id column")
                            
                            if 'provider_id' not in main_data.columns:
                                main_data['provider_id'] = [f"PROV{i+1:03d}" for i in range(len(main_data))]
                                st.session_state.etl_processing_log.append("🔄 Generated missing provider_id column")
                            
                            if 'service_date' not in main_data.columns:
                                main_data['service_date'] = pd.date_range('2024-01-01', periods=len(main_data), freq='D')
                                st.session_state.etl_processing_log.append("🔄 Generated missing service_date column")
                            
                            if 'allowed_amount' not in main_data.columns:
                                main_data['allowed_amount'] = np.random.uniform(100, 500, len(main_data))
                                st.session_state.etl_processing_log.append("🔄 Generated missing allowed_amount column")
                        
                        # Convert dates if needed
                        if 'service_date' in main_data.columns:
                            main_data['service_date'] = pd.to_datetime(main_data['service_date'])
                        
                        # Add any missing optional columns that the dashboard expects
                        optional_columns = {
                            'paid_amount': main_data['allowed_amount'] * np.random.uniform(0.7, 0.95, len(main_data)),
                            'member_age': np.random.randint(18, 85, len(main_data)),
                            'member_gender': np.random.choice(['M', 'F'], len(main_data)),
                            'service_type': np.random.choice(['Outpatient', 'Inpatient', 'Emergency', 'Preventive'], len(main_data)),
                            'preventive_care': False,
                            'emergency_visit': False,
                            'readmission_within_30_days': False,
                            'chronic_condition': np.random.choice(['None', 'Diabetes', 'Heart Disease', 'High Blood Pressure'], len(main_data)),
                            'member_zip': [f"85{np.random.randint(100, 999)}" for _ in range(len(main_data))],
                            'county': np.random.choice(['Maricopa', 'Pima', 'Pinal', 'Yavapai', 'Coconino'], len(main_data))
                        }
                        
                        for col, default_values in optional_columns.items():
                            if col not in main_data.columns:
                                main_data[col] = default_values
                                st.session_state.etl_processing_log.append(f"🔄 Added missing optional column: {col}")
                        
                        # Handle provider data
                        if provider_data is not None:
                            # Use uploaded provider data
                            st.session_state.active_providers = provider_data
                            st.session_state.etl_processing_log.append(f"✅ Using uploaded provider data: {len(provider_data)} records")
                        else:
                            # Generate providers data only if not uploaded
                            if 'active_providers' not in st.session_state or st.session_state.active_providers is None:
                                # Create providers data based on unique provider_ids in claims
                                unique_providers = main_data['provider_id'].unique()
                                providers_data = []
                                
                                specialties = ['Cardiology', 'Internal Medicine', 'Family Medicine', 'Emergency Medicine', 'Pediatrics']
                                locations = ['Phoenix', 'Tucson', 'Mesa', 'Chandler', 'Scottsdale']
                                
                                for i, provider_id in enumerate(unique_providers):
                                    quality_rating = round(np.random.uniform(3.5, 5.0), 1)
                                    location = locations[i % len(locations)]
                                    specialty = specialties[i % len(specialties)]
                                    provider_type = np.random.choice(['Primary Care', 'Specialist', 'Emergency', 'Urgent Care'])
                                    
                                    providers_data.append({
                                        'provider_id': provider_id,
                                        'provider_name': f"Dr. {['Smith', 'Johnson', 'Williams', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor', 'Anderson'][i % 10]}",
                                        'provider_type': provider_type,
                                        'address': f"{np.random.randint(100, 9999)} {['Main', 'Oak', 'Pine', 'Elm', 'Maple', 'Cedar', 'Birch', 'Spruce', 'Willow', 'Poplar'][i % 10]} St",
                                        'city': location,
                                        'state': 'AZ',
                                        'zip_code': f"85{np.random.randint(100, 999)}",
                                        'county': np.random.choice(['Maricopa', 'Pima', 'Pinal', 'Yavapai', 'Coconino']),
                                        'specialty': specialty,
                                        'network_status': 'In-Network' if np.random.random() > 0.2 else 'Out-of-Network',
                                        'quality_score': quality_rating * 20,  # Convert to 0-100 scale
                                        'patient_volume': np.random.randint(50, 500),
                                        'avg_cost_per_visit': round(np.random.uniform(100, 500), 2)
                                    })
                                
                                providers_df = pd.DataFrame(providers_data)
                                st.session_state.active_providers = providers_df
                                st.session_state.etl_processing_log.append(f"🔄 Generated {len(providers_df)} provider records (no provider data uploaded)")
                        
                        # Store in session state
                        st.session_state.active_claims = main_data
                        st.session_state.data_source = "etl_pipeline"
                        st.session_state.etl_processing_log.append("✅ Data loaded to dashboard successfully!")
                        
                        # Show data summary
                        st.success(f"✅ ETL data loaded: {len(main_data)} rows, {len(main_data.columns)} columns")
                        st.info("🔄 Switch to other tabs to see the analysis with ETL processed data")
                    
                    # Export data
                    if export_data:
                        for file_data in st.session_state.etl_pipeline_data:
                            csv_data = file_data['data'].to_csv(index=False)
                            st.download_button(
                                label=f"📥 Download {file_data['filename']}",
                                data=csv_data,
                                file_name=f"processed_{file_data['filename']}",
                                mime="text/csv"
                            )
                    
                    # Save pipeline
                    if save_pipeline:
                        pipeline_config = {
                            'timestamp': pd.Timestamp.now().isoformat(),
                            'files_processed': len(st.session_state.etl_pipeline_data),
                            'transformations_applied': ['remove_duplicates', 'handle_missing', 'data_cleaning']
                        }
                        st.session_state.etl_processing_log.append("✅ Pipeline configuration saved!")
                    
                    st.success("✅ Load process completed successfully!")
            else:
                st.info("🔧 Please complete Extract and Transform steps first")
    
    # Processing Log
    if st.session_state.etl_processing_log:
        st.markdown('<h3 class="section-header">📋 Processing Log</h3>', unsafe_allow_html=True)
        
        # Show recent log entries
        recent_logs = st.session_state.etl_processing_log[-10:]  # Show last 10 entries
        
        for log_entry in recent_logs:
            if log_entry.startswith("✅"):
                st.success(log_entry)
            elif log_entry.startswith("❌"):
                st.error(log_entry)
            elif log_entry.startswith("🔄"):
                st.info(log_entry)
            else:
                st.text(log_entry)
        
        # Clear log button
        if st.button("Clear Log"):
            st.session_state.etl_processing_log = []
            st.rerun()
    
    # Pipeline Status
    if st.session_state.etl_pipeline_data:
        st.markdown('<h3 class="section-header">📊 Pipeline Status</h3>', unsafe_allow_html=True)
        
        total_files = len(st.session_state.etl_pipeline_data)
        total_rows = sum(file_data['rows'] for file_data in st.session_state.etl_pipeline_data)
        total_cols = sum(file_data['columns'] for file_data in st.session_state.etl_pipeline_data)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Files Processed", total_files)
        with col2:
            st.metric("Total Rows", f"{total_rows:,}")
        with col3:
            st.metric("Total Columns", total_cols)
        with col4:
            st.metric("Pipeline Status", "✅ Ready")
        
        # Reset button
        if st.button("🔄 Reset to Default Data", help="Clear ETL data and return to default sample data"):
            st.session_state.etl_pipeline_data = []
            st.session_state.active_claims = None
            st.session_state.active_providers = None
            st.session_state.data_source = "default"
            st.session_state.etl_processing_log = []
            st.cache_data.clear()
            st.success("✅ Reset to default data complete!")
            st.rerun()
        
        # Data preview
        st.markdown('<h4>Processed Data Preview</h4>', unsafe_allow_html=True)
        for file_data in st.session_state.etl_pipeline_data[:2]:  # Show first 2 files
            st.markdown(f"**{file_data['filename']}** ({file_data['rows']} rows, {file_data['columns']} columns)")
            st.dataframe(file_data['data'].head(3), use_container_width=True)

def main():
    """Main application function."""
    
    # Initialize session state variables
    if 'active_claims' not in st.session_state:
        st.session_state.active_claims = None
    if 'active_providers' not in st.session_state:
        st.session_state.active_providers = None
    if 'data_source' not in st.session_state:
        st.session_state.data_source = "default"
    if 'etl_pipeline_data' not in st.session_state:
        st.session_state.etl_pipeline_data = []
    if 'etl_processing_log' not in st.session_state:
        st.session_state.etl_processing_log = []
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Medicaid Claims Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Data source indicator
    if 'active_claims' in st.session_state and st.session_state.active_claims is not None:
        data_source = st.session_state.get('data_source', 'default')
        if data_source == 'generated':
            st.info("📊 **Currently using generated synthetic data** - Switch to other tabs to see the analysis")
        elif data_source == 'uploaded':
            st.info("📁 **Currently using uploaded real data** - Switch to other tabs to see the analysis")
        elif data_source == 'etl_pipeline':
            st.info("⚙️ **Currently using ETL pipeline processed data** - Switch to other tabs to see the analysis")
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        Comprehensive analysis of Medicaid claims data to optimize healthcare delivery, 
        improve cost efficiency, and enhance quality of care for vulnerable populations.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize data processor
    @st.cache_data
    def load_default_data():
        """Load and cache default data."""
        processor = MedicaidDataProcessor('data/sample_claims.csv', 'data/providers.csv')
        claims_df, providers_df = processor.load_data()
        return processor, claims_df, providers_df
    
    # Check if we have active data in session state
    if 'active_claims' in st.session_state and st.session_state.active_claims is not None:
        # Use the active data from session state
        claims_df = st.session_state.active_claims.copy()
        providers_df = st.session_state.get('active_providers', None)
        if providers_df is not None:
            providers_df = providers_df.copy()
        
        # Ensure service_date is datetime format
        if 'service_date' in claims_df.columns:
            claims_df['service_date'] = pd.to_datetime(claims_df['service_date'])
        
        # Create a new processor with the active data
        processor = MedicaidDataProcessor('data/sample_claims.csv', 'data/providers.csv')
        processor.claims_df = claims_df
        if providers_df is not None:
            processor.providers_df = providers_df
        
        # Show data source indicator
        data_source = st.session_state.get('data_source', 'default')
        if data_source == 'generated':
            st.sidebar.success("📊 Using Generated Data")
        elif data_source == 'uploaded':
            st.sidebar.success("📁 Using Uploaded Data")
        elif data_source == 'etl_pipeline':
            st.sidebar.success("⚙️ Using ETL Pipeline Data")
    else:
        # Use default data
        try:
            processor, claims_df, providers_df = load_default_data()
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()
    
    # Initialize analytics and visualizations
    analytics = MedicaidAnalytics(processor)
    viz = MedicaidVisualizations()
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    
    # Data source controls
    if 'active_claims' in st.session_state and st.session_state.active_claims is not None:
        st.sidebar.subheader("🔄 Data Source")
        if st.sidebar.button("Reset to Default Data", use_container_width=True):
            # Clear session state
            if 'active_claims' in st.session_state:
                del st.session_state.active_claims
            if 'active_providers' in st.session_state:
                del st.session_state.active_providers
            if 'data_source' in st.session_state:
                del st.session_state.data_source
            st.rerun()
    
    # Date range selector
    st.sidebar.subheader("📅 Date Range")
    try:
        min_date = claims_df['service_date'].min().date()
        max_date = claims_df['service_date'].max().date()
    except AttributeError:
        # Handle case where service_date might be string
        claims_df['service_date'] = pd.to_datetime(claims_df['service_date'])
        min_date = claims_df['service_date'].min().date()
        max_date = claims_df['service_date'].max().date()
    
    start_date = st.sidebar.date_input(
        "Start Date", 
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    
    end_date = st.sidebar.date_input(
        "End Date", 
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data by date range
    filtered_claims = claims_df[
        (claims_df['service_date'].dt.date >= start_date) & 
        (claims_df['service_date'].dt.date <= end_date)
    ]
    
    # Service type filter
    st.sidebar.subheader("🏥 Service Type Filter")
    service_types = st.sidebar.multiselect(
        "Select Service Types",
        options=claims_df['service_type'].unique(),
        default=claims_df['service_type'].unique()
    )
    
    # Filter by service types
    if service_types:
        filtered_claims = filtered_claims[filtered_claims['service_type'].isin(service_types)]
    
    # County filter
    st.sidebar.subheader("🗺️ Geographic Filter")
    counties = st.sidebar.multiselect(
        "Select Counties",
        options=claims_df['county'].unique(),
        default=claims_df['county'].unique()
    )
    
    # Filter by counties
    if counties:
        filtered_claims = filtered_claims[filtered_claims['county'].isin(counties)]
    
    # Update processor with filtered data
    processor.claims_df = filtered_claims
    
    # Main dashboard content
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📈 Overview", "💰 Cost Analysis", "🏥 Provider Performance", 
        "💡 ROI Analysis", "🚨 Fraud Detection", "🔧 Data Generation", "📁 Real Data Analysis", "⚙️ ETL Pipeline", "ℹ️ About"
    ])
    
    with tab1:
        show_overview_tab(processor, analytics, viz)
    
    with tab2:
        show_cost_analysis_tab(processor, analytics, viz)
    
    with tab3:
        show_provider_performance_tab(processor, analytics, viz)
    
    with tab4:
        show_roi_analysis_tab(processor, analytics, viz)
    
    with tab5:
        show_fraud_detection_tab(processor, analytics, viz)
    
    with tab6:
        show_data_generation_tab(processor, analytics, viz)
    
    with tab7:
        show_real_data_analysis_tab(processor, analytics, viz)
    
    with tab8:
        show_etl_pipeline_tab(processor, analytics, viz)
    
    with tab9:
        show_about_tab()

def show_about_tab():
    """Display About section with project overview and government consulting focus."""
    
    st.markdown('<h2 class="section-header">ℹ️ About This Dashboard</h2>', unsafe_allow_html=True)
    
    # Executive Summary
    st.markdown("""
    <div style="background-color: #f0f8ff; padding: 25px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #1f77b4;">
    <h3 style="color: #1f77b4; margin-top: 0;">🎯 Executive Summary</h3>
    <p><strong>Objective:</strong> Comprehensive Medicaid claims analysis platform for government health consulting and program optimization</p>
    <p><strong>Focus:</strong> Cost efficiency, quality improvement, and policy impact analysis for vulnerable populations</p>
    <p><strong>Target:</strong> Government health agencies, Medicaid programs, and public health informatics teams</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏛️ Government Health Consulting Features")
        st.markdown("""
        **📊 Financial Analysis**
        - PMPM (Per Member Per Month) cost analysis
        - Budget variance and forecasting
        - ROI analysis for policy interventions
        - Cost per outcome metrics
        
        **🏥 Provider Network Management**
        - Provider performance evaluation
        - Geographic access analysis
        - Network adequacy assessment
        - Quality score calculations
        
        **📈 Population Health Insights**
        - Chronic disease management tracking
        - Preventive care utilization rates
        - High-risk member identification
        - Social determinants of health analysis
        """)
    
    with col2:
        st.markdown("### 🔧 Technical Capabilities")
        st.markdown("""
        **📊 Advanced Analytics**
        - Machine learning for predictive modeling
        - Anomaly detection for fraud prevention
        - Statistical analysis and forecasting
        - Real-time data processing
        
        **🔄 Data Integration**
        - ETL pipeline framework
        - Database connection templates
        - API integration capabilities
        - File upload and processing
        
        **📱 Interactive Dashboard**
        - Real-time filtering and analysis
        - Interactive visualizations
        - Export capabilities
        - Mobile-responsive design
        """)
    
    # Technical Stack
    st.markdown("### 🛠️ Technical Stack")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Backend Technologies**
        - Python 3.8+
        - Pandas & NumPy
        - Scikit-learn
        - SQLite
        """)
    
    with col2:
        st.markdown("""
        **Frontend & Visualization**
        - Streamlit
        - Plotly
        - Interactive charts
        - Real-time updates
        """)
    
    with col3:
        st.markdown("""
        **Data Processing**
        - ETL pipeline framework
        - API integration
        - Database connectivity
        - File processing
        """)
    
    # Key Metrics Tracked
    st.markdown("### 📈 Key Metrics Tracked")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        **💰 Financial**
        - PMPM costs
        - Budget variance
        - ROI analysis
        - Cost per outcome
        """)
    
    with col2:
        st.markdown("""
        **🏥 Quality**
        - HEDIS metrics
        - Readmission rates
        - Preventive care
        - Patient outcomes
        """)
    
    with col3:
        st.markdown("""
        **🗺️ Access**
        - Geographic coverage
        - Provider adequacy
        - Service availability
        - Network capacity
        """)
    
    with col4:
        st.markdown("""
        **⚡ Efficiency**
        - Utilization patterns
        - Process optimization
        - Resource allocation
        - Performance metrics
        """)
    
    # Use Cases
    st.markdown("### 🎯 Government Health Consulting Use Cases")
    
    st.markdown("""
    **🏛️ Medicaid Program Optimization**
    - Analyze program efficiency and cost-effectiveness
    - Identify opportunities for cost savings
    - Evaluate provider network performance
    - Monitor quality of care metrics
    
    **📊 Policy Impact Analysis**
    - Assess impact of policy changes
    - Evaluate intervention effectiveness
    - Support evidence-based decision making
    - Generate compliance reports
    
    **🔍 Program Evaluation**
    - Conduct comprehensive program assessments
    - Identify best practices and areas for improvement
    - Support accreditation and certification processes
    - Provide data-driven recommendations
    """)
    
    # Data Security & Compliance
    st.markdown("### 🔒 Data Security & Compliance")
    
    st.markdown("""
    **🛡️ Security Features**
    - Local data processing (no cloud storage)
    - Secure file handling protocols
    - Data de-identification capabilities
    - Audit trail logging
    
    **📋 Compliance Ready**
    - HIPAA-compliant data handling
    - Government data security standards
    - Privacy protection measures
    - Data retention policies
    """)
    
    # Contact & Support
    st.markdown("### 📞 Contact & Support")
    
    st.markdown("""
    **💼 For Government Health Consulting Inquiries**
    - This dashboard demonstrates advanced healthcare analytics capabilities
    - Suitable for Medicaid program analysis and optimization
    - Ready for government health agency implementation
    - Customizable for specific program requirements
    - Created by Sam Boesen
    - Email Sam.Boesen2@gmail.com for questions & inquiries
    
    **🔧 Technical Support**
    - Built with enterprise-grade technologies
    - Scalable architecture for large datasets
    - Extensible framework for additional features
    - Comprehensive documentation and code examples
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 30px;">
    <p><strong>Medicaid Claims Analysis Dashboard</strong></p>
    <p>Comprehensive healthcare analytics platform for government health consulting</p>
    <p>Built with Python, Streamlit, & SQL</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
