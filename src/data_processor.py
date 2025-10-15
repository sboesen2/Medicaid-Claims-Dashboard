"""
Medicaid Claims Data Processing Module

This module handles data loading, cleaning, and processing for the Medicaid Claims Analysis Dashboard.
It demonstrates skills in data validation, quality assurance, and analytical modeling.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Tuple, Optional

class MedicaidDataProcessor:
    """
    A class to handle Medicaid claims data processing and analysis.
    Demonstrates data validation, quality assurance, and analytical modeling skills.
    """
    
    def __init__(self, claims_file: str, providers_file: str):
        """
        Initialize the data processor with claims and provider data files.
        
        Args:
            claims_file: Path to claims CSV file
            providers_file: Path to providers CSV file
        """
        self.claims_file = claims_file
        self.providers_file = providers_file
        self.claims_df = None
        self.providers_df = None
        self.processed_data = {}
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load claims and provider data from CSV files.
        Implements data validation and quality assurance processes.
        
        Returns:
            Tuple of (claims_df, providers_df)
        """
        try:
            # Load claims data
            self.claims_df = pd.read_csv(self.claims_file)
            self.claims_df['service_date'] = pd.to_datetime(self.claims_df['service_date'])
            
            # Load provider data
            self.providers_df = pd.read_csv(self.providers_file)
            
            # Data validation
            self._validate_data()
            
            return self.claims_df, self.providers_df
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def _validate_data(self) -> None:
        """
        Validate data quality and integrity.
        Ensures accuracy and reliability of analyses as required by Mercer.
        """
        # Validate claims data
        assert not self.claims_df.empty, "Claims data is empty"
        assert 'claim_id' in self.claims_df.columns, "Missing claim_id column"
        assert 'allowed_amount' in self.claims_df.columns, "Missing allowed_amount column"
        assert 'paid_amount' in self.claims_df.columns, "Missing paid_amount column"
        
        # Check for missing values in critical fields
        critical_fields = ['claim_id', 'member_id', 'provider_id', 'service_date', 'allowed_amount']
        for field in critical_fields:
            if self.claims_df[field].isnull().any():
                print(f"Warning: Missing values found in {field}")
        
        # Validate provider data
        assert not self.providers_df.empty, "Provider data is empty"
        assert 'provider_id' in self.providers_df.columns, "Missing provider_id column"
        
        print("Data validation completed successfully")
    
    def calculate_pmpm_metrics(self) -> Dict:
        """
        Calculate Cost Per Member Per Month (PMPM) metrics.
        Key metric for Medicaid program optimization and cost efficiency analysis.
        
        Returns:
            Dictionary containing PMPM metrics
        """
        if self.claims_df is None:
            self.load_data()
        
        # Calculate monthly metrics
        monthly_data = self.claims_df.groupby(
            self.claims_df['service_date'].dt.to_period('M')
        ).agg({
            'member_id': 'nunique',  # Unique members per month
            'allowed_amount': 'sum',  # Total allowed amount
            'paid_amount': 'sum',    # Total paid amount
            'claim_id': 'count'      # Total claims
        }).reset_index()
        
        # Calculate PMPM
        monthly_data['pmpm_allowed'] = monthly_data['allowed_amount'] / monthly_data['member_id']
        monthly_data['pmpm_paid'] = monthly_data['paid_amount'] / monthly_data['member_id']
        monthly_data['claims_per_member'] = monthly_data['claim_id'] / monthly_data['member_id']
        
        # Calculate overall averages
        total_members = self.claims_df['member_id'].nunique()
        total_allowed = self.claims_df['allowed_amount'].sum()
        total_paid = self.claims_df['paid_amount'].sum()
        total_claims = len(self.claims_df)
        
        overall_pmpm_allowed = total_allowed / total_members
        overall_pmpm_paid = total_paid / total_members
        overall_claims_per_member = total_claims / total_members
        
        return {
            'monthly_data': monthly_data,
            'overall_pmpm_allowed': overall_pmpm_allowed,
            'overall_pmpm_paid': overall_pmpm_paid,
            'overall_claims_per_member': overall_claims_per_member,
            'total_members': total_members,
            'total_claims': total_claims
        }
    
    def analyze_service_categories(self) -> pd.DataFrame:
        """
        Analyze spending by service category.
        Identifies trends and insights for healthcare access and quality improvement.
        
        Returns:
            DataFrame with service category analysis
        """
        if self.claims_df is None:
            self.load_data()
        
        service_analysis = self.claims_df.groupby('service_type').agg({
            'claim_id': 'count',
            'allowed_amount': ['sum', 'mean'],
            'paid_amount': ['sum', 'mean'],
            'member_id': 'nunique'
        }).round(2)
        
        # Flatten the MultiIndex columns
        service_analysis.columns = [
            'total_claims', 'total_allowed', 'avg_allowed', 
            'total_paid', 'avg_paid', 'unique_members'
        ]
        
        # Calculate percentages
        total_allowed = service_analysis['total_allowed'].sum()
        service_analysis['pct_of_total_allowed'] = (
            service_analysis['total_allowed'] / total_allowed * 100
        ).round(2)
        
        return service_analysis.sort_values('total_allowed', ascending=False)
    
    def analyze_provider_performance(self) -> pd.DataFrame:
        """
        Analyze provider performance metrics.
        Evaluates cost efficiency and effectiveness of care delivery.
        
        Returns:
            DataFrame with provider performance metrics
        """
        if self.claims_df is None or self.providers_df is None:
            self.load_data()
        
        # Merge claims with provider data
        provider_claims = self.claims_df.merge(
            self.providers_df, 
            on='provider_id', 
            how='left'
        )
        
        # Calculate provider metrics
        provider_metrics = provider_claims.groupby('provider_id').agg({
            'claim_id': 'count',
            'allowed_amount': ['sum', 'mean'],
            'paid_amount': 'sum',
            'member_id': 'nunique',
            'quality_score': 'first',
            'provider_name': 'first',
            'provider_type': 'first',
            'specialty': 'first'
        }).round(2)
        
        # Flatten the MultiIndex columns
        provider_metrics.columns = [
            'total_claims', 'total_allowed', 'avg_cost_per_claim', 'total_paid', 
            'unique_members', 'quality_score', 'provider_name', 
            'provider_type', 'specialty'
        ]
        
        # Calculate efficiency metrics
        provider_metrics['cost_per_member'] = (
            provider_metrics['total_allowed'] / provider_metrics['unique_members']
        ).round(2)
        
        provider_metrics['claims_per_member'] = (
            provider_metrics['total_claims'] / provider_metrics['unique_members']
        ).round(2)
        
        # Calculate efficiency score (combination of quality and cost)
        # Higher quality score and lower cost per member = better efficiency
        provider_metrics['efficiency_score'] = (
            provider_metrics['quality_score'] / provider_metrics['cost_per_member'] * 1000
        ).round(2)
        
        return provider_metrics.sort_values('total_allowed', ascending=False)
    
    def identify_high_cost_members(self, threshold_percentile: float = 90) -> pd.DataFrame:
        """
        Identify high-cost members for targeted interventions.
        Supports population health management and cost optimization.
        
        Args:
            threshold_percentile: Percentile threshold for high-cost identification
            
        Returns:
            DataFrame of high-cost members with their metrics
        """
        if self.claims_df is None:
            self.load_data()
        
        # Convert string boolean columns to actual booleans
        self.claims_df['emergency_visit'] = self.claims_df['emergency_visit'].map({'True': True, 'False': False})
        self.claims_df['readmission_within_30_days'] = self.claims_df['readmission_within_30_days'].map({'True': True, 'False': False})
        
        # Calculate member-level metrics
        member_metrics = self.claims_df.groupby('member_id').agg({
            'claim_id': 'count',
            'allowed_amount': 'sum',
            'paid_amount': 'sum',
            'service_date': ['min', 'max'],
            'chronic_condition': 'first',
            'member_age': 'first',
            'member_gender': 'first',
            'county': 'first',
            'emergency_visit': 'sum',
            'readmission_within_30_days': 'sum'
        }).round(2)
        
        # Flatten column names
        member_metrics.columns = [
            'total_claims', 'total_allowed', 'total_paid', 
            'first_service_date', 'last_service_date', 'chronic_condition',
            'age', 'gender', 'county', 'emergency_visits', 'readmissions'
        ]
        
        # Reset index to make member_id a regular column
        member_metrics = member_metrics.reset_index()
        
        # Calculate cost threshold
        cost_threshold = np.percentile(member_metrics['total_allowed'], threshold_percentile)
        
        # Identify high-cost members
        high_cost_members = member_metrics[
            member_metrics['total_allowed'] >= cost_threshold
        ].copy()
        
        # Calculate additional metrics
        high_cost_members['days_active'] = (
            high_cost_members['last_service_date'] - high_cost_members['first_service_date']
        ).dt.days + 1
        
        high_cost_members['cost_per_day'] = (
            high_cost_members['total_allowed'] / high_cost_members['days_active']
        ).round(2)
        
        return high_cost_members.sort_values('total_allowed', ascending=False)
    
    def calculate_quality_metrics(self) -> Dict:
        """
        Calculate quality metrics for healthcare delivery.
        Supports quality improvement initiatives and outcome measurement.
        
        Returns:
            Dictionary containing quality metrics
        """
        if self.claims_df is None:
            self.load_data()
        
        # Convert string boolean columns to actual booleans with error handling
        try:
            self.claims_df['preventive_care'] = self.claims_df['preventive_care'].map({'True': True, 'False': False})
        except:
            self.claims_df['preventive_care'] = self.claims_df['preventive_care'].astype(bool)
        
        try:
            self.claims_df['emergency_visit'] = self.claims_df['emergency_visit'].map({'True': True, 'False': False})
        except:
            self.claims_df['emergency_visit'] = self.claims_df['emergency_visit'].astype(bool)
        
        try:
            self.claims_df['readmission_within_30_days'] = self.claims_df['readmission_within_30_days'].map({'True': True, 'False': False})
        except:
            self.claims_df['readmission_within_30_days'] = self.claims_df['readmission_within_30_days'].astype(bool)
        
        # Preventive care utilization rate
        total_members = self.claims_df['member_id'].nunique()
        preventive_care_members = self.claims_df[
            self.claims_df['preventive_care'] == True
        ]['member_id'].nunique()
        
        preventive_care_rate = (preventive_care_members / total_members) * 100 if total_members > 0 else 0
        
        
        # Emergency visit rate - count claims with emergency visits
        emergency_visits = self.claims_df['emergency_visit'].sum()
        total_claims = len(self.claims_df)
        emergency_visit_rate = (emergency_visits / total_claims) * 100 if total_claims > 0 else 0
        
        # Readmission rate - count claims with readmissions
        readmissions = self.claims_df['readmission_within_30_days'].sum()
        readmission_rate = (readmissions / total_claims) * 100 if total_claims > 0 else 0
        
        # Chronic condition management
        chronic_members = self.claims_df[
            self.claims_df['chronic_condition'] != 'None'
        ]['member_id'].nunique()
        chronic_condition_rate = (chronic_members / total_members) * 100 if total_members > 0 else 0
        
        return {
            'preventive_care_rate': round(preventive_care_rate, 2),
            'emergency_visit_rate': round(emergency_visit_rate, 2),
            'readmission_rate': round(readmission_rate, 2),
            'chronic_condition_rate': round(chronic_condition_rate, 2),
            'total_members': total_members,
            'total_claims': total_claims
        }
    
    def detect_anomalies(self) -> pd.DataFrame:
        """
        Detect potential fraud or anomalies in claims data.
        Implements pattern recognition for suspicious claims as required for government consulting.
        
        Returns:
            DataFrame containing potential anomalies
        """
        if self.claims_df is None:
            self.load_data()
        
        anomalies = []
        
        # High-cost claims (above 95th percentile)
        cost_threshold = np.percentile(self.claims_df['allowed_amount'], 95)
        high_cost_claims = self.claims_df[
            self.claims_df['allowed_amount'] > cost_threshold
        ].copy()
        high_cost_claims['anomaly_type'] = 'High Cost'
        anomalies.append(high_cost_claims)
        
        # Frequent emergency visits by same member
        member_emergency_counts = self.claims_df[
            self.claims_df['emergency_visit'] == True
        ]['member_id'].value_counts()
        
        frequent_emergency_members = member_emergency_counts[
            member_emergency_counts >= 3
        ].index
        
        frequent_emergency_claims = self.claims_df[
            (self.claims_df['member_id'].isin(frequent_emergency_members)) &
            (self.claims_df['emergency_visit'] == True)
        ].copy()
        frequent_emergency_claims['anomaly_type'] = 'Frequent Emergency Visits'
        anomalies.append(frequent_emergency_claims)
        
        # Claims with high allowed but low paid amounts (potential billing issues)
        billing_ratio = self.claims_df['paid_amount'] / self.claims_df['allowed_amount']
        billing_anomalies = self.claims_df[
            billing_ratio < 0.5
        ].copy()
        billing_anomalies['anomaly_type'] = 'Billing Discrepancy'
        anomalies.append(billing_anomalies)
        
        if anomalies:
            return pd.concat(anomalies, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def export_to_database(self, db_path: str = 'medicaid_data.db') -> None:
        """
        Export processed data to SQLite database.
        Demonstrates database management skills for government consulting.
        
        Args:
            db_path: Path to SQLite database file
        """
        if self.claims_df is None or self.providers_df is None:
            self.load_data()
        
        conn = sqlite3.connect(db_path)
        
        # Export tables
        self.claims_df.to_sql('claims', conn, if_exists='replace', index=False)
        self.providers_df.to_sql('providers', conn, if_exists='replace', index=False)
        
        # Export processed metrics
        pmpm_metrics = self.calculate_pmpm_metrics()
        pmpm_metrics['monthly_data'].to_sql('monthly_pmpm', conn, if_exists='replace', index=False)
        
        service_analysis = self.analyze_service_categories()
        service_analysis.to_sql('service_analysis', conn, if_exists='replace')
        
        provider_performance = self.analyze_provider_performance()
        provider_performance.to_sql('provider_performance', conn, if_exists='replace')
        
        conn.close()
        print(f"Data exported to {db_path}")
