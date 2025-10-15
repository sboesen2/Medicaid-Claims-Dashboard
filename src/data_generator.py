"""
Synthetic Medicaid Claims Data Generator

This module generates realistic synthetic Medicaid claims data for demonstration
and testing purposes. It creates various scenarios including normal operations,
high fraud scenarios, and geographic variations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random

class MedicaidDataGenerator:
    """
    Generates synthetic Medicaid claims data with various scenarios and patterns.
    Demonstrates data engineering skills for government health consulting.
    """
    
    def __init__(self):
        """Initialize the data generator with base parameters."""
        self.provider_ids = [f"PROV{i:03d}" for i in range(1, 26)]
        self.service_types = ['Outpatient', 'Emergency', 'Inpatient', 'Preventive']
        self.diagnosis_codes = ['250.00', '786.50', '414.01', '272.40', '428.0', '789.01', 'Z00.00']
        self.chronic_conditions = ['None', 'Diabetes', 'Heart Disease', 'High Cholesterol', 'Heart Failure']
        self.counties = ['Maricopa', 'Pima', 'Pinal', 'Yavapai', 'Coconino', 'Mohave', 'Cochise', 'Navajo', 'Apache', 'Gila', 'Santa Cruz', 'Graham', 'Greenlee', 'La Paz']
        self.genders = ['M', 'F']
        
        # Provider specialties and types
        self.provider_specialties = {
            'Primary Care': ['Internal Medicine', 'Family Medicine', 'Preventive Medicine'],
            'Emergency': ['Emergency Medicine'],
            'Specialist': ['Cardiology', 'Endocrinology', 'Pulmonology'],
            'Urgent Care': ['Urgent Care']
        }
        
        # Base cost ranges by service type
        self.cost_ranges = {
            'Outpatient': (100, 300),
            'Emergency': (300, 600),
            'Inpatient': (2000, 4000),
            'Preventive': (80, 150)
        }
    
    def generate_claims_data(self, 
                           num_claims: int = 500,
                           scenario: str = 'normal',
                           start_date: str = '2024-01-01',
                           end_date: str = '2024-03-31') -> pd.DataFrame:
        """
        Generate synthetic Medicaid claims data.
        
        Args:
            num_claims: Number of claims to generate
            scenario: Data scenario ('normal', 'high_fraud', 'rural', 'urban', 'seasonal')
            start_date: Start date for claims
            end_date: End date for claims
            
        Returns:
            DataFrame with synthetic claims data
        """
        # Generate base claims
        claims = []
        
        # Generate member pool
        num_members = max(50, num_claims // 3)  # Ensure reasonable member-to-claim ratio
        member_pool = self._generate_member_pool(num_members)
        
        # Generate claims based on scenario
        if scenario == 'normal':
            claims = self._generate_normal_claims(num_claims, member_pool, start_date, end_date)
        elif scenario == 'high_fraud':
            claims = self._generate_high_fraud_claims(num_claims, member_pool, start_date, end_date)
        elif scenario == 'rural':
            claims = self._generate_rural_claims(num_claims, member_pool, start_date, end_date)
        elif scenario == 'urban':
            claims = self._generate_urban_claims(num_claims, member_pool, start_date, end_date)
        elif scenario == 'seasonal':
            claims = self._generate_seasonal_claims(num_claims, member_pool, start_date, end_date)
        else:
            claims = self._generate_normal_claims(num_claims, member_pool, start_date, end_date)
        
        return pd.DataFrame(claims)
    
    def generate_provider_data(self) -> pd.DataFrame:
        """
        Generate synthetic provider network data.
        
        Returns:
            DataFrame with provider information
        """
        providers = []
        
        for i, provider_id in enumerate(self.provider_ids):
            # Determine provider type and specialty
            provider_type = random.choice(list(self.provider_specialties.keys()))
            specialty = random.choice(self.provider_specialties[provider_type])
            
            # Generate provider characteristics
            provider = {
                'provider_id': provider_id,
                'provider_name': f"{specialty} Center {i+1}",
                'provider_type': provider_type,
                'address': f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'Pine', 'Elm'])} St",
                'city': random.choice(['Phoenix', 'Tucson', 'Mesa', 'Chandler', 'Scottsdale']),
                'state': 'AZ',
                'zip_code': f"85{random.randint(100, 999)}",
                'county': random.choice(self.counties),
                'specialty': specialty,
                'network_status': 'In-Network',
                'quality_score': round(np.random.normal(4.0, 0.5), 1),
                'patient_volume': random.randint(200, 1500),
                'avg_cost_per_visit': round(np.random.normal(200, 50), 2)
            }
            
            # Ensure quality score is within valid range
            provider['quality_score'] = max(1.0, min(5.0, provider['quality_score']))
            
            providers.append(provider)
        
        return pd.DataFrame(providers)
    
    def _generate_member_pool(self, num_members: int) -> List[Dict]:
        """Generate a pool of members for claims generation."""
        members = []
        
        for i in range(num_members):
            member = {
                'member_id': f"MEM{i+1:03d}",
                'age': random.randint(18, 85),
                'gender': random.choice(self.genders),
                'county': random.choice(self.counties),
                'chronic_condition': random.choices(
                    self.chronic_conditions, 
                    weights=[0.4, 0.2, 0.15, 0.15, 0.1]
                )[0]
            }
            members.append(member)
        
        return members
    
    def _generate_normal_claims(self, num_claims: int, member_pool: List[Dict], 
                               start_date: str, end_date: str) -> List[Dict]:
        """Generate normal operational claims."""
        claims = []
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        for i in range(num_claims):
            member = random.choice(member_pool)
            service_type = random.choices(
                self.service_types,
                weights=[0.4, 0.2, 0.1, 0.3]  # More outpatient and preventive
            )[0]
            
            claim = self._create_base_claim(i, member, service_type, start_dt, end_dt)
            claims.append(claim)
        
        return claims
    
    def _generate_high_fraud_claims(self, num_claims: int, member_pool: List[Dict],
                                   start_date: str, end_date: str) -> List[Dict]:
        """Generate claims with higher fraud patterns."""
        claims = []
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Increase fraud indicators
        fraud_weights = [0.2, 0.3, 0.2, 0.3]  # More emergency and inpatient
        
        for i in range(num_claims):
            member = random.choice(member_pool)
            service_type = random.choices(self.service_types, weights=fraud_weights)[0]
            
            claim = self._create_base_claim(i, member, service_type, start_dt, end_dt)
            
            # Add fraud patterns
            if random.random() < 0.15:  # 15% chance of high-cost anomaly
                claim['allowed_amount'] *= random.uniform(2.0, 4.0)
            
            if random.random() < 0.10:  # 10% chance of billing discrepancy
                claim['paid_amount'] = claim['allowed_amount'] * random.uniform(0.3, 0.7)
            
            if random.random() < 0.20:  # 20% chance of frequent emergency visits
                claim['emergency_visit'] = True
            
            claims.append(claim)
        
        return claims
    
    def _generate_rural_claims(self, num_claims: int, member_pool: List[Dict],
                              start_date: str, end_date: str) -> List[Dict]:
        """Generate claims with rural patterns (fewer providers, higher costs)."""
        claims = []
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Rural areas have fewer providers and higher costs
        rural_providers = self.provider_ids[:10]  # Fewer providers
        rural_counties = ['Yavapai', 'Coconino']  # Rural counties
        
        for i in range(num_claims):
            member = random.choice(member_pool)
            if member['county'] not in rural_counties:
                member['county'] = random.choice(rural_counties)
            
            service_type = random.choices(
                self.service_types,
                weights=[0.3, 0.3, 0.2, 0.2]  # More emergency due to distance
            )[0]
            
            claim = self._create_base_claim(i, member, service_type, start_dt, end_dt)
            claim['provider_id'] = random.choice(rural_providers)
            
            # Rural areas typically have higher costs
            claim['allowed_amount'] *= random.uniform(1.2, 1.8)
            claim['paid_amount'] *= random.uniform(1.2, 1.8)
            
            claims.append(claim)
        
        return claims
    
    def _generate_urban_claims(self, num_claims: int, member_pool: List[Dict],
                              start_date: str, end_date: str) -> List[Dict]:
        """Generate claims with urban patterns (more providers, lower costs)."""
        claims = []
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Urban areas have more providers and competitive pricing
        urban_counties = ['Maricopa', 'Pima']  # Urban counties
        
        for i in range(num_claims):
            member = random.choice(member_pool)
            if member['county'] not in urban_counties:
                member['county'] = random.choice(urban_counties)
            
            service_type = random.choices(
                self.service_types,
                weights=[0.5, 0.15, 0.1, 0.25]  # More outpatient and preventive
            )[0]
            
            claim = self._create_base_claim(i, member, service_type, start_dt, end_dt)
            
            # Urban areas typically have more competitive pricing
            claim['allowed_amount'] *= random.uniform(0.8, 1.2)
            claim['paid_amount'] *= random.uniform(0.8, 1.2)
            
            claims.append(claim)
        
        return claims
    
    def _generate_seasonal_claims(self, num_claims: int, member_pool: List[Dict],
                                 start_date: str, end_date: str) -> List[Dict]:
        """Generate claims with seasonal patterns."""
        claims = []
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        for i in range(num_claims):
            member = random.choice(member_pool)
            service_type = random.choice(self.service_types)
            
            claim = self._create_base_claim(i, member, service_type, start_dt, end_dt)
            
            # Add seasonal patterns
            service_date = claim['service_date']
            month = service_date.month
            
            # Winter months (Dec, Jan, Feb) have more emergency visits
            if month in [12, 1, 2]:
                if service_type == 'Emergency':
                    claim['allowed_amount'] *= random.uniform(1.1, 1.3)
                if random.random() < 0.3:
                    claim['emergency_visit'] = True
            
            # Summer months (Jun, Jul, Aug) have more preventive care
            elif month in [6, 7, 8]:
                if service_type == 'Preventive':
                    claim['allowed_amount'] *= random.uniform(0.9, 1.1)
                if random.random() < 0.4:
                    claim['preventive_care'] = True
            
            claims.append(claim)
        
        return claims
    
    def _create_base_claim(self, claim_id: int, member: Dict, service_type: str,
                          start_dt: datetime, end_dt: datetime) -> Dict:
        """Create a base claim with common attributes."""
        # Generate random date within range
        time_between = end_dt - start_dt
        days_between = time_between.days
        random_days = random.randint(0, days_between)
        service_date = start_dt + timedelta(days=random_days)
        
        # Generate costs based on service type with more variation
        cost_range = self.cost_ranges[service_type]
        base_amount = random.uniform(cost_range[0], cost_range[1])
        
        # Add cost modifiers for different scenarios
        if service_type == 'Preventive':
            # Preventive care is typically cheaper
            allowed_amount = base_amount * random.uniform(0.8, 1.2)
        elif service_type == 'Emergency':
            # Emergency care is more expensive
            allowed_amount = base_amount * random.uniform(1.2, 2.0)
        elif service_type == 'Inpatient':
            # Inpatient is most expensive
            allowed_amount = base_amount * random.uniform(1.5, 2.5)
        else:  # Outpatient
            # Standard outpatient costs
            allowed_amount = base_amount * random.uniform(0.9, 1.3)
        
        paid_amount = allowed_amount * random.uniform(0.7, 0.95)  # Insurance typically pays 70-95%
        
        # Generate boolean flags with more realistic patterns
        emergency_visit = service_type == 'Emergency'
        preventive_care = service_type == 'Preventive'
        
        # Add some additional preventive care visits (not just service type)
        if not preventive_care and random.random() < 0.30:  # 30% chance of additional preventive care
            preventive_care = True
        
        # Add some emergency visits for non-emergency service types (urgent care, etc.)
        if not emergency_visit and service_type == 'Outpatient' and random.random() < 0.15:  # 15% chance
            emergency_visit = True
        
        # Add emergency visits for other service types too
        if not emergency_visit and service_type in ['Inpatient', 'Preventive'] and random.random() < 0.05:  # 5% chance
            emergency_visit = True
        
        readmission = random.random() < 0.08  # 8% readmission rate (more realistic)
        
        claim = {
            'claim_id': f"CLM{claim_id+1:03d}",
            'member_id': member['member_id'],
            'provider_id': random.choice(self.provider_ids),
            'service_date': service_date,
            'service_type': service_type,
            'diagnosis_code': random.choice(self.diagnosis_codes),
            'procedure_code': f"99{random.randint(200, 300)}",
            'allowed_amount': round(allowed_amount, 2),
            'paid_amount': round(paid_amount, 2),
            'member_age': member['age'],
            'member_gender': member['gender'],
            'member_zip': f"85{random.randint(100, 999)}",
            'county': member['county'],
            'chronic_condition': member['chronic_condition'],
            'preventive_care': str(preventive_care),
            'emergency_visit': str(emergency_visit),
            'readmission_within_30_days': str(readmission)
        }
        
        return claim
    
    def export_dataset(self, claims_df: pd.DataFrame, providers_df: pd.DataFrame,
                      filename_prefix: str = "medicaid_data") -> Dict[str, str]:
        """
        Export generated datasets to CSV files.
        
        Args:
            claims_df: Claims DataFrame
            providers_df: Providers DataFrame
            filename_prefix: Prefix for exported files
            
        Returns:
            Dictionary with file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        claims_file = f"data/{filename_prefix}_claims_{timestamp}.csv"
        providers_file = f"data/{filename_prefix}_providers_{timestamp}.csv"
        
        claims_df.to_csv(claims_file, index=False)
        providers_df.to_csv(providers_file, index=False)
        
        return {
            'claims_file': claims_file,
            'providers_file': providers_file
        }
