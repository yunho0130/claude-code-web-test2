"""
Test Fixtures for Data Stream Handler
Provides reusable test data and configurations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DataFixtures:
    """Collection of test data fixtures."""

    @staticmethod
    def create_sample_dataframe(rows: int = 100, seed: int = 42) -> pd.DataFrame:
        """
        Create a sample DataFrame for testing.

        Args:
            rows: Number of rows to generate
            seed: Random seed for reproducibility

        Returns:
            pd.DataFrame with sample data
        """
        np.random.seed(seed)

        return pd.DataFrame({
            'id': range(1, rows + 1),
            'value': np.random.uniform(0, 1000, rows),
            'category': np.random.choice(['A', 'B', 'C', 'D'], rows),
            'timestamp': [datetime.now() - timedelta(days=x) for x in range(rows)],
            'flag': np.random.choice([True, False], rows)
        })

    @staticmethod
    def create_time_series(days: int = 30, freq: str = '1H') -> pd.DataFrame:
        """
        Create time series data.

        Args:
            days: Number of days
            freq: Frequency string (e.g., '1H' for hourly)

        Returns:
            pd.DataFrame with time series data
        """
        date_range = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq=freq
        )

        return pd.DataFrame({
            'timestamp': date_range,
            'value': np.random.randn(len(date_range)).cumsum(),
            'metric': np.random.choice(['cpu', 'memory', 'disk'], len(date_range))
        })

    @staticmethod
    def create_nested_json() -> list:
        """
        Create nested JSON structure for testing.

        Returns:
            List of dictionaries with nested structure
        """
        return [
            {
                'id': 1,
                'user': {
                    'name': 'Alice',
                    'email': 'alice@example.com',
                    'age': 30
                },
                'purchases': [
                    {'item': 'laptop', 'price': 1200},
                    {'item': 'mouse', 'price': 25}
                ]
            },
            {
                'id': 2,
                'user': {
                    'name': 'Bob',
                    'email': 'bob@example.com',
                    'age': 25
                },
                'purchases': [
                    {'item': 'keyboard', 'price': 80},
                    {'item': 'monitor', 'price': 300}
                ]
            }
        ]

    @staticmethod
    def create_schema_example() -> dict:
        """
        Create example schema for validation.

        Returns:
            Dictionary representing data schema
        """
        return {
            'id': 'int64',
            'value': 'float64',
            'category': 'object',
            'timestamp': 'datetime64[ns]',
            'flag': 'bool'
        }


class ConfigFixtures:
    """Collection of configuration fixtures."""

    @staticmethod
    def get_default_config() -> dict:
        """
        Get default configuration.

        Returns:
            Dictionary with default configuration
        """
        return {
            'encoding': 'utf-8',
            'chunk_size': 10000,
            'date_format': '%Y-%m-%d',
            'decimal_separator': '.',
            'thousands_separator': ','
        }

    @staticmethod
    def get_csv_config() -> dict:
        """CSV-specific configuration."""
        return {
            'sep': ',',
            'encoding': 'utf-8',
            'na_values': ['NA', 'null', ''],
            'parse_dates': True
        }

    @staticmethod
    def get_json_config() -> dict:
        """JSON-specific configuration."""
        return {
            'orient': 'records',
            'date_format': 'iso',
            'indent': 2
        }
