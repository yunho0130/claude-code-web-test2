"""
Usage Examples for Data Stream Handler
Demonstrates various use cases and features
"""

from data_stream_handler import DataStreamHandler, DataStreamProcessor
from fixtures.test_fixtures import DataFixtures, ConfigFixtures
import pandas as pd


def example_1_basic_loading():
    """Example 1: Basic data loading from different formats."""
    print("=== Example 1: Basic Data Loading ===\n")

    handler = DataStreamHandler()

    print("Loading CSV file...")
    df_csv = handler.load_data("boston_house_prices.csv", skiprows=1)
    print(f"Loaded {len(df_csv)} rows from CSV")
    print(df_csv.head(3))
    print()

    print("Loading JSON file...")
    df_json = handler.load_data("test_data/sample_data.json")
    print(f"Loaded {len(df_json)} rows from JSON")
    print(df_json.head(3))
    print()

    print("Loading JSONL (streaming format)...")
    df_jsonl = handler.load_data("test_data/streaming_data.jsonl", format="jsonl")
    print(f"Loaded {len(df_jsonl)} rows from JSONL")
    print(df_jsonl.head(3))
    print()


def example_2_format_conversion():
    """Example 2: Converting between different formats."""
    print("=== Example 2: Format Conversion ===\n")

    handler = DataStreamHandler()

    print("Converting JSON to CSV...")
    handler.convert_format(
        source="test_data/sample_data.json",
        destination="test_data/sample_data_converted.csv",
        source_format="json",
        dest_format="csv"
    )
    print("Conversion complete: JSON -> CSV")
    print()

    print("Converting CSV to JSON...")
    handler.convert_format(
        source="test_data/sample_metrics.csv",
        destination="test_data/metrics_converted.json",
        source_format="csv",
        dest_format="json"
    )
    print("Conversion complete: CSV -> JSON")
    print()


def example_3_streaming_processing():
    """Example 3: Processing large files with streaming."""
    print("=== Example 3: Streaming Processing ===\n")

    handler = DataStreamHandler(config={'chunk_size': 100})

    print("Processing CSV in chunks...")
    chunk_count = 0
    for chunk in handler.load_streaming("boston_house_prices.csv", chunk_size=100):
        chunk_count += 1
        print(f"Processing chunk {chunk_count}: {len(chunk)} rows")
        if chunk_count >= 3:
            break
    print()


def example_4_data_validation():
    """Example 4: Validating data against schema."""
    print("=== Example 4: Data Validation ===\n")

    handler = DataStreamHandler()

    df = handler.load_data("test_data/sample_data.json")

    schema = {
        'id': 'int64',
        'name': 'object',
        'value': 'float64',
        'category': 'object',
        'active': 'bool'
    }

    print("Validating data against schema...")
    results = handler.validate_data(df, schema)

    print(f"Validation result: {'PASSED' if results['valid'] else 'FAILED'}")
    if results['errors']:
        print(f"Errors: {results['errors']}")
    if results['warnings']:
        print(f"Warnings: {results['warnings']}")
    print()


def example_5_custom_processing():
    """Example 5: Custom data processing with transformations."""
    print("=== Example 5: Custom Processing ===\n")

    processor = DataStreamProcessor()

    def transform_func(df):
        """Add computed columns."""
        df['value_squared'] = df['value'] ** 2
        df['category_upper'] = df['category'].str.upper()
        return df

    print("Applying custom transformation...")
    df = processor.handler.load_data("test_data/sample_data.json")
    transformed = transform_func(df)

    print("Original columns:", df.columns.tolist())
    print("Transformed columns:", transformed.columns.tolist())
    print(transformed.head(3))
    print()


def example_6_using_fixtures():
    """Example 6: Using test fixtures for development."""
    print("=== Example 6: Using Test Fixtures ===\n")

    print("Creating sample DataFrame...")
    df = DataFixtures.create_sample_dataframe(rows=10)
    print(df)
    print()

    print("Creating time series data...")
    ts_df = DataFixtures.create_time_series(days=7, freq='6H')
    print(ts_df.head())
    print()

    print("Creating nested JSON...")
    nested = DataFixtures.create_nested_json()
    print("Nested structure:", nested[0])
    print()


def example_7_configuration_based_loading():
    """Example 7: Loading data using configuration file."""
    print("=== Example 7: Configuration-Based Loading ===\n")

    import json

    with open('config/data_config.json', 'r') as f:
        config = json.load(f)

    handler = DataStreamHandler(config=config['processing'])

    for source_name, source_config in config['data_sources'].items():
        print(f"Loading {source_name}...")
        try:
            df = handler.load_data(
                source_config['path'],
                format=source_config['format']
            )
            print(f"  Loaded {len(df)} rows")
            print(f"  Description: {source_config['description']}")
        except Exception as e:
            print(f"  Error: {str(e)}")
        print()


def main():
    """Run all examples."""
    examples = [
        example_1_basic_loading,
        example_2_format_conversion,
        example_3_streaming_processing,
        example_4_data_validation,
        example_5_custom_processing,
        example_6_using_fixtures,
        example_7_configuration_based_loading
    ]

    print("\n" + "="*60)
    print("Data Stream Handler - Usage Examples")
    print("="*60 + "\n")

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {str(e)}\n")

    print("="*60)
    print("All examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
