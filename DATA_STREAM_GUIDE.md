# Data Stream Handler Guide

A flexible data stream handler for working with multiple data formats in Python. Designed for the claude-code-web-test2 testing environment.

## Features

- **Multiple Format Support**: CSV, JSON, JSON Lines, Parquet, Excel
- **Streaming Processing**: Handle large files efficiently with chunked reading
- **Format Conversion**: Easy conversion between different data formats
- **Data Validation**: Schema-based validation
- **Flexible Configuration**: JSON-based configuration system
- **Test Fixtures**: Pre-built fixtures for testing and development

## Quick Start

### Basic Usage

```python
from data_stream_handler import DataStreamHandler

# Initialize handler
handler = DataStreamHandler()

# Load data (auto-detects format from extension)
df = handler.load_data("data.csv")

# Load with explicit format
df = handler.load_data("data.json", format="json")

# Save data
handler.save_data(df, "output.csv")
```

### Format Conversion

```python
# Convert JSON to CSV
handler.convert_format(
    source="input.json",
    destination="output.csv"
)

# Convert CSV to Parquet
handler.convert_format(
    source="input.csv",
    destination="output.parquet"
)
```

### Streaming Large Files

```python
# Process large CSV in chunks
for chunk in handler.load_streaming("large_file.csv", chunk_size=10000):
    # Process each chunk
    processed = transform(chunk)
    # Save or aggregate results
```

## Directory Structure

```
claude-code-web-test2/
├── data_stream_handler.py       # Main handler module
├── test_data/                   # Sample test data
│   ├── sample_data.json         # JSON format example
│   ├── sample_metrics.csv       # CSV format example
│   └── streaming_data.jsonl     # JSON Lines format
├── config/                      # Configuration files
│   └── data_config.json         # Data source configurations
├── fixtures/                    # Test fixtures
│   └── test_fixtures.py         # Reusable test data generators
└── examples/                    # Usage examples
    └── usage_examples.py        # Comprehensive examples
```

## Supported Formats

### CSV (Comma-Separated Values)
- Standard tabular data format
- Configurable delimiters, encoding, and parsing options
- Streaming support for large files

### JSON (JavaScript Object Notation)
- Hierarchical data structures
- Supports multiple orientations (records, columns, etc.)
- Easy conversion to/from DataFrames

### JSON Lines (JSONL)
- Newline-delimited JSON
- Ideal for streaming data and logs
- Efficient for append-only operations

### Parquet
- Columnar storage format
- Compressed and efficient
- Great for large datasets

### Excel (XLSX, XLS)
- Spreadsheet formats
- Multiple sheet support
- Preserves formatting information

## Configuration System

The handler uses JSON-based configuration for data sources:

```json
{
  "data_sources": {
    "my_dataset": {
      "path": "data/my_data.csv",
      "format": "csv",
      "encoding": "utf-8",
      "description": "My dataset description"
    }
  },
  "processing": {
    "chunk_size": 10000,
    "default_encoding": "utf-8"
  }
}
```

Load from configuration:

```python
import json

with open('config/data_config.json') as f:
    config = json.load(f)

handler = DataStreamHandler(config=config['processing'])

for name, source in config['data_sources'].items():
    df = handler.load_data(source['path'], format=source['format'])
```

## Data Validation

Validate DataFrames against expected schemas:

```python
schema = {
    'id': 'int64',
    'name': 'object',
    'value': 'float64',
    'active': 'bool'
}

results = handler.validate_data(df, schema)

if not results['valid']:
    print("Validation errors:", results['errors'])
if results['warnings']:
    print("Warnings:", results['warnings'])
```

## Test Fixtures

Use pre-built fixtures for testing:

```python
from fixtures.test_fixtures import DataFixtures

# Create sample DataFrame
df = DataFixtures.create_sample_dataframe(rows=100)

# Create time series data
ts_df = DataFixtures.create_time_series(days=30, freq='1H')

# Create nested JSON structure
nested = DataFixtures.create_nested_json()
```

## Advanced Usage

### Custom Processing

```python
from data_stream_handler import DataStreamProcessor

processor = DataStreamProcessor()

def my_transform(df):
    # Your custom transformation
    df['new_column'] = df['value'] * 2
    return df

# Process in streaming mode
results = processor.process_streaming(
    source="large_file.csv",
    transform_func=my_transform,
    chunk_size=5000
)
```

### Aggregation on Streaming Data

```python
def aggregate_func(df):
    return df.groupby('category').agg({'value': 'sum'})

result = processor.aggregate_streaming(
    source="large_file.csv",
    agg_func=aggregate_func
)
```

## Examples

See `examples/usage_examples.py` for comprehensive examples:

```bash
python examples/usage_examples.py
```

Examples include:
1. Basic data loading from different formats
2. Format conversion
3. Streaming processing
4. Data validation
5. Custom transformations
6. Using test fixtures
7. Configuration-based loading

## Test Data

The repository includes sample test data:

- **boston_house_prices.csv**: Real estate pricing dataset
- **sample_data.json**: Generic JSON dataset
- **sample_metrics.csv**: Server metrics data
- **streaming_data.jsonl**: Event stream data

## Integration with Existing App

The data stream handler integrates seamlessly with the existing Streamlit app (`app.py`). You can extend the app to:

1. Support multiple data sources
2. Allow format conversion through UI
3. Enable large file processing
4. Provide data validation feedback

Example integration:

```python
from data_stream_handler import DataStreamHandler

handler = DataStreamHandler()

# In Streamlit app
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file:
    format = st.selectbox("Format", handler.SUPPORTED_FORMATS)
    df = handler.load_data(uploaded_file, format=format)
    st.dataframe(df)
```

## Best Practices

1. **Auto-detection**: Let the handler auto-detect formats when possible
2. **Streaming**: Use streaming for files larger than available RAM
3. **Validation**: Always validate data from external sources
4. **Configuration**: Use config files for production data sources
5. **Error Handling**: Wrap operations in try-except blocks

## Troubleshooting

### Format Detection Fails
```python
# Explicitly specify format
df = handler.load_data("data.txt", format="csv")
```

### Encoding Issues
```python
# Specify encoding
df = handler.load_data("data.csv", encoding="latin-1")
```

### Large File Memory Issues
```python
# Use streaming
for chunk in handler.load_streaming("large.csv", chunk_size=1000):
    process(chunk)
```

## Requirements

See `requirements.txt` for dependencies:
- pandas
- numpy
- pyarrow (for Parquet support)
- openpyxl (for Excel support)

## Contributing

When adding new features:
1. Update `data_stream_handler.py` with new functionality
2. Add corresponding examples to `examples/usage_examples.py`
3. Update this guide with documentation
4. Add test data if needed
5. Update `config/data_config.json` if adding new sources

## License

This is a test repository for Claude Code web functionality.
