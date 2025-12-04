"""
Flexible Data Stream Handler
Supports multiple data formats: CSV, JSON, Parquet, Excel, and streaming data
"""

import pandas as pd
import json
from pathlib import Path
from typing import Union, Optional, Dict, Any, List
from io import StringIO, BytesIO


class DataStreamHandler:
    """
    A flexible handler for multiple data stream formats.
    Supports CSV, JSON, Parquet, Excel, and provides streaming capabilities.
    """

    SUPPORTED_FORMATS = ['csv', 'json', 'parquet', 'xlsx', 'xls', 'jsonl']

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data stream handler.

        Args:
            config: Optional configuration dictionary for default parameters
        """
        self.config = config or {}
        self.default_encoding = self.config.get('encoding', 'utf-8')
        self.chunk_size = self.config.get('chunk_size', 10000)

    def load_data(self,
                  source: Union[str, Path, BytesIO, StringIO],
                  format: Optional[str] = None,
                  **kwargs) -> pd.DataFrame:
        """
        Load data from various sources and formats.

        Args:
            source: File path, URL, or file-like object
            format: Data format (csv, json, parquet, xlsx, etc.). Auto-detected if not provided
            **kwargs: Additional parameters passed to pandas read functions

        Returns:
            pd.DataFrame: Loaded data

        Raises:
            ValueError: If format is not supported or cannot be detected
        """
        if format is None:
            format = self._detect_format(source)

        format = format.lower()

        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}. Supported: {self.SUPPORTED_FORMATS}")

        loader_map = {
            'csv': self._load_csv,
            'json': self._load_json,
            'jsonl': self._load_jsonl,
            'parquet': self._load_parquet,
            'xlsx': self._load_excel,
            'xls': self._load_excel
        }

        return loader_map[format](source, **kwargs)

    def load_streaming(self,
                      source: Union[str, Path],
                      format: Optional[str] = None,
                      chunk_size: Optional[int] = None) -> pd.io.parsers.TextFileReader:
        """
        Load data in streaming mode (for large files).

        Args:
            source: File path or URL
            format: Data format (currently supports CSV)
            chunk_size: Number of rows per chunk

        Returns:
            Iterator yielding DataFrame chunks
        """
        if format is None:
            format = self._detect_format(source)

        chunk_size = chunk_size or self.chunk_size

        if format == 'csv':
            return pd.read_csv(source, chunksize=chunk_size, encoding=self.default_encoding)
        else:
            raise ValueError(f"Streaming not yet supported for format: {format}")

    def save_data(self,
                  df: pd.DataFrame,
                  destination: Union[str, Path],
                  format: Optional[str] = None,
                  **kwargs) -> None:
        """
        Save DataFrame to various formats.

        Args:
            df: DataFrame to save
            destination: Output file path
            format: Output format. Auto-detected from extension if not provided
            **kwargs: Additional parameters passed to pandas write functions
        """
        if format is None:
            format = self._detect_format(destination)

        format = format.lower()

        saver_map = {
            'csv': lambda: df.to_csv(destination, index=False, **kwargs),
            'json': lambda: df.to_json(destination, orient='records', **kwargs),
            'parquet': lambda: df.to_parquet(destination, **kwargs),
            'xlsx': lambda: df.to_excel(destination, index=False, **kwargs)
        }

        if format not in saver_map:
            raise ValueError(f"Unsupported output format: {format}")

        saver_map[format]()

    def convert_format(self,
                      source: Union[str, Path],
                      destination: Union[str, Path],
                      source_format: Optional[str] = None,
                      dest_format: Optional[str] = None,
                      **kwargs) -> None:
        """
        Convert data from one format to another.

        Args:
            source: Input file path
            destination: Output file path
            source_format: Source data format
            dest_format: Destination data format
            **kwargs: Additional parameters
        """
        df = self.load_data(source, format=source_format)
        self.save_data(df, destination, format=dest_format, **kwargs)

    def validate_data(self, df: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate DataFrame against a schema.

        Args:
            df: DataFrame to validate
            schema: Schema dictionary with column names and expected types

        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        for col, expected_type in schema.items():
            if col not in df.columns:
                results['valid'] = False
                results['errors'].append(f"Missing column: {col}")
            elif df[col].dtype != expected_type:
                results['warnings'].append(
                    f"Column {col} type mismatch: expected {expected_type}, got {df[col].dtype}"
                )

        extra_cols = set(df.columns) - set(schema.keys())
        if extra_cols:
            results['warnings'].append(f"Extra columns found: {extra_cols}")

        return results

    def _detect_format(self, source: Union[str, Path, BytesIO, StringIO]) -> str:
        """Detect format from file extension."""
        if isinstance(source, (BytesIO, StringIO)):
            raise ValueError("Cannot auto-detect format for file-like objects. Please specify format.")

        path = Path(source)
        ext = path.suffix.lstrip('.')

        if not ext:
            raise ValueError("Cannot detect format: no file extension found")

        return ext

    def _load_csv(self, source, **kwargs) -> pd.DataFrame:
        """Load CSV file."""
        encoding = kwargs.pop('encoding', self.default_encoding)
        return pd.read_csv(source, encoding=encoding, **kwargs)

    def _load_json(self, source, **kwargs) -> pd.DataFrame:
        """Load JSON file."""
        return pd.read_json(source, **kwargs)

    def _load_jsonl(self, source, **kwargs) -> pd.DataFrame:
        """Load JSON Lines file."""
        return pd.read_json(source, lines=True, **kwargs)

    def _load_parquet(self, source, **kwargs) -> pd.DataFrame:
        """Load Parquet file."""
        return pd.read_parquet(source, **kwargs)

    def _load_excel(self, source, **kwargs) -> pd.DataFrame:
        """Load Excel file."""
        return pd.read_excel(source, **kwargs)


class DataStreamProcessor:
    """
    Process data streams with transformations and aggregations.
    """

    def __init__(self, handler: Optional[DataStreamHandler] = None):
        """
        Initialize processor with a data stream handler.

        Args:
            handler: DataStreamHandler instance
        """
        self.handler = handler or DataStreamHandler()

    def process_streaming(self,
                         source: Union[str, Path],
                         transform_func: callable,
                         format: Optional[str] = None,
                         chunk_size: Optional[int] = None) -> List[pd.DataFrame]:
        """
        Process large files in streaming mode with transformation.

        Args:
            source: Input file path
            transform_func: Function to apply to each chunk
            format: Data format
            chunk_size: Rows per chunk

        Returns:
            List of processed DataFrame chunks
        """
        results = []

        for chunk in self.handler.load_streaming(source, format, chunk_size):
            processed = transform_func(chunk)
            results.append(processed)

        return results

    def aggregate_streaming(self,
                          source: Union[str, Path],
                          agg_func: callable,
                          format: Optional[str] = None,
                          chunk_size: Optional[int] = None) -> Any:
        """
        Aggregate data from large files in streaming mode.

        Args:
            source: Input file path
            agg_func: Aggregation function
            format: Data format
            chunk_size: Rows per chunk

        Returns:
            Aggregated result
        """
        accumulator = None

        for chunk in self.handler.load_streaming(source, format, chunk_size):
            if accumulator is None:
                accumulator = agg_func(chunk)
            else:
                accumulator = agg_func(pd.concat([accumulator, chunk]))

        return accumulator
