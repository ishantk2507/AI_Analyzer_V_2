"""
DuckDB data layer for embedded SQL queries.

Handles dataset registration, connection setup with memory limits,
and column/predicate pushdown queries. Never loads full tables into pandas.
"""

import duckdb
from pathlib import Path
from typing import Optional, Dict, Any, List
import os

from app.config import DUCKDB_MEMORY_LIMIT


def _quote_identifier(name: str) -> str:
    """Quote a SQL identifier for DuckDB (handles names starting with digits)."""
    return f'"{name}"'


def _quote_literal(name: str) -> str:
    """Quote a SQL string literal for DuckDB."""
    return f"'{name}'"


class DataLayer:
    """
    Embedded DuckDB wrapper for local data analysis.
    
    Manages a single connection per session with explicit memory limits.
    Supports CSV, Parquet, and XLSX files via direct file paths or directory mounts.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize DuckDB connection with memory limit.
        
        Args:
            data_dir: Optional path to data directory for mounting.
                     If provided, creates a view 'data' pointing to it.
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.conn = duckdb.connect(':memory:')
        
        # Set memory limit explicitly
        self.conn.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}'")
        
        # Disable parallel scanning to reduce memory spikes
        self.conn.execute("PRAGMA threads=1")
        
        # Register the data directory if provided
        if self.data_dir and self.data_dir.exists():
            self._register_data_directory()
    
    def _register_data_directory(self):
        """Register all supported files in the data directory as views."""
        if not self.data_dir:
            return
        
        # Create a schema for user data
        self.conn.execute("CREATE SCHEMA IF NOT EXISTS user_data")
        
        # Find and register CSV files
        for csv_file in self.data_dir.glob('*.csv'):
            view_name = f"user_data.{csv_file.stem}"
            self.conn.execute(
                f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_csv_auto('{csv_file}')"
            )
        
        # Find and register Parquet files
        for parquet_file in self.data_dir.glob('*.parquet'):
            view_name = f"user_data.{parquet_file.stem}"
            self.conn.execute(
                f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{parquet_file}')"
            )
        
        # Find and register XLSX files (requires xlrd or openpyxl, use pandas bridge)
        for xlsx_file in self.data_dir.glob('*.xlsx'):
            view_name = f"user_data.{xlsx_file.stem}"
            try:
                # Use duckdb's built-in Excel reader if available, else pandas bridge
                self.conn.execute(
                    f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM st_read('{xlsx_file}')"
                )
            except Exception:
                # Fallback: try pandas + duckdb
                import pandas as pd
                df = pd.read_excel(xlsx_file)
                self.conn.register(view_name, df)
    
    def register_file(self, file_path: str, view_name: Optional[str] = None) -> str:
        """
        Register a single file as a view.
        
        Args:
            file_path: Full path to the file (CSV, Parquet, or XLSX).
            view_name: Optional custom view name. Defaults to filename stem.
        
        Returns:
            The view name created.
        """
        path = Path(file_path)
        if not view_name:
            view_name = path.stem
        
        ext = path.suffix.lower()
        if ext == '.csv':
            self.conn.execute(
                f"CREATE OR REPLACE VIEW user_data.{view_name} AS SELECT * FROM read_csv_auto('{path}')"
            )
        elif ext == '.parquet':
            self.conn.execute(
                f"CREATE OR REPLACE VIEW user_data.{view_name} AS SELECT * FROM read_parquet('{path}')"
            )
        elif ext == '.xlsx':
            import pandas as pd
            df = pd.read_excel(path)
            self.conn.register(f'user_data.{view_name}', df)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        return f'user_data.{view_name}'
    
    def get_table_names(self) -> List[str]:
        """Get list of registered table/view names."""
        result = self.conn.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'user_data'
        """).fetchall()
        return [row[0] for row in result]
    
    def get_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get schema info for a table.
        
        Returns list of dicts with keys: column_name, data_type, is_nullable.
        """
        # Use string literal for table_name comparison in WHERE clause
        quoted_table_literal = _quote_literal(table_name)
        result = self.conn.execute(f"""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = {quoted_table_literal}
            ORDER BY ordinal_position
        """).fetchall()
        return [
            {'column_name': row[0], 'data_type': row[1], 'is_nullable': row[2] == 'YES'}
            for row in result
        ]
    
    def get_profile(self, table_name: str) -> Dict[str, Any]:
        """
        Get a statistical profile of a table.
        
        Returns row counts, null counts, distinct counts, and sample values.
        Designed to be compact for LLM context (aggregated stats only).
        """
        schema = self.get_schema(table_name)
        profile = {
            'table_name': table_name,
            'row_count': 0,
            'columns': []
        }
        
        # Get row count
        quoted_table = _quote_identifier(table_name)
        row_result = self.conn.execute(f"SELECT COUNT(*) FROM user_data.{quoted_table}").fetchone()
        profile['row_count'] = row_result[0] if row_result else 0
        
        # Get per-column stats
        for col_info in schema:
            col_name = col_info['column_name']
            col_type = col_info['data_type']
            quoted_col = _quote_identifier(col_name)
            
            # Build dynamic query for stats
            stats_query = f"""
                SELECT 
                    COUNT(*) as total,
                    COUNT({quoted_col}) as non_null,
                    COUNT(DISTINCT {quoted_col}) as distinct_count
                FROM user_data.{quoted_table}
            """
            stats_result = self.conn.execute(stats_query).fetchone()
            
            col_profile = {
                'name': col_name,
                'type': col_type,
                'total_rows': stats_result[0],
                'non_null': stats_result[1],
                'null_count': stats_result[0] - stats_result[1] if stats_result[1] else 0,
                'distinct_count': stats_result[2] if stats_result[2] else 0,
                'sample_values': []
            }
            
            # Get sample values for ALL columns (not just low-cardinality ones)
            # This helps the analyst understand what each column contains
            sample_query = f"""
                SELECT DISTINCT {quoted_col} 
                FROM user_data.{quoted_table} 
                WHERE {quoted_col} IS NOT NULL
                LIMIT 5
            """
            try:
                samples = self.conn.execute(sample_query).fetchall()
                col_profile['sample_values'] = [str(s[0]) for s in samples]
            except Exception:
                pass
            
            profile['columns'].append(col_profile)
        
        return profile
    
    def execute_query(self, sql: str, params: Optional[Dict] = None) -> List[Dict]:
        """
        Execute a SQL query and return results as list of dicts.
        
        Uses column/predicate pushdown - caller should write efficient SQL.
        Results are limited to 1000 rows max for safety.
        
        Args:
            sql: SQL query string.
            params: Optional parameter dict for prepared statements.
        
        Returns:
            List of row dicts, max 1000 rows.
        """
        if params:
            result = self.conn.execute(sql, params).fetchdf()
        else:
            result = self.conn.execute(sql).fetchdf()
        
        # Limit rows for safety
        if len(result) > 1000:
            result = result.head(1000)
        
        return result.to_dict(orient='records')
    
    def close(self):
        """Close the DuckDB connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
