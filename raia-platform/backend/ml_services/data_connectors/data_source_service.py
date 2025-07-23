# Data Source Connection and Management Service
import os
import json
import uuid
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Text, Boolean, JSON, UUID as PG_UUID, Integer, create_engine, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import SQLAlchemyError
import logging

# Connection drivers - conditional imports
try:
    import psycopg2
    from sqlalchemy.dialects import postgresql
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False

try:
    import pymysql
    from sqlalchemy.dialects import mysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import pymongo
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

try:
    import snowflake.connector
    from snowflake.sqlalchemy import URL as SNOWFLAKE_URL
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False

try:
    import boto3
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    from kafka import KafkaConsumer, KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

Base = declarative_base()
logger = logging.getLogger(__name__)

@dataclass
class ConnectionConfig:
    """Configuration for data source connections"""
    connector_type: str
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    schema: Optional[str] = None
    warehouse: Optional[str] = None  # For Snowflake
    account: Optional[str] = None    # For Snowflake
    bucket: Optional[str] = None     # For S3
    region: Optional[str] = None     # For AWS services
    topic: Optional[str] = None      # For Kafka
    collection: Optional[str] = None # For MongoDB
    additional_params: Optional[Dict[str, Any]] = None

@dataclass
class ConnectionTestResult:
    success: bool
    connection_time_ms: float
    message: str
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass
class DataPreview:
    columns: List[Dict[str, Any]]
    sample_data: List[Dict[str, Any]]
    row_count: Optional[int] = None
    data_types: Optional[Dict[str, str]] = None
    summary_stats: Optional[Dict[str, Any]] = None

@dataclass
class SchemaInfo:
    tables: List[str]
    views: List[str]
    schemas: List[str]
    table_details: Dict[str, Dict[str, Any]]

class DataSourceConnection(Base):
    """Store data source connection information"""
    __tablename__ = "data_source_connections"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    connection_id = Column(String(255), unique=True, nullable=False)
    name = Column(String(500), nullable=False)
    description = Column(Text)
    connector_type = Column(String(100), nullable=False)
    category = Column(String(100), nullable=False)  # database, cloud, file, api, streaming
    
    # Connection details (encrypted in production)
    connection_config = Column(JSON, nullable=False)
    
    # Connection status
    status = Column(String(50), default='inactive')  # active, inactive, error, testing
    last_connected = Column(DateTime)
    last_test_result = Column(JSON)
    
    # Schema and metadata
    schema_info = Column(JSON)
    supported_operations = Column(JSON)  # read, write, streaming, etc.
    
    # User and organization
    created_by = Column(String(255))
    organization_id = Column(String(255))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Access control
    is_shared = Column(Boolean, default=False)
    shared_with = Column(JSON)  # List of user IDs or team IDs

class DataSourceQuery(Base):
    """Store query history and results"""
    __tablename__ = "data_source_queries"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_id = Column(String(255), unique=True, nullable=False)
    connection_id = Column(String(255), nullable=False)
    
    # Query details
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50))  # select, insert, update, delete, schema
    
    # Execution results
    status = Column(String(50), nullable=False)  # success, failed, running
    execution_time_ms = Column(Integer)
    rows_affected = Column(Integer)
    result_preview = Column(JSON)  # First few rows for preview
    error_message = Column(Text)
    
    # Metadata
    executed_by = Column(String(255))
    executed_at = Column(DateTime, default=datetime.utcnow)

class DataSourceService:
    """Comprehensive service for managing data source connections"""
    
    def __init__(self, db_session: Session = None):
        self.db = db_session
        
        # Connection factories for different data source types
        self.connection_factories = {
            'postgresql': self._create_postgresql_connection,
            'mysql': self._create_mysql_connection,
            'mongodb': self._create_mongodb_connection,
            'snowflake': self._create_snowflake_connection,
            's3': self._create_s3_connection,
            'bigquery': self._create_bigquery_connection,
            'redshift': self._create_redshift_connection,
            'csv': self._create_csv_connection,
            'rest_api': self._create_rest_api_connection,
            'kafka': self._create_kafka_connection,
            'redis': self._create_redis_connection
        }

    async def create_connection(self, config: ConnectionConfig, 
                              created_by: str = None, 
                              organization_id: str = None,
                              name: str = None,
                              description: str = None) -> Dict[str, Any]:
        """Create and test a new data source connection"""
        
        connection_id = f"conn_{uuid.uuid4().hex}"
        
        # Test connection first
        test_result = await self.test_connection(config)
        
        if not test_result.success:
            return {
                'success': False,
                'error': f'Connection test failed: {test_result.error}',
                'connection_id': connection_id
            }
        
        # Create database record
        connection_record = DataSourceConnection(
            connection_id=connection_id,
            name=name or f"{config.connector_type} Connection",
            description=description,
            connector_type=config.connector_type,
            category=self._get_connector_category(config.connector_type),
            connection_config=self._sanitize_config_for_storage(asdict(config)),
            status='active',
            last_connected=datetime.utcnow(),
            last_test_result=asdict(test_result),
            created_by=created_by,
            organization_id=organization_id
        )
        
        if self.db:
            self.db.add(connection_record)
            self.db.commit()
        
        # Get schema information if applicable
        try:
            schema_info = await self.get_schema_info(connection_id, config)
            if schema_info and self.db:
                connection_record.schema_info = asdict(schema_info)
                self.db.commit()
        except Exception as e:
            logger.warning(f"Could not retrieve schema info: {e}")
        
        return {
            'success': True,
            'connection_id': connection_id,
            'test_result': test_result,
            'message': 'Connection created successfully'
        }

    async def test_connection(self, config: ConnectionConfig) -> ConnectionTestResult:
        """Test a data source connection"""
        
        start_time = datetime.utcnow()
        
        try:
            factory = self.connection_factories.get(config.connector_type)
            if not factory:
                return ConnectionTestResult(
                    success=False,
                    connection_time_ms=0,
                    message=f"Unsupported connector type: {config.connector_type}",
                    error=f"No factory available for {config.connector_type}"
                )
            
            # Test connection
            connection_result = await factory(config, test_only=True)
            
            end_time = datetime.utcnow()
            connection_time = (end_time - start_time).total_seconds() * 1000
            
            if connection_result.get('success', False):
                return ConnectionTestResult(
                    success=True,
                    connection_time_ms=connection_time,
                    message="Connection test successful",
                    metadata=connection_result.get('metadata')
                )
            else:
                return ConnectionTestResult(
                    success=False,
                    connection_time_ms=connection_time,
                    message="Connection test failed",
                    error=connection_result.get('error', 'Unknown error')
                )
                
        except Exception as e:
            end_time = datetime.utcnow()
            connection_time = (end_time - start_time).total_seconds() * 1000
            
            return ConnectionTestResult(
                success=False,
                connection_time_ms=connection_time,
                message="Connection test failed with exception",
                error=str(e)
            )

    async def get_data_preview(self, connection_id: str, 
                              table_name: str = None, 
                              query: str = None,
                              limit: int = 100) -> DataPreview:
        """Get a preview of data from the connection"""
        
        # Get connection config
        connection_record = None
        if self.db:
            connection_record = self.db.query(DataSourceConnection).filter(
                DataSourceConnection.connection_id == connection_id
            ).first()
        
        if not connection_record:
            raise ValueError(f"Connection {connection_id} not found")
        
        config = ConnectionConfig(**connection_record.connection_config)
        
        # Execute query or table select
        if query:
            result = await self._execute_query(config, query, limit)
        elif table_name:
            result = await self._execute_query(config, f"SELECT * FROM {table_name} LIMIT {limit}", limit)
        else:
            raise ValueError("Either table_name or query must be provided")
        
        if not result.get('success'):
            raise RuntimeError(f"Failed to get data preview: {result.get('error')}")
        
        df = result['data']
        
        # Generate preview data
        columns = [
            {
                'name': col,
                'type': str(df[col].dtype),
                'nullable': df[col].isnull().any(),
                'unique_values': df[col].nunique() if df[col].nunique() < 50 else None
            }
            for col in df.columns
        ]
        
        sample_data = df.head(limit).to_dict('records')
        
        # Generate summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        summary_stats = {}
        for col in numeric_cols:
            summary_stats[col] = {
                'mean': float(df[col].mean()) if not df[col].empty else None,
                'median': float(df[col].median()) if not df[col].empty else None,
                'min': float(df[col].min()) if not df[col].empty else None,
                'max': float(df[col].max()) if not df[col].empty else None,
                'std': float(df[col].std()) if not df[col].empty else None,
                'null_count': int(df[col].isnull().sum())
            }
        
        return DataPreview(
            columns=columns,
            sample_data=sample_data,
            row_count=len(df),
            data_types={col: str(df[col].dtype) for col in df.columns},
            summary_stats=summary_stats
        )

    async def get_schema_info(self, connection_id: str, config: ConnectionConfig = None) -> SchemaInfo:
        """Get schema information for a connection"""
        
        if not config:
            # Get connection config from database
            connection_record = None
            if self.db:
                connection_record = self.db.query(DataSourceConnection).filter(
                    DataSourceConnection.connection_id == connection_id
                ).first()
            
            if not connection_record:
                raise ValueError(f"Connection {connection_id} not found")
            
            config = ConnectionConfig(**connection_record.connection_config)
        
        # Get schema info based on connector type
        if config.connector_type in ['postgresql', 'mysql', 'snowflake', 'redshift']:
            return await self._get_sql_schema_info(config)
        elif config.connector_type == 'mongodb':
            return await self._get_mongodb_schema_info(config)
        else:
            # For other types, return empty schema
            return SchemaInfo(
                tables=[],
                views=[],
                schemas=[],
                table_details={}
            )

    async def execute_query(self, connection_id: str, query: str, 
                           executed_by: str = None) -> Dict[str, Any]:
        """Execute a query against a data source"""
        
        query_id = f"query_{uuid.uuid4().hex}"
        start_time = datetime.utcnow()
        
        # Get connection config
        connection_record = None
        if self.db:
            connection_record = self.db.query(DataSourceConnection).filter(
                DataSourceConnection.connection_id == connection_id
            ).first()
        
        if not connection_record:
            return {
                'success': False,
                'query_id': query_id,
                'error': f'Connection {connection_id} not found'
            }
        
        config = ConnectionConfig(**connection_record.connection_config)
        
        # Log query execution
        query_record = DataSourceQuery(
            query_id=query_id,
            connection_id=connection_id,
            query_text=query,
            query_type=self._detect_query_type(query),
            status='running',
            executed_by=executed_by
        )
        
        if self.db:
            self.db.add(query_record)
            self.db.commit()
        
        try:
            # Execute query
            result = await self._execute_query(config, query)
            
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            if result.get('success'):
                # Update query record
                if self.db:
                    query_record.status = 'success'
                    query_record.execution_time_ms = int(execution_time)
                    query_record.rows_affected = len(result['data']) if 'data' in result else 0
                    
                    # Store preview of results (first 10 rows)
                    if 'data' in result and isinstance(result['data'], pd.DataFrame):
                        preview_data = result['data'].head(10).to_dict('records')
                        query_record.result_preview = preview_data
                    
                    self.db.commit()
                
                return {
                    'success': True,
                    'query_id': query_id,
                    'execution_time_ms': execution_time,
                    'data': result['data'],
                    'row_count': len(result['data']) if 'data' in result else 0
                }
            else:
                # Update query record with error
                if self.db:
                    query_record.status = 'failed'
                    query_record.execution_time_ms = int(execution_time)
                    query_record.error_message = result.get('error', 'Unknown error')
                    self.db.commit()
                
                return {
                    'success': False,
                    'query_id': query_id,
                    'error': result.get('error', 'Query execution failed')
                }
                
        except Exception as e:
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            # Update query record with exception
            if self.db:
                query_record.status = 'failed'
                query_record.execution_time_ms = int(execution_time)
                query_record.error_message = str(e)
                self.db.commit()
            
            return {
                'success': False,
                'query_id': query_id,
                'error': str(e)
            }

    # Connection factory methods
    async def _create_postgresql_connection(self, config: ConnectionConfig, test_only: bool = False) -> Dict[str, Any]:
        """Create PostgreSQL connection"""
        
        if not POSTGRESQL_AVAILABLE:
            return {
                'success': False,
                'error': 'PostgreSQL driver not available. Install with: pip install psycopg2-binary'
            }
        
        try:
            connection_string = f"postgresql://{config.username}:{config.password}@{config.host}:{config.port or 5432}/{config.database}"
            
            engine = create_engine(connection_string)
            
            if test_only:
                # Just test the connection
                with engine.connect() as conn:
                    result = conn.execute("SELECT 1")
                    return {
                        'success': True,
                        'metadata': {
                            'server_version': 'PostgreSQL',
                            'database': config.database
                        }
                    }
            else:
                return {
                    'success': True,
                    'connection': engine
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def _create_mysql_connection(self, config: ConnectionConfig, test_only: bool = False) -> Dict[str, Any]:
        """Create MySQL connection"""
        
        if not MYSQL_AVAILABLE:
            return {
                'success': False,
                'error': 'MySQL driver not available. Install with: pip install pymysql'
            }
        
        try:
            connection_string = f"mysql+pymysql://{config.username}:{config.password}@{config.host}:{config.port or 3306}/{config.database}"
            
            engine = create_engine(connection_string)
            
            if test_only:
                with engine.connect() as conn:
                    result = conn.execute("SELECT 1")
                    return {
                        'success': True,
                        'metadata': {
                            'server_version': 'MySQL',
                            'database': config.database
                        }
                    }
            else:
                return {
                    'success': True,
                    'connection': engine
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def _create_mongodb_connection(self, config: ConnectionConfig, test_only: bool = False) -> Dict[str, Any]:
        """Create MongoDB connection"""
        
        if not MONGODB_AVAILABLE:
            return {
                'success': False,
                'error': 'MongoDB driver not available. Install with: pip install pymongo'
            }
        
        try:
            connection_string = f"mongodb://{config.username}:{config.password}@{config.host}:{config.port or 27017}/{config.database}"
            
            client = pymongo.MongoClient(connection_string)
            
            if test_only:
                # Test connection
                client.admin.command('ping')
                return {
                    'success': True,
                    'metadata': {
                        'server_version': 'MongoDB',
                        'database': config.database
                    }
                }
            else:
                return {
                    'success': True,
                    'connection': client
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def _create_snowflake_connection(self, config: ConnectionConfig, test_only: bool = False) -> Dict[str, Any]:
        """Create Snowflake connection"""
        
        if not SNOWFLAKE_AVAILABLE:
            return {
                'success': False,
                'error': 'Snowflake driver not available. Install with: pip install snowflake-connector-python'
            }
        
        try:
            connection_string = SNOWFLAKE_URL(
                account=config.account,
                user=config.username,
                password=config.password,
                database=config.database,
                warehouse=config.warehouse,
                schema=config.schema or 'PUBLIC'
            )
            
            engine = create_engine(connection_string)
            
            if test_only:
                with engine.connect() as conn:
                    result = conn.execute("SELECT CURRENT_VERSION()")
                    return {
                        'success': True,
                        'metadata': {
                            'server_version': 'Snowflake',
                            'database': config.database,
                            'warehouse': config.warehouse
                        }
                    }
            else:
                return {
                    'success': True,
                    'connection': engine
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def _create_s3_connection(self, config: ConnectionConfig, test_only: bool = False) -> Dict[str, Any]:
        """Create S3 connection"""
        
        if not AWS_AVAILABLE:
            return {
                'success': False,
                'error': 'AWS SDK not available. Install with: pip install boto3'
            }
        
        try:
            client = boto3.client(
                's3',
                region_name=config.region,
                aws_access_key_id=config.username,  # Using username field for access key
                aws_secret_access_key=config.password  # Using password field for secret key
            )
            
            if test_only:
                # List buckets to test connection
                response = client.list_buckets()
                return {
                    'success': True,
                    'metadata': {
                        'service': 'Amazon S3',
                        'region': config.region,
                        'bucket_count': len(response.get('Buckets', []))
                    }
                }
            else:
                return {
                    'success': True,
                    'connection': client
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def _create_csv_connection(self, config: ConnectionConfig, test_only: bool = False) -> Dict[str, Any]:
        """Create CSV file connection"""
        
        try:
            file_path = config.additional_params.get('file_path') if config.additional_params else None
            
            if not file_path:
                return {
                    'success': False,
                    'error': 'file_path is required for CSV connections'
                }
            
            if test_only:
                # Check if file exists and is readable
                if not os.path.exists(file_path):
                    return {
                        'success': False,
                        'error': f'File not found: {file_path}'
                    }
                
                # Try to read first few rows
                df = pd.read_csv(file_path, nrows=5)
                
                return {
                    'success': True,
                    'metadata': {
                        'file_type': 'CSV',
                        'columns': list(df.columns),
                        'column_count': len(df.columns)
                    }
                }
            else:
                return {
                    'success': True,
                    'connection': {'file_path': file_path, 'type': 'csv'}
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def _create_rest_api_connection(self, config: ConnectionConfig, test_only: bool = False) -> Dict[str, Any]:
        """Create REST API connection"""
        
        try:
            import requests
            
            base_url = f"http{'s' if config.port == 443 else ''}://{config.host}"
            if config.port and config.port not in [80, 443]:
                base_url += f":{config.port}"
            
            if test_only:
                # Make a simple GET request to test connectivity
                headers = {}
                if config.username and config.password:
                    # Basic auth
                    import base64
                    credentials = base64.b64encode(f"{config.username}:{config.password}".encode()).decode()
                    headers['Authorization'] = f'Basic {credentials}'
                
                response = requests.get(base_url, headers=headers, timeout=10)
                
                return {
                    'success': response.status_code < 400,
                    'metadata': {
                        'service': 'REST API',
                        'base_url': base_url,
                        'status_code': response.status_code
                    }
                }
            else:
                return {
                    'success': True,
                    'connection': {'base_url': base_url, 'type': 'rest_api'}
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def _create_kafka_connection(self, config: ConnectionConfig, test_only: bool = False) -> Dict[str, Any]:
        """Create Kafka connection"""
        
        if not KAFKA_AVAILABLE:
            return {
                'success': False,
                'error': 'Kafka client not available. Install with: pip install kafka-python'
            }
        
        try:
            bootstrap_servers = f"{config.host}:{config.port or 9092}"
            
            if test_only:
                # Create a simple consumer to test connection
                consumer = KafkaConsumer(
                    bootstrap_servers=[bootstrap_servers],
                    consumer_timeout_ms=5000,
                    api_version=(0, 10, 1)
                )
                
                # Get cluster metadata
                metadata = consumer.list_consumer_groups()
                
                consumer.close()
                
                return {
                    'success': True,
                    'metadata': {
                        'service': 'Apache Kafka',
                        'bootstrap_servers': bootstrap_servers
                    }
                }
            else:
                return {
                    'success': True,
                    'connection': {'bootstrap_servers': bootstrap_servers, 'type': 'kafka'}
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def _create_redis_connection(self, config: ConnectionConfig, test_only: bool = False) -> Dict[str, Any]:
        """Create Redis connection"""
        
        if not REDIS_AVAILABLE:
            return {
                'success': False,
                'error': 'Redis client not available. Install with: pip install redis'
            }
        
        try:
            r = redis.Redis(
                host=config.host,
                port=config.port or 6379,
                password=config.password,
                db=int(config.database) if config.database else 0
            )
            
            if test_only:
                # Ping Redis server
                response = r.ping()
                
                return {
                    'success': response,
                    'metadata': {
                        'service': 'Redis',
                        'host': config.host,
                        'port': config.port or 6379
                    }
                }
            else:
                return {
                    'success': True,
                    'connection': r
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def _create_bigquery_connection(self, config: ConnectionConfig, test_only: bool = False) -> Dict[str, Any]:
        """Create BigQuery connection"""
        
        # Placeholder for BigQuery implementation
        return {
            'success': False,
            'error': 'BigQuery connector not yet implemented'
        }

    async def _create_redshift_connection(self, config: ConnectionConfig, test_only: bool = False) -> Dict[str, Any]:
        """Create Redshift connection"""
        
        # Similar to PostgreSQL but with Redshift-specific connection string
        try:
            connection_string = f"redshift+psycopg2://{config.username}:{config.password}@{config.host}:{config.port or 5439}/{config.database}"
            
            engine = create_engine(connection_string)
            
            if test_only:
                with engine.connect() as conn:
                    result = conn.execute("SELECT 1")
                    return {
                        'success': True,
                        'metadata': {
                            'server_version': 'Amazon Redshift',
                            'database': config.database
                        }
                    }
            else:
                return {
                    'success': True,
                    'connection': engine
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    # Helper methods
    def _get_connector_category(self, connector_type: str) -> str:
        """Get category for connector type"""
        categories = {
            'postgresql': 'database',
            'mysql': 'database',
            'mongodb': 'database',
            'snowflake': 'cloud',
            'bigquery': 'cloud',
            'redshift': 'cloud',
            's3': 'cloud',
            'csv': 'file',
            'rest_api': 'api',
            'kafka': 'streaming',
            'redis': 'cache'
        }
        return categories.get(connector_type, 'other')

    def _sanitize_config_for_storage(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize connection config for storage (remove sensitive data)"""
        sanitized = config_dict.copy()
        # In production, encrypt sensitive fields
        # For demo, we'll mask the password
        if 'password' in sanitized and sanitized['password']:
            sanitized['password'] = '***masked***'
        return sanitized

    def _detect_query_type(self, query: str) -> str:
        """Detect the type of SQL query"""
        query_lower = query.lower().strip()
        
        if query_lower.startswith('select'):
            return 'select'
        elif query_lower.startswith('insert'):
            return 'insert'
        elif query_lower.startswith('update'):
            return 'update'
        elif query_lower.startswith('delete'):
            return 'delete'
        elif query_lower.startswith(('create', 'alter', 'drop')):
            return 'schema'
        else:
            return 'other'

    async def _execute_query(self, config: ConnectionConfig, query: str, limit: int = None) -> Dict[str, Any]:
        """Execute a query against the data source"""
        
        try:
            if config.connector_type in ['postgresql', 'mysql', 'snowflake', 'redshift']:
                return await self._execute_sql_query(config, query, limit)
            elif config.connector_type == 'mongodb':
                return await self._execute_mongodb_query(config, query, limit)
            elif config.connector_type == 'csv':
                return await self._execute_csv_query(config, query, limit)
            else:
                return {
                    'success': False,
                    'error': f'Query execution not supported for {config.connector_type}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def _execute_sql_query(self, config: ConnectionConfig, query: str, limit: int = None) -> Dict[str, Any]:
        """Execute SQL query"""
        
        connection_result = await self.connection_factories[config.connector_type](config)
        
        if not connection_result.get('success'):
            return connection_result
        
        engine = connection_result['connection']
        
        try:
            # Add limit if specified and query is a SELECT
            if limit and query.lower().strip().startswith('select'):
                query += f" LIMIT {limit}"
            
            df = pd.read_sql_query(query, engine)
            
            return {
                'success': True,
                'data': df
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            engine.dispose()

    async def _execute_mongodb_query(self, config: ConnectionConfig, query: str, limit: int = None) -> Dict[str, Any]:
        """Execute MongoDB query"""
        
        connection_result = await self._create_mongodb_connection(config)
        
        if not connection_result.get('success'):
            return connection_result
        
        client = connection_result['connection']
        
        try:
            # For demo purposes, assume query is a collection name
            # In production, would parse actual MongoDB query syntax
            collection_name = query.strip()
            
            db = client[config.database]
            collection = db[collection_name]
            
            cursor = collection.find()
            if limit:
                cursor = cursor.limit(limit)
            
            documents = list(cursor)
            df = pd.DataFrame(documents)
            
            return {
                'success': True,
                'data': df
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            client.close()

    async def _execute_csv_query(self, config: ConnectionConfig, query: str, limit: int = None) -> Dict[str, Any]:
        """Execute query on CSV file"""
        
        try:
            file_path = config.additional_params.get('file_path')
            
            if not file_path:
                return {
                    'success': False,
                    'error': 'file_path is required for CSV connections'
                }
            
            # For CSV, query is ignored - just read the file
            df = pd.read_csv(file_path, nrows=limit)
            
            return {
                'success': True,
                'data': df
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def _get_sql_schema_info(self, config: ConnectionConfig) -> SchemaInfo:
        """Get schema information for SQL databases"""
        
        connection_result = await self.connection_factories[config.connector_type](config)
        
        if not connection_result.get('success'):
            raise RuntimeError(f"Could not connect to database: {connection_result.get('error')}")
        
        engine = connection_result['connection']
        
        try:
            inspector = inspect(engine)
            
            # Get schemas
            schemas = inspector.get_schema_names() if hasattr(inspector, 'get_schema_names') else [config.schema or 'public']
            
            # Get tables and views
            tables = inspector.get_table_names(schema=config.schema)
            views = inspector.get_view_names(schema=config.schema) if hasattr(inspector, 'get_view_names') else []
            
            # Get table details
            table_details = {}
            for table in tables[:10]:  # Limit to first 10 tables for performance
                columns = inspector.get_columns(table, schema=config.schema)
                table_details[table] = {
                    'columns': [
                        {
                            'name': col['name'],
                            'type': str(col['type']),
                            'nullable': col.get('nullable', True),
                            'primary_key': col.get('primary_key', False)
                        }
                        for col in columns
                    ],
                    'row_count': None  # Would require separate query to get accurate count
                }
            
            return SchemaInfo(
                tables=tables,
                views=views,
                schemas=schemas,
                table_details=table_details
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to get schema info: {str(e)}")
        finally:
            engine.dispose()

    async def _get_mongodb_schema_info(self, config: ConnectionConfig) -> SchemaInfo:
        """Get schema information for MongoDB"""
        
        connection_result = await self._create_mongodb_connection(config)
        
        if not connection_result.get('success'):
            raise RuntimeError(f"Could not connect to MongoDB: {connection_result.get('error')}")
        
        client = connection_result['connection']
        
        try:
            db = client[config.database]
            collections = db.list_collection_names()
            
            # Get collection details (sample documents to infer schema)
            collection_details = {}
            for collection_name in collections[:10]:  # Limit to first 10 collections
                collection = db[collection_name]
                sample_doc = collection.find_one()
                
                if sample_doc:
                    fields = []
                    for key, value in sample_doc.items():
                        fields.append({
                            'name': key,
                            'type': type(value).__name__,
                            'nullable': True,
                            'primary_key': key == '_id'
                        })
                    
                    collection_details[collection_name] = {
                        'columns': fields,
                        'row_count': collection.estimated_document_count()
                    }
            
            return SchemaInfo(
                tables=collections,
                views=[],
                schemas=[config.database],
                table_details=collection_details
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to get MongoDB schema info: {str(e)}")
        finally:
            client.close()

    async def list_connections(self, user_id: str = None, 
                             organization_id: str = None,
                             connector_type: str = None) -> List[Dict[str, Any]]:
        """List all connections for a user or organization"""
        
        if not self.db:
            return []
        
        query = self.db.query(DataSourceConnection)
        
        if user_id:
            query = query.filter(DataSourceConnection.created_by == user_id)
        if organization_id:
            query = query.filter(DataSourceConnection.organization_id == organization_id)
        if connector_type:
            query = query.filter(DataSourceConnection.connector_type == connector_type)
        
        connections = query.order_by(DataSourceConnection.created_at.desc()).all()
        
        return [
            {
                'connection_id': conn.connection_id,
                'name': conn.name,
                'description': conn.description,
                'connector_type': conn.connector_type,
                'category': conn.category,
                'status': conn.status,
                'last_connected': conn.last_connected.isoformat() if conn.last_connected else None,
                'created_at': conn.created_at.isoformat(),
                'is_shared': conn.is_shared,
                'schema_available': bool(conn.schema_info)
            }
            for conn in connections
        ]

    async def delete_connection(self, connection_id: str) -> Dict[str, Any]:
        """Delete a data source connection"""
        
        if self.db:
            connection = self.db.query(DataSourceConnection).filter(
                DataSourceConnection.connection_id == connection_id
            ).first()
            
            if connection:
                # Delete related query records
                self.db.query(DataSourceQuery).filter(
                    DataSourceQuery.connection_id == connection_id
                ).delete()
                
                # Delete connection record
                self.db.delete(connection)
                self.db.commit()
                
                return {'success': True, 'message': 'Connection deleted successfully'}
        
        return {'success': False, 'message': 'Connection not found'}

# Factory function
def create_data_source_service(db_session: Session = None) -> DataSourceService:
    """Create and return a DataSourceService instance"""
    return DataSourceService(db_session)