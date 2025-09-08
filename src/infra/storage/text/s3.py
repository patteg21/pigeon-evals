import json
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Optional, List, Dict, Any

from .base import TextStorageBase, TextStorageError
from src.models.documents import DocumentChunk
from models.configs.storage import S3Config


class S3Error(TextStorageError):
    """S3-specific exception for operations"""
    pass


class S3Storage(TextStorageBase):
    def __init__(self, config: S3Config):
        """Initialize S3 client with configuration"""
        super().__init__(config)
        self.bucket_name = self.config.bucket_name or 'pigeon-evals-documents'
        self.prefix = self.config.prefix or 'documents/'
        
        try:
            self.client = boto3.client(
                's3',
                aws_access_key_id=self.config.get('access_key_id'),
                aws_secret_access_key=self.config.get('secret_access_key'),
                region_name=self.config.get('region', 'us-east-1')
            )
            self._ensure_bucket_exists()
        except (NoCredentialsError, Exception) as e:
            raise S3Error(f"Failed to initialize S3 client: {str(e)}")
    
    @property
    def provider_name(self) -> str:
        return "s3"

    def _ensure_bucket_exists(self):
        """Ensure the bucket exists, create if it doesn't"""
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                try:
                    self.client.create_bucket(Bucket=self.bucket_name)
                except ClientError as create_error:
                    raise S3Error(f"Failed to create bucket {self.bucket_name}: {str(create_error)}")
            else:
                raise S3Error(f"Failed to access bucket {self.bucket_name}: {str(e)}")

    def _get_object_key(self, doc_id: str) -> str:
        """Generate S3 object key for document ID"""
        return f"{self.prefix}{doc_id}.json"

    def store_document(self, doc_id: str, doc_data: Dict[str, Any]) -> bool:
        """Store document data in S3"""
        try:
            key = self._get_object_key(doc_id)
            document = {
                'id': doc_id,
                'text': doc_data.get('text'),
                'document_data': doc_data.get('document_data'),
                'embedding': doc_data.get('embedding'),
                'created_at': doc_data.get('created_at')
            }
            
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json.dumps(document),
                ContentType='application/json'
            )
            return True
        except Exception as e:
            raise S3Error(f"Failed to store document {doc_id}: {str(e)}")

    def store_document_chunk(self, chunk: DocumentChunk) -> bool:
        """Store DocumentChunk in S3"""
        try:
            key = self._get_object_key(chunk.id)
            document = {
                'id': chunk.id,
                'text': chunk.text,
                'document': {
                    'id': chunk.document.id,
                    'name': chunk.document.name,
                    'path': chunk.document.path,
                    'text': chunk.document.text
                },
                'embedding': chunk.embeddding
            }
            
            self.client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json.dumps(document),
                ContentType='application/json'
            )
            return True
        except Exception as e:
            raise S3Error(f"Failed to store document chunk {chunk.id}: {str(e)}")

    def retrieve_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document by ID"""
        try:
            key = self._get_object_key(doc_id)
            response = self.client.get_object(Bucket=self.bucket_name, Key=key)
            content = response['Body'].read().decode('utf-8')
            return json.loads(content)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return None
            raise S3Error(f"Failed to retrieve document {doc_id}: {str(e)}")
        except Exception as e:
            raise S3Error(f"Failed to retrieve document {doc_id}: {str(e)}")

    def retrieve_documents(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve multiple documents by IDs"""
        if not doc_ids:
            return []
            
        documents = []
        for doc_id in doc_ids:
            doc = self.retrieve_document(doc_id)
            if doc:
                documents.append(doc)
        return documents

    def delete_document(self, doc_id: str) -> bool:
        """Delete document by ID"""
        try:
            key = self._get_object_key(doc_id)
            self.client.delete_object(Bucket=self.bucket_name, Key=key)
            return True
        except Exception as e:
            raise S3Error(f"Failed to delete document {doc_id}: {str(e)}")

    def get_document_count(self) -> int:
        """Get total number of documents"""
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.prefix
            )
            return response.get('KeyCount', 0)
        except Exception as e:
            raise S3Error(f"Failed to get document count: {str(e)}")

    def clear_all(self) -> bool:
        """Clear all documents from S3 bucket"""
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.prefix
            )
            
            if 'Contents' in response:
                objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
                
                if objects_to_delete:
                    self.client.delete_objects(
                        Bucket=self.bucket_name,
                        Delete={'Objects': objects_to_delete}
                    )
            return True
        except Exception as e:
            raise S3Error(f"Failed to clear documents: {str(e)}")