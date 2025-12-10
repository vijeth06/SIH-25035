"""
Data ingestion service for processing uploaded files and extracting comments.
Supports CSV, Excel, and text files with deduplication and validation.
MongoDB-compatible version.
"""

import pandas as pd
import aiofiles
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib
import json
from io import BytesIO, StringIO
import re
from motor.motor_asyncio import AsyncIOMotorDatabase
from fastapi import UploadFile, HTTPException

from backend.app.core.config import settings
from backend.app.models.mongo_models import CommentCreate, CommentInDB, SystemLogBase
from backend.app.utils.text_utils import TextValidator, DuplicationDetector


class IngestionService:
    """Service for processing and ingesting comment data from various file formats."""

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.validator = TextValidator()
        self.deduplicator = DuplicationDetector()
        
        # Supported file extensions
        self.supported_extensions = {
            '.csv': self._process_csv,
            '.xlsx': self._process_excel,
            '.xls': self._process_excel,
            '.txt': self._process_text,
            '.json': self._process_json
        }
        
        # Common column mappings for CSV/Excel files
        self.column_mappings = {
            'comment': ['comment', 'text', 'feedback', 'response', 'content', 'message'],
            'id': ['id', 'comment_id', 'feedback_id', 'response_id'],
            'law_section': ['law_section', 'section', 'article', 'clause'],
            'stakeholder_type': ['stakeholder_type', 'user_type', 'respondent_type'],
            'stakeholder_category': ['stakeholder_category', 'category', 'user_category'],
            'location': ['location', 'city', 'state', 'region'],
            'submitted_at': ['submitted_at', 'date', 'timestamp', 'created_at'],
            'consultation_id': ['consultation_id', 'process_id', 'survey_id']
        }
    
    async def process_upload(self, file: UploadFile, user_id: str,
                           consultation_id: str = None) -> Dict[str, Any]:
        """
        Process an uploaded file and extract comments.

        Args:
            file: Uploaded file
            user_id: User ID who uploaded the file
            consultation_id: Optional consultation process ID

        Returns:
            dict: Processing results with statistics
        """
        try:
            # Validate file
            self._validate_file(file)

            # Save file temporarily
            file_path = await self._save_temp_file(file)

            # Process based on file type
            file_extension = Path(file.filename).suffix.lower()
            processor = self.supported_extensions.get(file_extension)

            if not processor:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format: {file_extension}"
                )

            # Extract data from file
            raw_data = await processor(file_path)

            # Process and save comments
            result = await self._process_comments(
                raw_data, file.filename, user_id, consultation_id
            )

            # Clean up temp file
            Path(file_path).unlink(missing_ok=True)

            # Log successful ingestion
            await self._log_ingestion(
                user_id=user_id,
                filename=file.filename,
                records_processed=result['total_processed'],
                success=True
            )

            return result

        except Exception as e:
            # Log failed ingestion
            await self._log_ingestion(
                user_id=user_id,
                filename=file.filename,
                records_processed=0,
                success=False,
                error=str(e)
            )
            raise

    async def _log_ingestion(self, user_id: str, filename: str,
                           records_processed: int, success: bool,
                           error: str = None):
        """Log ingestion activity to MongoDB."""
        log_entry = {
            "user_id": user_id,
            "action": "data_ingestion",
            "resource": "comments",
            "details": {
                "filename": filename,
                "records_processed": records_processed
            },
            "success": success,
            "error_message": error,
            "created_at": datetime.utcnow()
        }

        await self.db.system_logs.insert_one(log_entry)
    
    def _validate_file(self, file: UploadFile):
        """Validate uploaded file."""
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file extension
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not supported. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
        
        # Check file size (if we can determine it)
        if hasattr(file, 'size') and file.size:
            if file.size > settings.MAX_UPLOAD_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE / (1024*1024):.1f}MB"
                )
    
    async def _save_temp_file(self, file: UploadFile) -> str:
        """Save uploaded file temporarily."""
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = Path(settings.UPLOAD_DIR) / filename
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        return str(file_path)
    
    async def _process_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Process CSV file."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("Could not read CSV file with any supported encoding")
            
            return self._dataframe_to_records(df)
            
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error processing CSV file: {str(e)}"
            )
    
    async def _process_excel(self, file_path: str) -> List[Dict[str, Any]]:
        """Process Excel file."""
        try:
            # Read Excel file (first sheet by default)
            df = pd.read_excel(file_path, engine='openpyxl')
            return self._dataframe_to_records(df)
            
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error processing Excel file: {str(e)}"
            )
    
    async def _process_text(self, file_path: str) -> List[Dict[str, Any]]:
        """Process plain text file."""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            
            # Split into lines/paragraphs and treat each as a comment
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            return [
                {
                    'comment': line,
                    'row_number': i + 1
                }
                for i, line in enumerate(lines)
            ]
            
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error processing text file: {str(e)}"
            )
    
    async def _process_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Process JSON file."""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
            
            # If it's a list of objects, return as is
            if isinstance(data, list):
                return data
            # If it's a single object with a list inside, extract it
            elif isinstance(data, dict):
                # Look for common list keys
                for key in ['comments', 'data', 'records', 'responses']:
                    if key in data and isinstance(data[key], list):
                        return data[key]
                # If no list found, wrap the object in a list
                return [data]
            else:
                raise ValueError("JSON must contain a list of objects or an object with a list")
                
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error processing JSON file: {str(e)}"
            )
    
    def _dataframe_to_records(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert pandas DataFrame to list of records."""
        # Normalize column names
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        
        # Convert to records
        records = df.to_dict('records')
        
        # Add row numbers
        for i, record in enumerate(records):
            record['row_number'] = i + 2  # +2 because Excel/CSV rows start at 1 and we skip header
        
        return records
    
    async def _process_comments(self, raw_data: List[Dict[str, Any]],
                              source_file: str, user_id: str,
                              consultation_id: str = None) -> Dict[str, Any]:
        """
        Process raw comment data and save to database.
        
        Args:
            raw_data: List of raw comment records
            source_file: Original filename
            user_id: ID of user who uploaded the file
            consultation_id: Optional consultation process ID
            
        Returns:
            dict: Processing statistics
        """
        stats = {
            'total_records': len(raw_data),
            'valid_comments': 0,
            'invalid_comments': 0,
            'duplicates_found': 0,
            'comments_saved': 0,
            'total_processed': 0,
            'errors': []
        }
        
        processed_comments = []
        
        for record in raw_data:
            try:
                # Extract comment text
                comment_text = self._extract_comment_text(record)
                
                if not comment_text:
                    stats['invalid_comments'] += 1
                    stats['errors'].append(f"Row {record.get('row_number', '?')}: No comment text found")
                    continue
                
                # Validate comment
                if not self.validator.is_valid_comment(comment_text):
                    stats['invalid_comments'] += 1
                    stats['errors'].append(f"Row {record.get('row_number', '?')}: Invalid comment format")
                    continue
                
                stats['valid_comments'] += 1

                # Create comment dictionary for MongoDB (matching schema validation)
                comment_dict = {
                    'text': comment_text,
                    'original_text': comment_text,  # Keep both for compatibility
                    'source_file': source_file,
                    'source_row': record.get('row_number'),
                    'comment_id_external': self._extract_field(record, 'id'),
                    'law_section': self._extract_field(record, 'law_section'),
                    'consultation_id': consultation_id or self._extract_field(record, 'consultation_id'),
                    'stakeholder_type': self._extract_field(record, 'stakeholder_type'),
                    'stakeholder_category': self._extract_field(record, 'stakeholder_category'),
                    'location': self._extract_field(record, 'location'),
                    'submitted_at': self._parse_datetime(self._extract_field(record, 'submitted_at')),
                    'uploaded_by': user_id,
                    'word_count': len(comment_text.split()),
                    'character_count': len(comment_text),
                    'sentiment': {'label': 'neutral', 'score': 0.0},  # Default sentiment
                    'created_at': datetime.utcnow(),
                    'updated_at': datetime.utcnow()
                }

                processed_comments.append(comment_dict)
                
            except Exception as e:
                stats['invalid_comments'] += 1
                stats['errors'].append(f"Row {record.get('row_number', '?')}: {str(e)}")
        
        # Detect duplicates using text-based comparison
        duplicate_pairs = self._find_duplicates_by_text(processed_comments)

        # Mark duplicates
        for original_idx, duplicate_idx in duplicate_pairs:
            processed_comments[duplicate_idx]['is_duplicate'] = True
            processed_comments[duplicate_idx]['duplicate_of'] = str(original_idx)
            stats['duplicates_found'] += 1

        # Save comments to database
        if processed_comments:
            try:
                result = await self.db.comments.insert_many(processed_comments)
                stats['comments_saved'] = len(result.inserted_ids)
            except Exception as e:
                stats['errors'].append(f"Database error: {str(e)}")

        stats['total_processed'] = len(processed_comments)

        return stats

    def _find_duplicates_by_text(self, comments: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
        """Find duplicate comments using text-based comparison."""
        duplicates = []

        # First pass: exact matches using hash
        hash_to_index = {}

        for i, comment in enumerate(comments):
            text = comment['original_text']
            text_hash = hashlib.md5(text.encode()).hexdigest()

            if text_hash in hash_to_index:
                duplicates.append((hash_to_index[text_hash], i))
            else:
                hash_to_index[text_hash] = i

        # Second pass: near-duplicates using similarity
        processed = set()
        similarity_threshold = 0.85

        for i, comment1 in enumerate(comments):
            if i in processed:
                continue

            text1 = comment1['original_text']

            for j, comment2 in enumerate(comments[i+1:], i+1):
                if j in processed:
                    continue

                text2 = comment2['original_text']

                if self._are_texts_similar(text1, text2, similarity_threshold):
                    duplicates.append((i, j))
                    processed.add(j)

        return duplicates

    def _are_texts_similar(self, text1: str, text2: str, threshold: float) -> bool:
        """Check if two texts are similar using sequence matching."""
        from difflib import SequenceMatcher

        # Normalize texts
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)

        # Skip if texts are too different in length
        len_ratio = min(len(norm1), len(norm2)) / max(len(norm1), len(norm2))
        if len_ratio < 0.5:
            return False

        # Calculate similarity
        similarity = SequenceMatcher(None, norm1, norm2).ratio()
        return similarity >= threshold

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def _extract_comment_text(self, record: Dict[str, Any]) -> Optional[str]:
        """Extract comment text from record using flexible column mapping."""
        for possible_key in self.column_mappings['comment']:
            if possible_key in record and record[possible_key]:
                text = str(record[possible_key]).strip()
                if text and text.lower() != 'nan':
                    return text
        return None
    
    def _extract_field(self, record: Dict[str, Any], field_type: str) -> Optional[str]:
        """Extract a specific field from record using flexible column mapping."""
        possible_keys = self.column_mappings.get(field_type, [field_type])
        
        for key in possible_keys:
            if key in record and record[key]:
                value = str(record[key]).strip()
                if value and value.lower() != 'nan':
                    return value
        return None
    
    def _parse_datetime(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse datetime string with multiple format support."""
        if not date_str:
            return None
        
        # Common datetime formats
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d',
            '%d/%m/%Y %H:%M:%S',
            '%d/%m/%Y',
            '%m/%d/%Y %H:%M:%S',
            '%m/%d/%Y',
            '%d-%m-%Y %H:%M:%S',
            '%d-%m-%Y'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    async def get_ingestion_stats(self, user_id: str = None) -> Dict[str, Any]:
        """Get ingestion statistics."""
        query = {"action": "data_ingestion"}

        if user_id:
            query["user_id"] = user_id

        cursor = self.db.system_logs.find(query).sort("created_at", -1)
        logs = await cursor.to_list(length=None)

        total_files = len(logs)
        successful_files = sum(1 for log in logs if log.get('success', False))
        total_records = sum(log.get('details', {}).get('records_processed', 0) for log in logs)

        return {
            'total_files_processed': total_files,
            'successful_uploads': successful_files,
            'failed_uploads': total_files - successful_files,
            'total_records_processed': total_records,
            'recent_uploads': [
                {
                    'filename': log.get('details', {}).get('filename', 'Unknown'),
                    'records': log.get('details', {}).get('records_processed', 0),
                    'success': log.get('success', False),
                    'timestamp': log.get('created_at').isoformat() if log.get('created_at') else None
                }
                for log in logs[:10]  # First 10 (most recent)
            ]
        }
    
    async def bulk_process_directory(self, directory_path: str, user_id: str,
                                   consultation_id: str = None) -> Dict[str, Any]:
        """
        Process all supported files in a directory.
        
        Args:
            directory_path: Path to directory containing files
            user: User performing the bulk processing
            consultation_id: Optional consultation process ID
            
        Returns:
            dict: Bulk processing results
        """
        directory = Path(directory_path)
        
        if not directory.exists() or not directory.is_dir():
            raise HTTPException(
                status_code=400,
                detail="Directory does not exist"
            )
        
        results = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_comments': 0,
            'file_results': [],
            'errors': []
        }
        
        # Get all supported files
        supported_files = []
        for ext in self.supported_extensions.keys():
            supported_files.extend(directory.glob(f"*{ext}"))
        
        results['total_files'] = len(supported_files)
        
        for file_path in supported_files:
            try:
                # Create a mock UploadFile
                class MockUploadFile:
                    def __init__(self, path: Path):
                        self.filename = path.name
                        self.file = open(path, 'rb')
                    
                    async def read(self):
                        content = self.file.read()
                        self.file.close()
                        return content
                
                mock_file = MockUploadFile(file_path)
                file_result = await self.process_upload(mock_file, user_id, consultation_id)
                
                results['processed_files'] += 1
                results['total_comments'] += file_result['comments_saved']
                results['file_results'].append({
                    'filename': file_path.name,
                    'success': True,
                    'comments_saved': file_result['comments_saved'],
                    'total_processed': file_result['total_processed']
                })
                
            except Exception as e:
                results['failed_files'] += 1
                results['errors'].append(f"{file_path.name}: {str(e)}")
                results['file_results'].append({
                    'filename': file_path.name,
                    'success': False,
                    'error': str(e)
                })
        
        return results