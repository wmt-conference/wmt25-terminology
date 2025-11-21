#!/usr/bin/env python3
"""
Shared utilities for MT evaluation and metrics processing.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import nltk


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_jsonl(file_path: Union[str, Path]) -> List[Dict]:
    """
    Load JSONL file with error handling.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries from JSONL file
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file contains invalid JSON
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")
    
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_num} in {file_path}: {e}")
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Error reading JSONL file {file_path}: {e}")
    
    logging.info(f"Loaded {len(data)} records from {file_path}")
    return data


def count_tokens(text: str, language: str) -> int:
    """
    Count tokens (words for English, characters for Chinese) in text.
    
    Args:
        text: Input text to tokenize
        language: Language code ('en' or 'zh')
        
    Returns:
        Number of tokens
    """
    if not text or not text.strip():
        return 0
    
    if language == 'zh':
        # For Chinese: count characters excluding whitespace
        return len(re.sub(r'\s+', '', text))
    else:
        # For English: split on whitespace and count words
        return len(text.split())


def segment_text(text: str, language: str, k: int = 1) -> List[str]:
    """
    Split text into segments of k sentences each.
    
    Args:
        text: Input text to segment
        language: 'en' for English, 'zh' for Chinese
        k: Number of sentences per segment
        
    Returns:
        List of text segments
    """
    if language == 'en':
        # Always use NLTK for English sentence tokenization, no fallback
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError as e:
            raise RuntimeError(
                f"NLTK sentence tokenizer data not available: {e}. "
                f"Please install NLTK data with: python -m nltk.downloader punkt"
            )
    else:  # Chinese
        # Use regex for Chinese sentence boundaries, preserving punctuation
        sentences = re.split(r'(?<=[。！？；])', text)
        sentences = [s for s in sentences if s]  # Only remove truly empty strings
    
    # Group sentences into segments of size k
    segments = []
    for i in range(0, len(sentences), k):
        if language == 'zh':
            # For Chinese, join without spaces
            segment = ''.join(sentences[i:i+k])
        else:
            # For English, join with spaces
            segment = ' '.join(sentences[i:i+k])
        
        if segment.strip():
            segments.append(segment.strip())
    
    return segments


def save_jsonl(data: List[Dict], file_path: Union[str, Path]) -> None:
    """
    Save data to JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Output file path
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logging.info(f"Saved {len(data)} records to {file_path}")


def validate_required_fields(data: Dict, required_fields: List[str], context: str = "") -> None:
    """
    Validate that required fields are present in data dictionary.
    
    Args:
        data: Dictionary to validate
        required_fields: List of required field names
        context: Context string for error messages
        
    Raises:
        ValueError: If any required field is missing
    """
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        context_str = f" in {context}" if context else ""
        raise ValueError(f"Missing required fields{context_str}: {missing_fields}")


def validate_segment_structure(segment: Dict) -> bool:
    """
    Validate that segment data has the expected structure for evaluation processing.
    
    Args:
        segment: Dictionary containing segment evaluation data
        
    Returns:
        True if segment has valid structure, False otherwise
    """
    if not isinstance(segment, dict):
        logging.error("Segment data is not a dictionary")
        return False
    
    # Check for required metadata fields
    metadata = segment.get('metadata')
    if not isinstance(metadata, dict):
        logging.error("Missing or invalid 'metadata' field in segment")
        return False
    
    required_metadata_fields = ['document_index', 'segment_index']
    for field in required_metadata_fields:
        if field not in metadata:
            logging.error(f"Missing required metadata field: {field}")
            return False
    
    # Check for target_segment field
    if 'target_segment' not in segment:
        logging.error("Missing 'target_segment' field in segment")
        return False
    
    # Check for judge_response field
    judge_response = segment.get('judge_response')
    if not isinstance(judge_response, dict):
        logging.error("Missing or invalid 'judge_response' field in segment")
        return False
    
    return True