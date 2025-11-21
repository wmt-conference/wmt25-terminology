#!/usr/bin/env python3
"""
MT Evaluation System using LLM-as-a-Judge
Evaluates machine translation quality using OpenAI or Cohere APIs with segment-level evaluation.
"""

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import cohere
import nltk
from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI

from utils import setup_logging, segment_text, load_jsonl


def is_cohere_model(model: str) -> bool:
    """Check if the model is a Cohere model based on model name."""
    return "command" in model.lower()


def load_files(input_file: str, test_data_dir: str, target_document_key: str = "output", require_terms: bool = False) -> Tuple[List[Dict], str, str]:
    """
    Load MT output file and find matching source data.
    
    Args:
        input_file: Path to MT system output JSONL file
        test_data_dir: Directory containing test data
        require_terms: Whether to require 'terms' field in source data
        
    Returns:
        Tuple of (mt_data, source_lang, target_lang)
    """
    # Load MT output file
    mt_data = load_jsonl(input_file)
    
    # Extract metadata from filename
    filename = Path(input_file).name
    # Split by dots and find the language pair
    parts = filename.split('.')
    
    # Find the language pair (should be exactly one occurrence of 'enzh' or 'zhen')
    lang_pair_candidates = [part for part in parts if part in ['enzh', 'zhen']]
    
    if len(lang_pair_candidates) == 0:
        raise ValueError(f"No language pair (enzh/zhen) found in filename: {filename}")
    elif len(lang_pair_candidates) > 1:
        raise ValueError(f"Multiple language pairs found in filename: {filename}")
    
    lang_pair = lang_pair_candidates[0]
    lang_pair_index = parts.index(lang_pair)
    
    # Year should be the part before language pair
    if lang_pair_index == 0:
        raise ValueError(f"Language pair cannot be the first part in filename: {filename}")
    year = parts[lang_pair_index - 1]
    
    # Condition should be the part after language pair, conditions: no_term, random, proper
    if lang_pair_index >= len(parts) - 1:
        raise ValueError(f"Language pair cannot be the last part in filename: {filename}")
    condition = parts[lang_pair_index + 1]
    
    # Determine source and target languages
    if lang_pair.startswith('en'):
        assert lang_pair.endswith('zh'), f"Invalid language pair: {lang_pair}"
        source_lang_code = 'en'
        target_lang_code = 'zh'
        source_lang_name = 'English'
        target_lang_name = 'Traditional Chinese (Hong Kong)'
    else:  # zh
        assert lang_pair.endswith('en'), f"Invalid language pair: {lang_pair}"
        source_lang_code = 'zh'
        target_lang_code = 'en'
        source_lang_name = 'Traditional Chinese (Hong Kong)'
        target_lang_name = 'English'
    
    # Find matching source file
    # For V2T template, always use 'proper' condition for reference data
    if require_terms:
        source_filename = f"{year}.{lang_pair}.proper.jsonl"
    else:
        source_filename = f"{year}.{lang_pair}.{condition}.jsonl"
    source_file = Path(test_data_dir) / source_filename
    
    if not source_file.exists():
        raise FileNotFoundError(f"Source file not found: {source_file}")
    
    # Load source data
    source_data = load_jsonl(source_file)

    # Sanity check
    if len(mt_data) != len(source_data):
        raise ValueError(f"Number of MT outputs ({len(mt_data)}) does not match number of source documents ({len(source_data)})")
    
    # Validate and match MT submission data (mt_data) against test data (source_data)
    matched_data = []
    for i, (mt_item, src_item) in enumerate(zip(mt_data, source_data)):
        # Validate that required keys exist
        if source_lang_code not in mt_item:
            raise KeyError(f"Source language key '{source_lang_code}' not found in MT data item {i}")
        if source_lang_code not in src_item:
            raise KeyError(f"Source language key '{source_lang_code}' not found in test data item {i}")
        if target_document_key not in mt_item:
            raise KeyError(f"Target document key '{target_document_key}' not found in MT data item {i}")
        
        # Validate that source texts match between submission and test files
        submission_source = mt_item[source_lang_code]
        test_source = src_item[source_lang_code]
        if submission_source != test_source:
            raise ValueError(f"Source text mismatch at document {i}: submission file and test file contain different source texts")
        
        # Extract terminology if required (for filtered terms data)
        if require_terms:
            if 'terms' not in src_item:
                raise KeyError(f"'terms' field not found in test data item {i}. This suggests the data is not from test_data_filtered_terms.")
            terminology_dict = src_item['terms']
        else:
            terminology_dict = src_item.get('terms', {})
        
        matched_data.append({
            'document_index': i,
            'source_document': src_item[source_lang_code],
            'target_document': mt_item[target_document_key],
            'terminology_dict': terminology_dict,
            'metadata': {
                'year': year,
                'lang_pair': lang_pair,
                'condition': condition,
                'source_lang': source_lang_code,
                'target_lang': target_lang_code
            }
        })
    
    return matched_data, source_lang_name, target_lang_name


def call_judge_api(source_doc: str, target_doc: str, target_segment: str, 
                   src_lang: str, tgt_lang: str, model: str, client: Union[OpenAI, cohere.ClientV2], 
                   template: Template, terminology_dict: Optional[Dict] = None, temperature: float = 0.0, max_tokens: int = 8192, 
                   max_retries: int = 3) -> Dict:
    """
    Call LLM API (OpenAI or Cohere) to evaluate a translation segment.
    Fails fast by raising exception if all retries fail.
    
    Args:
        source_doc: Full source document
        target_doc: Full target document
        target_segment: Specific segment to evaluate
        src_lang: Source language code
        tgt_lang: Target language name
        model: Model to use (OpenAI or Cohere)
        client: API client (OpenAI or Cohere)
        template: Jinja2 template for the prompt
        terminology_dict: Optional terminology dictionary for templates that support it
        
    Returns:
        Judge response as dictionary
        
    Raises:
        RuntimeError: If API call fails after all retries
    """
    # Render the prompt
    template_vars = {
        'src_lang': src_lang,
        'src': source_doc,
        'tgt_lang': tgt_lang,
        'output_seq': target_doc,
        'target_segment': target_segment
    }
    
    # Add terminology_dict if provided (for templates that support it)
    if terminology_dict is not None:
        template_vars['terminology_dict'] = terminology_dict
    
    prompt = template.render(**template_vars)
    
    # Determine if this is a Cohere model
    use_cohere = is_cohere_model(model)
    
    # Call API with retry logic
    for attempt in range(max_retries):
        try:
            if use_cohere:
                # Cohere API call
                response = client.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                # Extract content from Cohere response
                content = response.message.content[0].text
            else:
                # OpenAI API call
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                # Extract content from OpenAI response
                content = response.choices[0].message.content
            
            # Parse JSON response (same for both APIs)
            if content.startswith('```json'):
                content = content[7:-3].strip()
                if content.endswith('```'):
                    content = content[:-3].strip()

            result = json.loads(content)
            return result
            
        except json.JSONDecodeError as e:
            # Show the complete response for debugging
            logging.warning(f"JSON decode error (attempt {attempt + 1}/{max_retries}): {e}")
            logging.warning(f"Complete response content: {repr(content)}")
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to parse JSON after {max_retries} attempts: {e}. Response: {repr(content)}")
        except Exception as e:
            logging.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise RuntimeError(f"API call failed after {max_retries} attempts: {e}")


def get_run_directory(base_output_dir: str, eval_runname: str) -> Path:
    """Create run-specific output directory."""
    return Path(base_output_dir) / eval_runname


def save_run_config(run_dir: Path, args) -> None:
    """Save complete argparse configuration."""
    config_file = run_dir / "config.json"
    config = vars(args)  # Convert argparse Namespace to dict
    
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_and_compare_config(run_dir: Path, current_args) -> Tuple[bool, str]:
    """
    Load saved config and compare with current args.
    Returns (configs_match, message)
    """
    config_file = run_dir / "config.json"
    
    if not config_file.exists():
        return False, "No previous config found"
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            saved_config = json.load(f)
        
        current_config = vars(current_args)
        
        if saved_config == current_config:
            return True, "Configuration matches - resuming"
        else:
            # Find differences for user feedback
            differences = []
            all_keys = set(saved_config.keys()) | set(current_config.keys())
            
            for key in all_keys:
                if saved_config.get(key) != current_config.get(key):
                    differences.append(f"{key}: {saved_config.get(key)} -> {current_config.get(key)}")
            
            return False, f"Configuration mismatch: {'; '.join(differences)}"
    
    except Exception as e:
        return False, f"Error reading config: {e}"


def load_existing_results(run_dir: Path) -> Tuple[Dict[str, Dict], set]:
    """Load existing results from run directory."""
    results_file = run_dir / "results.jsonl"
    doc_info_file = run_dir / "results_doc_info.jsonl"
    
    existing_results = {}
    processed_docs = set()
    
    if results_file.exists():
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    key = f"{result['document_index']}_{result['segment_index']}"
                    existing_results[key] = result
    
    if doc_info_file.exists():
        with open(doc_info_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    doc_info = json.loads(line)
                    processed_docs.add(doc_info['document_index'])
    
    return existing_results, processed_docs


def append_result_to_run(result: Dict, run_dir: Path) -> None:
    """Append result to run's results file."""
    results_file = run_dir / "results.jsonl"
    with open(results_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


def save_document_info(doc_data: Dict, total_segments: int, run_dir: Path) -> None:
    """Save document-level information to separate file."""
    doc_info_file = run_dir / "results_doc_info.jsonl"
    
    doc_info = {
        'document_index': doc_data['document_index'],
        'source_document': doc_data['source_document'],
        'target_document': doc_data['target_document'],
        'metadata': doc_data['metadata'],
        'total_segments': total_segments,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    }
    
    with open(doc_info_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(doc_info, ensure_ascii=False) + '\n')


def save_results(results: List[Dict], output_dir: str, metadata: Dict) -> None:
    """
    Save evaluation results to JSONL file.
    
    Args:
        results: List of segment evaluation results
        output_dir: Output directory
        metadata: File metadata
    """
    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create output filename
    output_filename = f"{metadata['year']}.{metadata['lang_pair']}.{metadata['condition']}.evaluated.jsonl"
    output_file = output_path / output_filename
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    logging.info(f"Results saved to: {output_file}")


def main():
    """Main function."""
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Evaluate MT quality using LLM-as-a-Judge')
    parser.add_argument('--input-file', required=True, help='MT output JSONL file')
    parser.add_argument('--output-dir', required=True, help='Base output directory')
    parser.add_argument('--eval-runname', default="debug", help='Evaluation run name for organization and resume')
    parser.add_argument('--test-data-dir', default='data/test_data/track2', 
                       help='Test data directory')
    parser.add_argument('--segment-size', type=int, default=1, 
                       help='Number of sentences per segment')
    parser.add_argument('--model', default='gpt-4o', 
                       help='Model to use (OpenAI: gpt-4o, gpt-4, etc. | Cohere: command-a-03-2025, etc.)')
    parser.add_argument('--base-url', default='https://api.openai.com/v1',
                       help='OpenAI API base URL (default: https://api.openai.com/v1, ignored for Cohere models)')
    parser.add_argument('--template-path', default='templates/fsp_judge_v1.jinja',
                       help='Path to judge template')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Temperature for API calls (default: 0.0)')
    parser.add_argument('--max-tokens', type=int, default=1000,
                       help='Maximum tokens for API response (default: 1000)')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum number of API retry attempts (default: 3)')
    parser.add_argument('--target-document-key', default='output',
                       help='Key name for target document in MT output files (default: output)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Check for appropriate API key based on model
    use_cohere = is_cohere_model(args.model)
    
    if use_cohere:
        if not os.getenv('COHERE_API_KEY'):
            logging.error("COHERE_API_KEY environment variable not set for Cohere model")
            return 1
    else:
        if not os.getenv('OPENAI_API_KEY'):
            logging.error("OPENAI_API_KEY environment variable not set for OpenAI model")
            return 1
    
    try:
        # Create run-specific directory
        run_dir = get_run_directory(args.output_dir, args.eval_runname)
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for existing configuration
        configs_match, config_message = load_and_compare_config(run_dir, args)
        
        if configs_match:
            logging.info(f"Resuming evaluation run: {args.eval_runname}")
            logging.info(f"Config check: {config_message}")
            existing_results, processed_docs = load_existing_results(run_dir)
            logging.info(f"Found {len(existing_results)} existing results from {len(processed_docs)} documents")
        else:
            if (run_dir / "config.json").exists():
                # Configuration mismatch - raise error to prevent continuing with different config
                logging.error(f"Configuration mismatch for run: {args.eval_runname}")
                logging.error(f"Config check: {config_message}")
                raise ValueError(
                    f"Configuration mismatch detected for run '{args.eval_runname}'. "
                    f"Details: {config_message}. "
                    f"To continue, either use a different --eval-runname or manually delete the run directory: {run_dir}"
                )
            else:
                logging.info(f"Starting new evaluation run: {args.eval_runname}")
            
            existing_results, processed_docs = {}, set()
        
        # Save current configuration
        save_run_config(run_dir, args)
        
        # Initialize appropriate API client based on model
        if use_cohere:
            client = cohere.ClientV2(api_key=os.getenv('COHERE_API_KEY'))
            logging.info(f"Using Cohere API with model: {args.model}")
        else:
            client = OpenAI(base_url=args.base_url)
            logging.info(f"Using OpenAI API with model: {args.model}")
        
        # Load judge template
        with open(args.template_path, 'r', encoding='utf-8') as f:
            template = Template(f.read())
        
        # Validate template is supported
        template_name = Path(args.template_path).name
        supported_templates = ['fsp_judge_v1.jinja', 'fsp_judge_v2.jinja', 'fsp_judge_v2_T.jinja']
        if template_name not in supported_templates:
            raise ValueError(f"Template '{template_name}' is not supported. Supported templates: {', '.join(supported_templates)}")
        
        # Validate that v2_T template is only used with filtered terms data
        if template_name == 'fsp_judge_v2_T.jinja':
            test_data_path = Path(args.test_data_dir)
            # Check if 'test_data_filtered_terms' exists anywhere in the path
            if 'test_data_filtered_terms' not in test_data_path.parts:
                raise ValueError(
                    f"Template 'fsp_judge_v2_T.jinja' can only be used with 'test_data_filtered_terms' data. "
                    f"Current test data directory: '{args.test_data_dir}'. "
                    f"Please use '--test-data-dir data/test_data_filtered_terms/track2' or similar path."
                )
        
        # Load files
        logging.info(f"Loading files from {args.input_file}")
        require_terms = template_name == 'fsp_judge_v2_T.jinja'
        matched_data, source_lang_name, target_lang_name = load_files(args.input_file, args.test_data_dir, args.target_document_key, require_terms)
        
        logging.info(f"Processing {len(matched_data)} documents")
        logging.info(f"Language pair: {source_lang_name} -> {target_lang_name}")
        logging.info(f"Run directory: {run_dir}")
        
        successful_count = len(existing_results)
        
        # Process each document with fail-fast behavior
        for doc_data in matched_data:
            doc_idx = doc_data['document_index']
            
            # Skip if document already fully processed
            if doc_idx in processed_docs:
                logging.debug(f"Skipping already processed document {doc_idx}")
                continue
                
            logging.info(f"Processing document {doc_idx}")
            
            # Determine target language for proper segmentation
            target_lang_code = doc_data['metadata']['target_lang']
            
            # Segment only the target text for evaluation
            target_segments = segment_text(doc_data['target_document'], target_lang_code, args.segment_size)
            
            # Save document info once per document
            save_document_info(doc_data, len(target_segments), run_dir)
            
            # Evaluate each target segment
            for seg_idx, tgt_seg in enumerate(target_segments):
                # Skip if already processed
                segment_key = f"{doc_idx}_{seg_idx}"
                if segment_key in existing_results:
                    logging.debug(f"Skipping already processed segment {segment_key}")
                    continue
                
                logging.info(f"Processing doc {doc_idx}, segment {seg_idx + 1}/{len(target_segments)}")
                
                # Call judge API (will raise exception if all retries fail)
                judge_result = call_judge_api(
                    source_doc=doc_data['source_document'],      # Full source document
                    target_doc=doc_data['target_document'],      # Full target document  
                    target_segment=tgt_seg,                      # Only the target segment to evaluate
                    src_lang=source_lang_name,
                    tgt_lang=target_lang_name,
                    model=args.model,
                    client=client,
                    template=template,
                    terminology_dict=doc_data['terminology_dict'],  # Pass terminology dictionary
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    max_retries=args.max_retries
                )
                
                # Create result (now without redundant document data)
                result = {
                    'document_index': doc_idx,
                    'segment_index': seg_idx,
                    'target_segment': tgt_seg,
                    'judge_response': judge_result,
                    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                }
                
                # Write immediately to file
                append_result_to_run(result, run_dir)
                successful_count += 1
                
                # Optional: small delay for rate limiting
                # time.sleep(0.1)
        
        logging.info(f"Evaluation complete: {successful_count} segments processed successfully")
        logging.info(f"Results saved in: {run_dir}")
        return 0
        
    except Exception as e:
        logging.error(f"Evaluation stopped due to error: {e}")
        logging.info(f"Progress saved. To resume, run the same command with identical parameters.")
        return 1


if __name__ == '__main__':
    exit(main())