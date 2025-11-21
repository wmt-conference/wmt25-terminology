#!/usr/bin/env python3
"""
MQM Score Computation for MT Evaluation Results
Simple script to compute quality and error scores from LLM evaluation results.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from utils import load_jsonl, save_jsonl
from visualization.html_visualizer import create_visualization


def save_to_excel(results, excel_file, weights):
    """
    Save results to Excel file with multiple sheets.
    
    Args:
        results: List of processed segment results
        excel_file: Path to Excel output file
        weights: Dictionary of error weights used
    """
    try:
        # Create DataFrame from results
        df = pd.DataFrame(results)
        
        # Create document-level summary
        doc_summary = df.groupby('document_index').agg({
            'quality_score': ['mean', 'count'],
            'error_score': 'sum',
            'segment_length': 'sum'
        }).round(2)
        
        # Flatten column names
        doc_summary.columns = ['avg_quality_score', 'segment_count', 'total_error_score', 'total_length']
        doc_summary = doc_summary.reset_index()
        
        # Calculate scores per 1000 characters/words
        doc_summary['quality_score_per_1k'] = (doc_summary['avg_quality_score']).round(2)
        doc_summary['error_score_per_1k'] = (doc_summary['total_error_score'] / doc_summary['total_length'] * 1000).round(2)
        
        # Create metadata info
        metadata = pd.DataFrame([
            ['Error Weights', ''],
            ['Minor Weight', weights['minor']],
            ['Major Weight', weights['major']],
            ['Critical Weight', weights['critical']],
            ['', ''],
            ['Summary', ''],
            ['Total Documents', len(doc_summary)],
            ['Total Segments', len(results)],
            ['Avg Quality Score', df['quality_score'].mean().round(2) if df['quality_score'].notna().any() else 'N/A'],
            ['Total Error Score', df['error_score'].sum().round(2)]
        ], columns=['Parameter', 'Value'])
        
        # Write to Excel with multiple sheets
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Segment-level data
            df.to_excel(writer, sheet_name='Segment_Scores', index=False)
            
            # Document-level summary
            doc_summary.to_excel(writer, sheet_name='Document_Summary', index=False)
            
            # Metadata and parameters
            metadata.to_excel(writer, sheet_name='Metadata', index=False)
        
        logging.info(f"Excel file saved: {excel_file}")
        
    except Exception as e:
        logging.warning(f"Failed to save Excel file: {e}")


def compute_scores(input_file, output_file, minor_weight=1.0, major_weight=2.0, critical_weight=5.0, save_excel=True, create_html_viz=False, doc_info_file=None):
    """
    Compute MQM-style scores from evaluation results.
    
    Args:
        input_file: Path to JSONL file with evaluation results
        output_file: Path to output JSONL file with computed scores
        minor_weight: Weight for minor errors (default: 1.0)
        major_weight: Weight for major errors (default: 2.0)  
        critical_weight: Weight for critical errors (default: 5.0)
        save_excel: Whether to also save Excel output (default: True)
        create_html_viz: Whether to create HTML visualization (default: False)
        doc_info_file: Path to document info JSONL file (required for HTML viz)
    """
    weights = {
        'minor': minor_weight,
        'major': major_weight,
        'critical': critical_weight
    }
    
    logging.info(f"Processing {input_file} with weights: {weights}")
    
    # Load input data using utils function
    try:
        segments = load_jsonl(input_file)
    except Exception as e:
        logging.error(f"Failed to load input file: {e}")
        return
    
    results = []
    processed_count = 0
    error_count = 0
    
    # Process each segment
    for segment in segments:
        try:
            # Extract basic info - handle both formats
            if 'metadata' in segment:
                doc_idx = segment['metadata']['document_index']
                seg_idx = segment['metadata']['segment_index']
            else:
                doc_idx = segment['document_index']
                seg_idx = segment['segment_index']
            
            target_text = segment['target_segment']
            judge_response = segment['judge_response']
            
            # Extract quality score
            quality_score = judge_response.get('quality_score')
            if quality_score is not None:
                try:
                    quality_score = float(quality_score)
                    if not (0 <= quality_score <= 100):
                        quality_score = None
                except (ValueError, TypeError):
                    quality_score = None
            
            # Compute error score
            error_score = 0.0
            errors = judge_response.get('errors', [])
            for error in errors:
                severity = error.get('severity')
                if severity in weights:
                    error_score += weights[severity]
            
            # Create result
            result = {
                'document_index': doc_idx,
                'segment_index': seg_idx,
                'target_segment': target_text,
                'quality_score': quality_score,
                'error_score': error_score,
                'segment_length': len(target_text) if target_text else 0
            }
            
            results.append(result)
            processed_count += 1
            
        except Exception as e:
            logging.warning(f"Error processing segment {segment.get('metadata', {}).get('document_index', 'unknown')}_{segment.get('metadata', {}).get('segment_index', 'unknown')}: {e}")
            error_count += 1
            continue
    
    # Save JSONL output using utils function
    save_jsonl(results, output_file)
    
    # Also save to Excel if requested
    if save_excel:
        excel_file = output_file.replace('.jsonl', '.xlsx')
        save_to_excel(results, excel_file, weights)
        logging.info(f"Results saved to {output_file} and {excel_file}")
    else:
        logging.info(f"Results saved to {output_file}")
    
    logging.info(f"Processed {processed_count} segments, {error_count} errors")
    
    # Create HTML visualization if requested
    if create_html_viz and doc_info_file:
        try:
            html_file = output_file.replace('.jsonl', '_visualization.html')
            create_visualization(input_file, doc_info_file, html_file)
            logging.info(f"HTML visualization created: {html_file}")
        except Exception as e:
            logging.warning(f"Failed to create HTML visualization: {e}")
    elif create_html_viz and not doc_info_file:
        logging.warning("HTML visualization requested but doc_info_file not provided")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Compute MQM scores from MT evaluation results')
    
    parser.add_argument('--input-file', '-i', required=True, help='Input JSONL file')
    parser.add_argument('--output-file', '-o', required=True, help='Output JSONL file')
    parser.add_argument('--minor-weight', type=float, default=1.0, help='Weight for minor errors')
    parser.add_argument('--major-weight', type=float, default=2.0, help='Weight for major errors')
    parser.add_argument('--critical-weight', type=float, default=5.0, help='Weight for critical errors')
    parser.add_argument('--no-excel', action='store_true', help='Skip Excel output generation')
    parser.add_argument('--create-html-viz', action='store_true', help='Create HTML visualization')
    parser.add_argument('--doc-info-file', help='Document info JSONL file (required for HTML visualization)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Validate input
    if not Path(args.input_file).exists():
        logging.error(f"Input file not found: {args.input_file}")
        return 1
    
    try:
        compute_scores(
            args.input_file, 
            args.output_file,
            args.minor_weight,
            args.major_weight, 
            args.critical_weight,
            save_excel=not args.no_excel,
            create_html_viz=args.create_html_viz,
            doc_info_file=args.doc_info_file
        )
        return 0
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())