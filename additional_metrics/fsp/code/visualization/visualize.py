#!/usr/bin/env python3
"""
Simple utility script to create MQM error visualizations.
"""

import argparse
from pathlib import Path
from html_visualizer import create_visualization


def main():
    """Main function for the visualization utility."""
    parser = argparse.ArgumentParser(description='Create HTML visualization for MQM error annotations')
    
    parser.add_argument('--results', '-r', required=True, 
                       help='Path to results JSONL file')
    parser.add_argument('--doc-info', '-d', required=True,
                       help='Path to document info JSONL file')
    parser.add_argument('--output', '-o', 
                       help='Output HTML file path (default: results_visualization.html)')
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if not args.output:
        results_path = Path(args.results)
        args.output = results_path.parent / f"{results_path.stem}_visualization.html"
    
    # Validate input files
    if not Path(args.results).exists():
        print(f"Error: Results file not found: {args.results}")
        return 1
    
    if not Path(args.doc_info).exists():
        print(f"Error: Document info file not found: {args.doc_info}")
        return 1
    
    try:
        create_visualization(args.results, args.doc_info, args.output)
        print(f"✓ Visualization created successfully: {args.output}")
        return 0
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return 1


if __name__ == '__main__':
    exit(main())