#!/usr/bin/env python3
"""
HTML Visualizer for MQM Error Annotations
Generates interactive HTML pages showing source/target documents with highlighted errors.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import html


class MQMHTMLVisualizer:
    """Visualizer for MQM error annotations in HTML format."""
    
    def __init__(self):
        self.severity_colors = {
            'minor': '#fff4e6',      # Light orange (matching stats)
            'major': '#ffe6e6',      # Light red (matching stats)  
            'critical': '#f3e6ff'    # Light purple (matching stats)
        }
        self.severity_border_colors = {
            'minor': '#f39c12',      # Orange (matching stats)
            'major': '#e74c3c',      # Red (matching stats)
            'critical': '#8e44ad'    # Purple (matching stats)
        }
    
    def load_data(self, results_file: str, doc_info_file: str) -> Tuple[List[Dict], List[Dict]]:
        """Load results and document info from JSONL files."""
        results = []
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
        
        doc_info = []
        with open(doc_info_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    doc_info.append(json.loads(line))
        
        return results, doc_info
    
    def find_error_positions(self, target_text: str, error_span: str) -> List[Tuple[int, int]]:
        """Find all positions of error_span in target_text."""
        positions = []
        start = 0
        
        # Try exact match first
        while True:
            pos = target_text.find(error_span, start)
            if pos == -1:
                break
            positions.append((pos, pos + len(error_span)))
            start = pos + 1
        
        # If no exact match, try various normalization approaches
        if not positions:
            # Method 1: Normalize whitespace
            normalized_target = re.sub(r'\s+', ' ', target_text.strip())
            normalized_span = re.sub(r'\s+', ' ', error_span.strip())
            
            pos = normalized_target.find(normalized_span)
            if pos != -1:
                # Try to map back to original position
                original_pos = self._map_normalized_position(target_text, normalized_target, pos)
                if original_pos != -1:
                    positions.append((original_pos, original_pos + len(error_span)))
            
            # Method 2: Try partial matching for long spans
            if not positions and len(error_span) > 50:
                # Try matching first 30 characters
                partial_span = error_span[:30]
                pos = target_text.find(partial_span)
                if pos != -1:
                    # Extend to find the full span
                    end_pos = min(pos + len(error_span), len(target_text))
                    positions.append((pos, end_pos))
            
            # Method 3: Try case-insensitive matching
            if not positions:
                pos = target_text.lower().find(error_span.lower())
                if pos != -1:
                    positions.append((pos, pos + len(error_span)))
        
        return positions
    
    def _map_normalized_position(self, original: str, normalized: str, norm_pos: int) -> int:
        """Map position from normalized text back to original text."""
        try:
            # Simple approximation - count characters up to the position
            orig_chars = 0
            norm_chars = 0
            
            for char in original:
                if norm_chars >= norm_pos:
                    return orig_chars
                if not char.isspace() or (norm_chars < len(normalized) and normalized[norm_chars] == ' '):
                    norm_chars += 1
                orig_chars += 1
            
            return orig_chars
        except:
            return -1
    
    def _generate_summary_stats(self, results: List[Dict]) -> str:
        """Generate summary statistics section."""
        total_segments = len(results)
        total_errors = 0
        severity_counts = {'minor': 0, 'major': 0, 'critical': 0}
        category_counts = {}
        quality_scores = []
        
        for result in results:
            errors = result.get('judge_response', {}).get('errors', [])
            total_errors += len(errors)
            
            quality_score = result.get('judge_response', {}).get('quality_score')
            if quality_score is not None:
                try:
                    quality_scores.append(float(quality_score))
                except (ValueError, TypeError):
                    pass
            
            for error in errors:
                severity = error.get('severity', 'unknown')
                if severity in severity_counts:
                    severity_counts[severity] += 1
                
                category = error.get('error_category', 'unknown')
                category_counts[category] = category_counts.get(category, 0) + 1
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        # Generate top error categories
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return f'''
        <div class="summary-stats">
            <h2>📊 Summary Statistics</h2>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-number">{total_segments}</div>
                    <div class="stat-label">Total Segments</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{total_errors}</div>
                    <div class="stat-label">Total Errors</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{avg_quality:.1f}</div>
                    <div class="stat-label">Avg Quality Score</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{total_errors/total_segments:.1f}</div>
                    <div class="stat-label">Errors per Segment</div>
                </div>
            </div>
            
            <div class="severity-breakdown">
                <h3>Error Severity Breakdown</h3>
                <div class="severity-bars">
                    <div class="severity-bar">
                        <span class="severity-label">
                            <span class="severity-badge severity-minor">Minor</span>
                            {severity_counts['minor']} errors
                        </span>
                        <div class="bar-container">
                            <div class="bar minor-bar" style="width: {(severity_counts['minor']/total_errors*100) if total_errors > 0 else 0:.1f}%"></div>
                        </div>
                    </div>
                    <div class="severity-bar">
                        <span class="severity-label">
                            <span class="severity-badge severity-major">Major</span>
                            {severity_counts['major']} errors
                        </span>
                        <div class="bar-container">
                            <div class="bar major-bar" style="width: {(severity_counts['major']/total_errors*100) if total_errors > 0 else 0:.1f}%"></div>
                        </div>
                    </div>
                    <div class="severity-bar">
                        <span class="severity-label">
                            <span class="severity-badge severity-critical">Critical</span>
                            {severity_counts['critical']} errors
                        </span>
                        <div class="bar-container">
                            <div class="bar critical-bar" style="width: {(severity_counts['critical']/total_errors*100) if total_errors > 0 else 0:.1f}%"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="category-breakdown">
                <h3>Top Error Categories</h3>
                <div class="category-list">
                    {''.join([f'<div class="category-item"><span class="category-name">{cat.title()}</span><span class="category-count">{count}</span></div>' for cat, count in top_categories])}
                </div>
            </div>
        </div>
        '''
    
    def highlight_errors(self, target_text: str, errors: List[Dict]) -> Tuple[str, List[Dict]]:
        """Highlight error spans in target text and return unmatchable errors."""
        highlighted_text = target_text
        unmatchable_errors = []
        
        # Sort errors by position to avoid overlap issues
        error_positions = []
        
        for error in errors:
            error_span = error.get('error_span', '')
            # Skip empty error spans to avoid character-by-character highlighting
            if not error_span.strip():
                unmatchable_errors.append(error)
                continue
                
            positions = self.find_error_positions(target_text, error_span)
            
            if positions:
                for start, end in positions:
                    error_positions.append({
                        'start': start,
                        'end': end,
                        'error': error
                    })
            else:
                unmatchable_errors.append(error)
        
        # Sort by start position (reverse order for replacement)
        error_positions.sort(key=lambda x: x['start'], reverse=True)
        
        # Build the result by processing text in order (not reverse)
        error_positions.sort(key=lambda x: x['start'])  # Sort in normal order
        
        result_parts = []
        current_pos = 0
        
        for pos_info in error_positions:
            start, end = pos_info['start'], pos_info['end']
            error = pos_info['error']
            
            # Add escaped text before this error
            if start > current_pos:
                result_parts.append(html.escape(target_text[current_pos:start]))
            
            # Add the highlighted error span
            severity = error.get('severity', 'minor')
            bg_color = self.severity_colors.get(severity, '#fff4e6')
            border_color = self.severity_border_colors.get(severity, '#f39c12')
            
            original_text = target_text[start:end]
            
            # Store error data in separate data attributes
            error_span_attr = html.escape(error.get('error_span', ''))
            explanation_attr = html.escape(error.get('explanation', ''))
            category_attr = html.escape(error.get('error_category', ''))
            type_attr = html.escape(error.get('error_type', ''))
            
            highlighted_span = f'<span class="error-highlight" style="background-color: {bg_color}; border-bottom: 2px solid {border_color}; cursor: help;" data-severity="{severity}" data-error-span="{error_span_attr}" data-explanation="{explanation_attr}" data-category="{category_attr}" data-type="{type_attr}" onmouseenter="showErrorTooltip(this, event)" onmouseleave="hideErrorTooltip()" onmousemove="moveErrorTooltip(event)">{html.escape(original_text)}</span>'
            
            result_parts.append(highlighted_span)
            current_pos = end
        
        # Add any remaining text after the last error
        if current_pos < len(target_text):
            result_parts.append(html.escape(target_text[current_pos:]))
        
        return ''.join(result_parts), unmatchable_errors
    

    
    def _create_tooltip_content(self, error: Dict) -> str:
        """Create tooltip content for an error."""
        parts = []
        if error.get('severity'):
            parts.append(f"Severity: {error['severity'].title()}")
        if error.get('error_category'):
            parts.append(f"Category: {error['error_category'].title()}")
        if error.get('error_type'):
            parts.append(f"Type: {error['error_type'].title()}")
        if error.get('explanation'):
            parts.append(f"Explanation: {error['explanation']}")
        
        return " | ".join(parts)
    
    def generate_html(self, results: List[Dict], doc_info: List[Dict], output_file: str):
        """Generate complete HTML visualization."""
        
        # Group results by document
        doc_results = {}
        for result in results:
            doc_idx = result['document_index']
            if doc_idx not in doc_results:
                doc_results[doc_idx] = []
            doc_results[doc_idx].append(result)
        
        # Create document info lookup
        doc_lookup = {doc['document_index']: doc for doc in doc_info}
        
        html_content = self._generate_html_template()
        
        # Generate summary statistics
        summary_stats = self._generate_summary_stats(results)
        
        # Generate content for each document
        document_sections = []
        
        for doc_idx in sorted(doc_results.keys()):
            if doc_idx not in doc_lookup:
                continue
                
            doc_data = doc_lookup[doc_idx]
            segments = doc_results[doc_idx]
            
            doc_section = self._generate_document_section(doc_data, segments)
            document_sections.append(doc_section)
        
        # Insert summary and document sections into template
        final_html = html_content.replace('{{SUMMARY_STATS}}', summary_stats)
        final_html = final_html.replace('{{DOCUMENT_SECTIONS}}', '\n'.join(document_sections))
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_html)
    
    def _generate_document_section(self, doc_data: Dict, segments: List[Dict]) -> str:
        """Generate HTML section for a single document."""
        source_doc = doc_data.get('source_document', '')
        target_doc = doc_data.get('target_document', '')
        metadata = doc_data.get('metadata', {})
        
        # Process all segments and concatenate target text
        all_highlighted_segments = []
        total_errors = 0
        all_unmatchable_errors = []
        
        # Sort segments by segment_index to ensure proper order
        sorted_segments = sorted(segments, key=lambda x: x.get('segment_index', 0))
        
        for segment in sorted_segments:
            target_segment = segment.get('target_segment', '')
            errors = segment.get('judge_response', {}).get('errors', [])
            quality_score = segment.get('judge_response', {}).get('quality_score', 'N/A')
            segment_index = segment.get('segment_index', 0)
            
            highlighted_target, unmatchable_errors = self.highlight_errors(target_segment, errors)
            all_unmatchable_errors.extend(unmatchable_errors)
            total_errors += len(errors)
            
            # Add subtle segment separator (except for first segment)
            if all_highlighted_segments:
                all_highlighted_segments.append(f'<span class="segment-separator">• Segment {segment_index} (Score: {quality_score}) •</span>')
            else:
                all_highlighted_segments.append(f'<span class="segment-separator">Segment {segment_index} (Score: {quality_score})</span>')
            
            all_highlighted_segments.append(highlighted_target)
        
        # Join all segments with line breaks
        combined_target_text = '\n\n'.join(all_highlighted_segments)
        
        # Generate unmatchable errors section
        unmatchable_section = ""
        if all_unmatchable_errors:
            unmatchable_items = []
            for error in all_unmatchable_errors:
                error_html = f'''
                <div class="unmatchable-error">
                    <strong>Error Span:</strong> "{html.escape(error.get('error_span', ''))}"<br>
                    <strong>Severity:</strong> {error.get('severity', 'N/A').title()}<br>
                    <strong>Category:</strong> {error.get('error_category', 'N/A').title()}<br>
                    <strong>Type:</strong> {error.get('error_type', 'N/A').title()}<br>
                    <strong>Explanation:</strong> {html.escape(error.get('explanation', ''))}
                </div>
                '''
                unmatchable_items.append(error_html)
            
            unmatchable_section = f'''
            <div class="unmatchable-errors">
                <h4>Unmatchable Errors ({len(all_unmatchable_errors)})</h4>
                <div class="unmatchable-list">
                    {''.join(unmatchable_items)}
                </div>
            </div>
            '''
        
        return f'''
        <div class="document">
            <div class="document-header">
                <h2>Document {doc_data['document_index']} ({total_errors} errors total)</h2>
                <div class="metadata">
                    <span><strong>Language Pair:</strong> {metadata.get('lang_pair', 'N/A')}</span>
                    <span><strong>Source Language:</strong> {metadata.get('source_lang', 'N/A')}</span>
                    <span><strong>Target Language:</strong> {metadata.get('target_lang', 'N/A')}</span>
                </div>
            </div>
            
            <div class="document-content-columns">
                <div class="source-column">
                    <h3>Source Document</h3>
                    <div class="text-content">{html.escape(source_doc)}</div>
                </div>
                
                <div class="target-column">
                    <h3>Target Document (with Error Annotations)</h3>
                    <div class="text-content highlighted-content">{combined_target_text}</div>
                </div>
            </div>
            
            {unmatchable_section}
        </div>
        '''
    
    def _generate_html_template(self) -> str:
        """Generate the HTML template with CSS and JavaScript."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MQM Error Visualization</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }
        
        .document {
            margin-bottom: 40px;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }
        
        .document-header {
            background: #007bff;
            color: white;
            padding: 15px 20px;
        }
        
        .document-header h2 {
            margin: 0 0 10px 0;
        }
        
        .metadata {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        .metadata span {
            margin-right: 20px;
        }
        
        .document-content {
            padding: 20px;
        }
        
        .document-content-columns {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            padding: 20px;
        }
        
        .source-column, .target-column {
            min-height: 400px;
        }
        
        .source-column h3, .target-column h3 {
            color: #495057;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 8px;
            margin-bottom: 15px;
            margin-top: 0;
        }
        
        .segment-separator {
            display: block;
            color: #6c757d;
            font-size: 11px;
            font-weight: 500;
            text-align: center;
            margin: 15px 0 10px 0;
            padding: 5px 0;
            border-top: 1px solid #e9ecef;
            border-bottom: 1px solid #e9ecef;
            background: #f8f9fa;
            opacity: 0.8;
        }
        
        .text-content {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #007bff;
            white-space: pre-wrap;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', 'Helvetica Neue', Helvetica, Arial, sans-serif;
            font-size: 14px;
            line-height: 1.6;
        }
        
        .highlighted-content {
            white-space: pre-wrap;
            word-break: keep-all;
            overflow-wrap: break-word;
        }
        

        
        .error-highlight {
            position: relative;
            padding: 3px 6px;
            border-radius: 4px;
            transition: all 0.3s ease;
            margin: 0 1px;
            display: inline-block;
            animation: subtle-pulse 2s ease-in-out infinite;
            font-weight: 500;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        @keyframes subtle-pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.85; }
        }
        
        .error-highlight:hover {
            animation: none;
        }
        
        .error-highlight:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(0,0,0,0.25);
            z-index: 10;
            position: relative;
            filter: brightness(1.05) saturate(1.1);
        }
        
        /* Enhanced Tooltip Styles */
        .error-tooltip {
            position: absolute;
            background: #2c3e50;
            color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            z-index: 1000;
            max-width: 400px;
            min-width: 250px;
            font-size: 13px;
            line-height: 1.4;
            opacity: 0;
            transform: translateY(10px);
            transition: all 0.3s ease;
            pointer-events: none;
            border: 1px solid #34495e;
            display: none;
        }
        
        .error-tooltip.show {
            opacity: 1;
            transform: translateY(0);
            display: block;
        }
        
        .error-tooltip::before {
            content: '';
            position: absolute;
            top: -8px;
            left: 50%;
            transform: translateX(-50%);
            width: 0;
            height: 0;
            border-left: 8px solid transparent;
            border-right: 8px solid transparent;
            border-bottom: 8px solid #2c3e50;
        }
        
        .error-tooltip.arrow-top::before {
            top: auto;
            bottom: -8px;
            border-bottom: none;
            border-top: 8px solid #2c3e50;
        }
        
        .error-tooltip.arrow-left::before {
            left: 20px;
            transform: none;
        }
        
        .error-tooltip.arrow-right::before {
            left: auto;
            right: 20px;
            transform: none;
        }
        
        .tooltip-header {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid #34495e;
        }
        
        .severity-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
            text-transform: uppercase;
            margin-right: 10px;
            letter-spacing: 0.5px;
        }
        
        .severity-minor {
            background: linear-gradient(135deg, #f39c12, #e67e22);
            color: white;
            box-shadow: 0 2px 4px rgba(243, 156, 18, 0.3);
        }
        
        .severity-major {
            background: linear-gradient(135deg, #e74c3c, #c0392b);
            color: white;
            box-shadow: 0 2px 4px rgba(231, 76, 60, 0.3);
        }
        
        .severity-critical {
            background: linear-gradient(135deg, #8e44ad, #9b59b6);
            color: white;
            box-shadow: 0 2px 4px rgba(142, 68, 173, 0.3);
        }
        
        .tooltip-category {
            font-size: 12px;
            color: #bdc3c7;
            font-weight: 500;
        }
        
        .tooltip-section {
            margin-bottom: 8px;
        }
        
        .tooltip-label {
            font-weight: bold;
            color: #3498db;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 3px;
            display: block;
        }
        
        .tooltip-value {
            color: #ecf0f1;
            line-height: 1.3;
        }
        
        .tooltip-explanation {
            background: rgba(52, 73, 94, 0.5);
            padding: 8px;
            border-radius: 4px;
            margin-top: 8px;
            border-left: 3px solid #3498db;
        }
        
        .unmatchable-errors {
            margin-top: 30px;
            padding: 20px;
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
        }
        
        .unmatchable-errors h4 {
            color: #856404;
            margin-top: 0;
        }
        
        .unmatchable-error {
            background: white;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
        }
        
        .legend {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
            border: 1px solid #e9ecef;
        }
        
        .legend h3 {
            margin-top: 0;
            color: #495057;
        }
        
        .legend-item {
            display: inline-block;
            margin-right: 20px;
            margin-bottom: 10px;
        }
        
        .legend-color {
            display: inline-block;
            width: 20px;
            height: 20px;
            border-radius: 3px;
            margin-right: 8px;
            vertical-align: middle;
            border: 2px solid;
        }
        
        .minor-legend { background-color: #fff4e6; border-color: #f39c12; }
        .major-legend { background-color: #ffe6e6; border-color: #e74c3c; }
        .critical-legend { background-color: #f3e6ff; border-color: #8e44ad; }
        
        .tooltip {
            position: relative;
            cursor: help;
        }
        
        /* Summary Statistics Styles */
        .summary-stats {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .summary-stats h2 {
            margin-top: 0;
            margin-bottom: 20px;
            text-align: center;
            font-size: 24px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }
        
        .stat-card {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .stat-number {
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .stat-label {
            font-size: 12px;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .severity-breakdown, .category-breakdown {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            backdrop-filter: blur(10px);
        }
        
        .severity-breakdown h3, .category-breakdown h3 {
            margin-top: 0;
            margin-bottom: 15px;
            font-size: 16px;
        }
        
        .severity-bar {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .severity-label {
            min-width: 120px;
            font-size: 13px;
            display: flex;
            align-items: center;
        }
        
        .bar-container {
            flex: 1;
            height: 20px;
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            margin-left: 10px;
            overflow: hidden;
        }
        
        .bar {
            height: 100%;
            border-radius: 10px;
            transition: width 0.8s ease;
        }
        
        .minor-bar {
            background: linear-gradient(90deg, #f39c12, #e67e22);
        }
        
        .major-bar {
            background: linear-gradient(90deg, #e74c3c, #c0392b);
        }
        
        .critical-bar {
            background: linear-gradient(90deg, #8e44ad, #9b59b6);
        }
        
        .category-list {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 8px;
        }
        
        .category-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255,255,255,0.1);
            padding: 8px 12px;
            border-radius: 5px;
            font-size: 13px;
        }
        
        .category-name {
            font-weight: 500;
        }
        
        .category-count {
            background: rgba(255,255,255,0.2);
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }
            
            .document-content {
                padding: 15px;
            }
            
            .document-content-columns {
                grid-template-columns: 1fr;
                gap: 20px;
                padding: 15px;
            }
            
            .metadata span {
                display: block;
                margin-bottom: 5px;
            }
            
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
            
            .severity-label {
                min-width: 100px;
                font-size: 12px;
            }
            
            .category-list {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>MQM Error Visualization</h1>
        
        {{SUMMARY_STATS}}
        
        <div class="legend">
            <h3>Error Severity Legend</h3>
            <div class="legend-item">
                <span class="legend-color minor-legend"></span>
                <span>Minor Errors</span>
            </div>
            <div class="legend-item">
                <span class="legend-color major-legend"></span>
                <span>Major Errors</span>
            </div>
            <div class="legend-item">
                <span class="legend-color critical-legend"></span>
                <span>Critical Errors</span>
            </div>
            <p><em>Hover over highlighted text to see error details</em></p>
        </div>
        
        {{DOCUMENT_SECTIONS}}
    </div>
    
    <script>
        let tooltip = null;
        let tooltipTimeout = null;
        
        function createTooltip() {
            if (!tooltip) {
                tooltip = document.createElement('div');
                tooltip.className = 'error-tooltip';
                document.body.appendChild(tooltip);
            }
            return tooltip;
        }
        
        function showErrorTooltip(element, event) {
            clearTimeout(tooltipTimeout);
            
            const tooltip = createTooltip();
            
            try {
                // Get error data from separate attributes
                const severity = element.dataset.severity;
                const errorSpan = element.dataset.errorSpan || '';
                const explanation = element.dataset.explanation || '';
                const category = element.dataset.category || 'N/A';
                const type = element.dataset.type || 'N/A';
            
            // Build tooltip content
            let content = `
                <div class="tooltip-header">
                    <span class="severity-badge severity-${severity}">${severity}</span>
                    <span class="tooltip-category">${category} • ${type}</span>
                </div>
            `;
            
            if (errorSpan) {
                content += `
                    <div class="tooltip-section">
                        <span class="tooltip-label">Error Span</span>
                        <div class="tooltip-value">"${errorSpan}"</div>
                    </div>
                `;
            }
            
            if (explanation) {
                content += `
                    <div class="tooltip-explanation">
                        <span class="tooltip-label">Explanation</span>
                        <div class="tooltip-value">${explanation}</div>
                    </div>
                `;
            }
            
            tooltip.innerHTML = content;
            
            // Position tooltip
            const rect = element.getBoundingClientRect();
            
            // Set initial position to get dimensions
            tooltip.style.visibility = 'hidden';
            tooltip.style.display = 'block';
            tooltip.style.left = '0px';
            tooltip.style.top = '0px';
            
            // Get tooltip dimensions after content is set
            const tooltipRect = tooltip.getBoundingClientRect();
            
            let left = rect.left + window.scrollX + (rect.width / 2) - (tooltipRect.width / 2);
            let top = rect.top + window.scrollY - tooltipRect.height - 10;
            
            // Adjust if tooltip goes off screen
            tooltip.classList.remove('arrow-top', 'arrow-left', 'arrow-right');
            
            if (left < 10) {
                left = 10;
                tooltip.classList.add('arrow-left');
            }
            if (left + tooltipRect.width > window.innerWidth - 10) {
                left = window.innerWidth - tooltipRect.width - 10;
                tooltip.classList.add('arrow-right');
            }
            if (top < 10) {
                top = rect.bottom + window.scrollY + 10;
                tooltip.classList.add('arrow-top');
            }
            
            tooltip.style.left = left + 'px';
            tooltip.style.top = top + 'px';
            tooltip.style.visibility = 'visible';
            
            // Show tooltip with animation
            setTimeout(() => {
                tooltip.classList.add('show');
            }, 10);
            
            } catch (error) {
                console.error('Error showing tooltip:', error);
                console.error('Element severity:', element.dataset.severity);
                console.error('Element text:', element.textContent);
            }
        }
        
        function hideErrorTooltip() {
            tooltipTimeout = setTimeout(() => {
                if (tooltip) {
                    tooltip.classList.remove('show');
                }
            }, 100);
        }
        
        function moveErrorTooltip(event) {
            // Optional: Update tooltip position on mouse move for better UX
            // Currently keeping it fixed to avoid jitter
        }
        
        // Enhanced interaction
        document.addEventListener('DOMContentLoaded', function() {
            // Add click functionality for mobile devices
            const errorHighlights = document.querySelectorAll('.error-highlight');
            
            errorHighlights.forEach(function(element) {
                element.addEventListener('click', function(e) {
                    e.preventDefault();
                    if (window.innerWidth <= 768) { // Mobile devices
                        showErrorTooltip(element, e);
                        setTimeout(() => hideErrorTooltip(), 3000); // Auto-hide after 3s on mobile
                    }
                });
            });
            
            // Hide tooltip when clicking elsewhere
            document.addEventListener('click', function(e) {
                if (!e.target.classList.contains('error-highlight')) {
                    hideErrorTooltip();
                }
            });
            
            // Hide tooltip on scroll
            window.addEventListener('scroll', hideErrorTooltip);
        });
    </script>
</body>
</html>'''


def create_visualization(results_file: str, doc_info_file: str, output_file: str):
    """
    Create HTML visualization from MQM evaluation results.
    
    Args:
        results_file: Path to results JSONL file
        doc_info_file: Path to document info JSONL file  
        output_file: Path to output HTML file
    """
    visualizer = MQMHTMLVisualizer()
    results, doc_info = visualizer.load_data(results_file, doc_info_file)
    visualizer.generate_html(results, doc_info, output_file)
    print(f"HTML visualization saved to: {output_file}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python html_visualizer.py <results_file> <doc_info_file> <output_file>")
        sys.exit(1)
    
    create_visualization(sys.argv[1], sys.argv[2], sys.argv[3])