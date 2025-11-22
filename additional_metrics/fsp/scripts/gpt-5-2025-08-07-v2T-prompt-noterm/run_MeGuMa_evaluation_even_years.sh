#!/bin/bash

# MeGuMa Evaluation Script for GPT-5-2025-08-07 - EVEN YEARS ONLY (ZH→EN)
# This script evaluates .jsonl files containing "noterm" for even years (2016, 2018, 2020, 2022, 2024)

echo "=========================================="
echo "MeGuMa Evaluation - GPT-5-2025-08-07 (NoTerm) - V2T Template - EVEN YEARS (ZH→EN)"
echo "=========================================="
echo ""

# Configuration
MODEL="gpt-5-chat-latest"
BASE_OUTPUT_DIR="output/$MODEL-v2T-noterm"
SUBMISSIONS_DIR="data/submissions/MeGuMa"
TEST_DATA_DIR="data/test_data_filtered_terms/track2"
TEMPLATE_PATH="templates/fsp_judge_v2_T.jinja"
TEMPERATURE="0.0"
SEGMENT_SIZE="3"
BASE_URL="https://api.openai.com/v1"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Base output directory: $BASE_OUTPUT_DIR"
echo "  Submissions directory: $SUBMISSIONS_DIR"
echo "  Test data directory: $TEST_DATA_DIR"
echo "  Template path: $TEMPLATE_PATH"
echo "  Temperature: $TEMPERATURE"
echo "  Segment size: $SEGMENT_SIZE"
echo "  Base URL: $BASE_URL"
echo ""

# Check if submissions directory exists
if [ ! -d "$SUBMISSIONS_DIR" ]; then
    echo "❌ ERROR: Submissions directory not found: $SUBMISSIONS_DIR"
    exit 1
fi

# Check if test data directory exists
if [ ! -d "$TEST_DATA_DIR" ]; then
    echo "❌ ERROR: Test data directory not found: $TEST_DATA_DIR"
    exit 1
fi

# Check if .env file exists for API key
if [ ! -f ".env" ]; then
    echo "⚠️  WARNING: .env file not found in project root"
    echo "Make sure OPENAI_API_KEY is set in your environment"
else
    echo "✅ .env file found"
fi

# Define even years (ZH→EN direction)
EVEN_YEARS=(2016 2018 2020 2022 2024)

# Find .jsonl files containing "noterm" for even years only
echo "Scanning for even year files (ZH→EN) containing 'noterm' in: $SUBMISSIONS_DIR"
NOTERM_FILES=()

for year in "${EVEN_YEARS[@]}"; do
    # Look for files matching pattern: MeGuMa.YEAR.zhen.noterm.jsonl
    file_pattern="$SUBMISSIONS_DIR/MeGuMa.$year.zhen.noterm.jsonl"
    if [ -f "$file_pattern" ]; then
        NOTERM_FILES+=("$file_pattern")
        echo "  ✅ Found: $(basename "$file_pattern")"
    else
        echo "  ❌ Missing: $(basename "$file_pattern")"
    fi
done

if [ ${#NOTERM_FILES[@]} -eq 0 ]; then
    echo "❌ ERROR: No .jsonl files for even years found in $SUBMISSIONS_DIR"
    echo "Expected files:"
    for year in "${EVEN_YEARS[@]}"; do
        echo "  - MeGuMa.$year.zhen.noterm.jsonl"
    done
    exit 1
fi

echo ""
echo "Found ${#NOTERM_FILES[@]} even year files to evaluate:"
for file in "${NOTERM_FILES[@]}"; do
    echo "  📄 $(basename "$file")"
done

echo ""
read -p "Do you want to proceed with evaluating these even year files? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Evaluation cancelled by user."
    exit 0
fi

echo ""
echo "Starting batch evaluation for even years (ZH→EN)..."
echo "=========================================="

# Create base output directory
mkdir -p "$BASE_OUTPUT_DIR"

# Counter for tracking progress
TOTAL_FILES=${#NOTERM_FILES[@]}
CURRENT_FILE=0
SUCCESSFUL_EVALUATIONS=0
FAILED_EVALUATIONS=0

# Process each file
for INPUT_FILE in "${NOTERM_FILES[@]}"; do
    CURRENT_FILE=$((CURRENT_FILE + 1))
    
    echo ""
    echo "=========================================="
    echo "Processing even year file $CURRENT_FILE of $TOTAL_FILES"
    echo "=========================================="
    
    # Extract filename without extension and path
    INPUT_BASENAME=$(basename "$INPUT_FILE" .jsonl)
    
    # Create unique output directory for this file
    OUTPUT_DIR="$BASE_OUTPUT_DIR/$INPUT_BASENAME"
    EVAL_RUNNAME="$INPUT_BASENAME"
    
    echo "Current file: $INPUT_FILE"
    echo "Output directory: $OUTPUT_DIR"
    echo "Evaluation run name: $EVAL_RUNNAME"
    echo ""
    
    # Check if input file exists (should exist, but double-check)
    if [ ! -f "$INPUT_FILE" ]; then
        echo "❌ ERROR: Input file not found: $INPUT_FILE"
        FAILED_EVALUATIONS=$((FAILED_EVALUATIONS + 1))
        continue
    fi
    
    echo "✅ Input file found: $INPUT_FILE"
    
    # For even years, all files are ZH→EN direction
    TARGET_KEY="en"  # Chinese to English
    echo "🔤 Language direction: ZH→EN (target key: en)"
    
    echo ""
    echo "Starting evaluation for: $(basename "$INPUT_FILE")"
    echo ""
    
    # Run the evaluation
    python code/evaluate_mt.py \
        --input-file "$INPUT_FILE" \
        --output-dir "$BASE_OUTPUT_DIR" \
        --eval-runname "$EVAL_RUNNAME" \
        --test-data-dir "$TEST_DATA_DIR" \
        --model "$MODEL" \
        --template-path "$TEMPLATE_PATH" \
        --temperature "$TEMPERATURE" \
        --segment-size "$SEGMENT_SIZE" \
        --base-url "$BASE_URL" \
        --target-document-key "$TARGET_KEY" \
        --verbose
    
    # Check the exit status
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Evaluation completed successfully for: $(basename "$INPUT_FILE")"
        SUCCESSFUL_EVALUATIONS=$((SUCCESSFUL_EVALUATIONS + 1))
        
        # Proceed to MQM computation
        echo ""
        echo "🎯 Proceeding to MQM score computation..."
        
        # MQM Computation Configuration
        MQM_INPUT_FILE="$OUTPUT_DIR/results.jsonl"
        MQM_OUTPUT_FILE="$OUTPUT_DIR/mqm_scores.jsonl"
        MINOR_WEIGHT="1.0"
        MAJOR_WEIGHT="2.0"
        CRITICAL_WEIGHT="5.0"
        
        echo "MQM Configuration:"
        echo "  Input file: $MQM_INPUT_FILE"
        echo "  Output file: $MQM_OUTPUT_FILE"
        echo "  Error weights: Minor=$MINOR_WEIGHT, Major=$MAJOR_WEIGHT, Critical=$CRITICAL_WEIGHT"
        
        # Check if results file exists
        if [ ! -f "$MQM_INPUT_FILE" ]; then
            echo "❌ ERROR: Results file not found: $MQM_INPUT_FILE"
            echo "Evaluation may have failed. Skipping MQM computation."
            continue
        fi
        
        # Set document info file for HTML visualization
        DOC_INFO_FILE="$OUTPUT_DIR/results_doc_info.jsonl"
        
        # Run the MQM computation with HTML visualization
        python code/mqm_score_computation.py \
            --input-file "$MQM_INPUT_FILE" \
            --output-file "$MQM_OUTPUT_FILE" \
            --minor-weight "$MINOR_WEIGHT" \
            --major-weight "$MAJOR_WEIGHT" \
            --critical-weight "$CRITICAL_WEIGHT" \
            --create-html-viz \
            --doc-info-file "$DOC_INFO_FILE" \
            --verbose
        
        # Check MQM computation exit status
        if [ $? -eq 0 ]; then
            echo "✅ MQM computation completed successfully for: $(basename "$INPUT_FILE")"
            
            # Show created files
            if [ -d "$OUTPUT_DIR" ]; then
                echo "📁 Results saved to: $OUTPUT_DIR"
                echo "   Files created:"
                ls -la "$OUTPUT_DIR/" | grep -E '\.(jsonl|xlsx|html)' | awk '{print "     " $9}'
            fi
        else
            echo "❌ MQM computation failed for: $(basename "$INPUT_FILE")"
        fi
        
    else
        echo ""
        echo "❌ Evaluation failed for: $(basename "$INPUT_FILE")"
        FAILED_EVALUATIONS=$((FAILED_EVALUATIONS + 1))
    fi
    
    echo ""
    echo "Progress: $CURRENT_FILE/$TOTAL_FILES even year files processed"
    echo "Successful: $SUCCESSFUL_EVALUATIONS, Failed: $FAILED_EVALUATIONS"
done

echo ""
echo "=========================================="
echo "Even Years Batch Evaluation Summary"
echo "=========================================="
echo "Total even year files processed: $TOTAL_FILES"
echo "Successful evaluations: $SUCCESSFUL_EVALUATIONS"
echo "Failed evaluations: $FAILED_EVALUATIONS"
echo ""

if [ $SUCCESSFUL_EVALUATIONS -gt 0 ]; then
    echo "✅ Results saved to: $BASE_OUTPUT_DIR"
    echo ""
    echo "Output structure (even years only):"
    if [ -d "$BASE_OUTPUT_DIR" ]; then
        find "$BASE_OUTPUT_DIR" -type d -name "MeGuMa.201[68].zhen.*" -o -name "MeGuMa.202[024].zhen.*" | head -10 | while read dir; do
            echo "  📁 $(basename "$dir")"
        done
    fi
fi

if [ $FAILED_EVALUATIONS -eq 0 ]; then
    echo ""
    echo "🎉 All even year evaluations completed successfully!"
else
    echo ""
    echo "⚠️  Some even year evaluations failed. Check the logs above for details."
fi

echo ""
echo "Even years MeGuMa evaluation script completed."