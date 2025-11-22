#!/bin/bash

# MeGuMa Evaluation Script for GPT-5-2025-08-07
# This script evaluates all .jsonl files containing "random" in MeGuMa submissions

echo "=========================================="
echo "MeGuMa Evaluation - GPT-5-2025-08-07 (Random) - V2T Template"
echo "=========================================="
echo ""

# Configuration
MODEL="gpt-5-chat-latest"
BASE_OUTPUT_DIR="output/$MODEL-v2T-random"
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

# Find all .jsonl files containing "random" in their names
echo "Scanning for files containing 'random' in: $SUBMISSIONS_DIR"
RANDOM_FILES=($(find "$SUBMISSIONS_DIR" -name "*.jsonl" -name "*random*" | sort))

if [ ${#RANDOM_FILES[@]} -eq 0 ]; then
    echo "❌ ERROR: No .jsonl files containing 'random' found in $SUBMISSIONS_DIR"
    exit 1
fi

echo ""
echo "Found ${#RANDOM_FILES[@]} files to evaluate:"
for file in "${RANDOM_FILES[@]}"; do
    echo "  📄 $(basename "$file")"
done

echo ""
read -p "Do you want to proceed with evaluating these files? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Evaluation cancelled by user."
    exit 0
fi

echo ""
echo "Starting batch evaluation..."
echo "=========================================="

# Create base output directory
mkdir -p "$BASE_OUTPUT_DIR"

# Counter for tracking progress
TOTAL_FILES=${#RANDOM_FILES[@]}
CURRENT_FILE=0
SUCCESSFUL_EVALUATIONS=0
FAILED_EVALUATIONS=0

# Process each file
for INPUT_FILE in "${RANDOM_FILES[@]}"; do
    CURRENT_FILE=$((CURRENT_FILE + 1))
    
    echo ""
    echo "=========================================="
    echo "Processing file $CURRENT_FILE of $TOTAL_FILES"
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
    
    # Determine target document key based on language direction in filename
    if [[ "$INPUT_BASENAME" == *".enzh."* ]]; then
        TARGET_KEY="zh"  # English to Chinese
        echo "🔤 Language direction: EN→ZH (target key: zh)"
    elif [[ "$INPUT_BASENAME" == *".zhen."* ]]; then
        TARGET_KEY="en"  # Chinese to English
        echo "🔤 Language direction: ZH→EN (target key: en)"
    else
        echo "❌ ERROR: Cannot determine language direction from filename: $INPUT_BASENAME"
        FAILED_EVALUATIONS=$((FAILED_EVALUATIONS + 1))
        continue
    fi
    
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
    echo "Progress: $CURRENT_FILE/$TOTAL_FILES files processed"
    echo "Successful: $SUCCESSFUL_EVALUATIONS, Failed: $FAILED_EVALUATIONS"
done

echo ""
echo "=========================================="
echo "Batch Evaluation Summary"
echo "=========================================="
echo "Total files processed: $TOTAL_FILES"
echo "Successful evaluations: $SUCCESSFUL_EVALUATIONS"
echo "Failed evaluations: $FAILED_EVALUATIONS"
echo ""

if [ $SUCCESSFUL_EVALUATIONS -gt 0 ]; then
    echo "✅ Results saved to: $BASE_OUTPUT_DIR"
    echo ""
    echo "Output structure:"
    if [ -d "$BASE_OUTPUT_DIR" ]; then
        find "$BASE_OUTPUT_DIR" -type d -name "MeGuMa.*" | head -10 | while read dir; do
            echo "  📁 $(basename "$dir")"
        done
        if [ $(find "$BASE_OUTPUT_DIR" -type d -name "MeGuMa.*" | wc -l) -gt 10 ]; then
            echo "  ... and more"
        fi
    fi
fi

if [ $FAILED_EVALUATIONS -eq 0 ]; then
    echo ""
    echo "🎉 All evaluations completed successfully!"
else
    echo ""
    echo "⚠️  Some evaluations failed. Check the logs above for details."
fi

echo ""
echo "MeGuMa evaluation script completed."