#!/bin/bash
# Wait for notebook completion and download results

NOTEBOOK="kitastro/hybrid-latent-state-language-model"
OUTPUT_DIR="kaggle_output"

echo "Monitoring Kaggle notebook: $NOTEBOOK"
echo "URL: https://www.kaggle.com/code/$NOTEBOOK"
echo ""

while true; do
    status=$(kaggle kernels status $NOTEBOOK)
    timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    
    echo "[$timestamp] $status"
    
    if [[ "$status" == *"complete"* ]]; then
        echo ""
        echo "✅ Notebook completed successfully!"
        echo "Downloading results..."
        echo ""
        
        # Clean old output
        rm -rf $OUTPUT_DIR
        mkdir -p $OUTPUT_DIR
        
        # Download all outputs
        kaggle kernels output $NOTEBOOK -p $OUTPUT_DIR
        
        # Show what we got
        echo ""
        echo "📦 Downloaded files:"
        find $OUTPUT_DIR -type f -exec ls -lh {} \;
        
        # Check for models
        echo ""
        if [ -d "$OUTPUT_DIR/experiments" ]; then
            echo "🎯 Model files found:"
            find $OUTPUT_DIR/experiments -name "*.pt" -exec ls -lh {} \;
        else
            echo "⚠️  No experiments directory found"
        fi
        
        break
        
    elif [[ "$status" == *"error"* ]]; then
        echo ""
        echo "❌ Notebook failed with error!"
        echo "Downloading logs..."
        
        rm -rf $OUTPUT_DIR
        mkdir -p $OUTPUT_DIR
        kaggle kernels output $NOTEBOOK -p $OUTPUT_DIR
        
        # Show last part of log
        if ls $OUTPUT_DIR/*.log 1> /dev/null 2>&1; then
            echo ""
            echo "📋 Last 100 lines of log:"
            tail -100 $OUTPUT_DIR/*.log
        fi
        
        break
    fi
    
    # Wait 2 minutes before checking again
    sleep 120
done
