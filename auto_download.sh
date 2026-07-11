#!/bin/bash
# Automatically download Kaggle results when notebook completes

NOTEBOOK="kitastro/hybrid-latent-state-language-model"
OUTPUT_DIR="kaggle_output"
LOG_FILE="kaggle_auto_download.log"

echo "$(date): Starting automatic monitoring for $NOTEBOOK" | tee -a $LOG_FILE
echo "This will check status every 2 minutes and download when complete..." | tee -a $LOG_FILE

while true; do
    status=$(kaggle kernels status $NOTEBOOK 2>&1)
    echo "$(date): $status" | tee -a $LOG_FILE
    
    if [[ "$status" == *"complete"* ]]; then
        echo "" | tee -a $LOG_FILE
        echo "✅ Notebook completed successfully!" | tee -a $LOG_FILE
        echo "Downloading results..." | tee -a $LOG_FILE
        
        # Create output directory
        mkdir -p $OUTPUT_DIR
        
        # Download all outputs
        kaggle kernels output $NOTEBOOK -p $OUTPUT_DIR 2>&1 | tee -a $LOG_FILE
        
        # Show what we got
        echo "" | tee -a $LOG_FILE
        echo "📦 Downloaded files:" | tee -a $LOG_FILE
        find $OUTPUT_DIR -type f -exec ls -lh {} \; | tee -a $LOG_FILE
        
        # Extract if it's a zip
        cd $OUTPUT_DIR
        if ls *.zip 1> /dev/null 2>&1; then
            echo "" | tee -a ../$LOG_FILE
            echo "📂 Extracting zip files..." | tee -a ../$LOG_FILE
            for zipfile in *.zip; do
                unzip -q "$zipfile"
                echo "  Extracted: $zipfile" | tee -a ../$LOG_FILE
            done
            cd ..
            echo "" | tee -a $LOG_FILE
            echo "📁 Final structure:" | tee -a $LOG_FILE
            find $OUTPUT_DIR -type f | head -20 | tee -a $LOG_FILE
        else
            cd ..
        fi
        
        echo "" | tee -a $LOG_FILE
        echo "✅ Download complete! Results in $OUTPUT_DIR/" | tee -a $LOG_FILE
        exit 0
        
    elif [[ "$status" == *"error"* ]] || [[ "$status" == *"Error"* ]]; then
        echo "" | tee -a $LOG_FILE
        echo "❌ Notebook failed! Downloading logs..." | tee -a $LOG_FILE
        mkdir -p $OUTPUT_DIR
        kaggle kernels output $NOTEBOOK -p $OUTPUT_DIR 2>&1 | tee -a $LOG_FILE
        echo "" | tee -a $LOG_FILE
        echo "📋 Last 50 lines of log:" | tee -a $LOG_FILE
        if [ -f "$OUTPUT_DIR/"*.log ]; then
            tail -50 $OUTPUT_DIR/*.log | tee -a $LOG_FILE
        fi
        exit 1
        
    elif [[ "$status" == *"cancel"* ]]; then
        echo "⚠️ Notebook was cancelled" | tee -a $LOG_FILE
        exit 1
    fi
    
    # Wait 2 minutes before checking again
    sleep 120
done
