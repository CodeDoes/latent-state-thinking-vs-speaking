#!/bin/bash
echo "Monitoring Kaggle notebook execution..."
echo "URL: https://www.kaggle.com/code/kitastro/hybrid-latent-state-language-model"
echo ""

while true; do
    status=$(kaggle kernels status kitastro/hybrid-latent-state-language-model 2>&1)
    echo "$(date): $status"
    
    if [[ "$status" == *"complete"* ]] || [[ "$status" == *"error"* ]] || [[ "$status" == *"cancel"* ]]; then
        echo ""
        echo "========================================="
        if [[ "$status" == *"complete"* ]]; then
            echo "✅ Notebook completed successfully!"
            echo "Downloading results..."
            kaggle kernels output kitastro/hybrid-latent-state-language-model -p kaggle_output/
            echo "✅ Results downloaded to kaggle_output/"
            ls -lh kaggle_output/
        elif [[ "$status" == *"error"* ]]; then
            echo "❌ Notebook failed! Downloading logs..."
            kaggle kernels output kitastro/hybrid-latent-state-language-model -p kaggle_output/
            echo "Logs downloaded to kaggle_output/"
        else
            echo "⚠️ Notebook was cancelled"
        fi
        echo "========================================="
        break
    fi
    
    sleep 60  # Check every minute
done
