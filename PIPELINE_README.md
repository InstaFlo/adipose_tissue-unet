# Complete Adipose U-Net Pipeline

This automated pipeline runs the complete end-to-end workflow for adipose tissue segmentation, from raw data to final validation results.

## 🚀 Quick Start

```bash
# Run the complete pipeline
./run_complete_pipeline.sh
```

## 📋 What It Does

### Phase 1: Dataset Building
- **Cleans and rebuilds** the entire `_build` folder
- **Enhanced split ratios**: 60/20/20 (train/val/test) - improved from 60/25/15
- **Stain normalization**: SYBR Gold + Eosin color correction applied during tiling
- **Quality overlays**: Generated for visual QA of annotations
- **Fat segmentation**: Target masks with bubbles subtraction and morphological cleanup

### Phase 2: Model Training  
- **2-phase training**: 50 epochs (frozen encoder) + 100 epochs (full fine-tuning)
- **Proven architecture**: 2-channel softmax U-Net with dilated convolutions
- **Timestamped checkpoints**: No more overwrites! Each run gets unique directory
- **Best model selection**: Automatically saves best weights by validation Dice score
- **Enhanced callbacks**: Improved monitoring and early stopping

### Phase 3: Full Validation
- **Comprehensive evaluation**: Both validation and test datasets
- **Test Time Augmentation**: Enabled for better accuracy
- **Performance organization**: Results sorted by Dice score buckets
- **Rich reporting**: Detailed visualizations and metrics

## 📊 Expected Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| **Dataset Building** | 10-30 min | Depends on data size and processing |
| **Model Training** | 2-4 hours | 150 total epochs with monitoring |
| **Full Validation** | 30-60 min | TTA and comprehensive reporting |
| **Total** | 3-5 hours | Complete end-to-end pipeline |

## 📁 Output Structure

```
~/Data_for_ML/Meat_Luci_Tulane/
├── _build/
│   ├── dataset/           # Train/val/test splits (60/20/20)
│   ├── masks/             # Generated binary masks
│   ├── overlays/          # QA overlays (bubbles=blue, fat=yellow, muscle=green)
│   └── evaluation/        # Validation results and reports
│
checkpoints/
└── YYYYMMDD_HHMMSS_adipose_sybreosin_1024_finetune/
    ├── phase1_best.weights.h5        # Best Phase 1 model
    ├── phase2_best.weights.h5        # Best Phase 2 model
    ├── weights_best_overall.weights.h5  # FINAL BEST MODEL
    ├── phase1_training.log           # Phase 1 training logs
    └── phase2_training.log           # Phase 2 training logs
```

## 🔧 Configuration

The script uses these default settings:

**Dataset Building:**
- Target: `fat` masks with `bubbles` subtraction
- Tile size: 1024x1024 pixels
- Quality filtering: white ratio, blur detection, mask density
- Stain normalization: Enabled (SYBR Gold + Eosin)

**Training:**
- Loss: BCE + Dice (proven combination)
- Normalization: Z-score (proven for this dataset)
- Augmentation: Moderate (rotation, flip, elastic)
- Batch size: 2 (recommended for 1024x1024)

**Validation:**
- TTA: Enabled for better accuracy
- Metrics: Dice, IoU, Hausdorff distance
- Visualization: Performance bucketing

## 🛠️ Customization

To modify settings, edit the individual scripts:

```bash
# Custom dataset options
python build_dataset.py --val-ratio 0.15 --test-ratio 0.25 --no-stain-normalize

# Custom training options  
python train_adipose_unet_2.py --batch-size 4 --epochs-phase1 75 --use-focal-loss

# Custom validation options
python run_full_validation.py --no-tta --threshold 0.3
```

## 📝 Monitoring Progress

The script provides:
- **Real-time logging** with timestamps and colors
- **Progress indicators** for each phase
- **Error handling** with clear failure messages
- **Detailed log file** saved as `pipeline_YYYYMMDD_HHMMSS.log`

## 🎯 Key Improvements

This pipeline includes several enhancements over previous versions:

### Dataset Building
✅ **Better split ratios** (60/20/20 vs 60/25/15)  
✅ **Confidence filtering** (accepts scores ≥2)  
✅ **Stain normalization** during tiling  
✅ **Quality overlays** for annotation QA  

### Training  
✅ **Timestamped checkpoints** (no overwrites)  
✅ **Improved callbacks** (monitor Dice, not loss)  
✅ **Better early stopping** (reduced patience)  
✅ **Load best Phase 1** before Phase 2  

### Validation
✅ **TTA enabled** by default  
✅ **Performance bucketing** for result organization  
✅ **Comprehensive reporting** with visualizations  

## 🚨 Prerequisites

- **Conda environment**: `adipose-tf2` must be created and configured
- **Data directory**: `~/Data_for_ML/Meat_Luci_Tulane` with proper structure
- **Dependencies**: All packages installed in the conda environment
- **GPU**: Recommended for training (will use CPU if not available)

## 🔍 Troubleshooting

**Common Issues:**

1. **Environment not found**: Ensure `conda env list` shows `adipose-tf2`
2. **Data not found**: Check that `~/Data_for_ML/Meat_Luci_Tulane` exists
3. **Permission errors**: Run `chmod +x run_complete_pipeline.sh`
4. **Memory issues**: Reduce batch size in training script
5. **GPU errors**: The pipeline will fall back to CPU automatically

**Check logs**: All output is saved to timestamped log files for debugging.

## 📈 Expected Results

With ~1000 tiles and proper data quality:
- **Training Dice**: 0.85+ (validation set)
- **Test Dice**: 0.80+ (independent test set)  
- **Processing time**: 3-5 hours total
- **Output**: Ready-to-use model and comprehensive evaluation

Ready to run your complete adipose tissue segmentation pipeline! 🧬
