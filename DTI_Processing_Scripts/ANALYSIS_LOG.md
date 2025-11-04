# WM Proportion Analysis - Session Log

**Date:** 2025-11-04
**Objective:** Address reviewer feedback on DTI analysis regarding potential confounding by white matter tissue composition changes over time

---

## Context

### Reviewer Feedback (Page 258)
> "The analysis performed did not control for potential changes in total white matter per ROI as the brain expands and returns to baseline following DC. The results presented in figure 6.17 (decrease in FA but no change in MD) could be explained by an underlying shift in total white matter over time. This is because FA is very different between grey and white matter, while MD is very similar for the two tissue types and therefore it wouldn't be as sensitive to changes in the white-grey matter content. Please quantify the total white matter within the ROI used for the longitudinal analysis to rule that out as a potential explanation for the changes observed."

### Key Concern
The reviewer hypothesizes that:
1. As brain herniates/resolves after decompressive craniectomy, total WM within ROIs may change
2. If WM proportion decreases over time, this could cause FA to decrease (since grey matter has lower FA)
3. MD wouldn't change much (since grey and white matter have similar MD values)
4. Therefore, observed FA changes might reflect tissue composition shifts rather than true microstructural changes

---

## Data Files Available

### Primary Data
- **`DTI_Processing_Scripts/results/all_wm_proportion_analysis_10x4vox_NEW_filtered.csv`**
  - Contains WM proportion data for all patients/timepoints
  - Columns: `patient_id`, `timepoint`, plus for each region (ant, post, baseline_ant, baseline_post) and ring (5, 6, 7):
    - `WM_count_{region}_ring_{N}`: Number of white matter voxels
    - `total_count_{region}_ring_{N}`: Total number of voxels in ROI
    - `WM_prop_{region}_ring_{N}`: WM proportion (WM_count / total_count)
  - Rings 5, 6, 7 are the ROIs used in longitudinal FA/MD analysis
  - 10x4vox configuration (10 rings, 4 voxel radius each)

### Supporting Data
- **`DTI_Processing_Scripts/results/all_metrics_10x4vox_NEW_filtered.csv`**
  - FA and MD values for whole ROI (includes grey + white matter)

- **`DTI_Processing_Scripts/results/all_metrics_10x4vox_NEW_filtered_wm.csv`**
  - FA and MD values extracted from white matter only

---

## Analysis Strategy (5-Part Plan)

### **Analysis 1: Test if WM proportion changes over time** ✓ SCRIPT CREATED
- **Null Hypothesis:** WM proportion does NOT change significantly over time
- **Method:** Linear Mixed Effects models disaggregated by region
  - Model: `WM_prop ~ timepoint + (1|patient_id)`
  - Separate models for: anterior, posterior, baseline_anterior, baseline_posterior
  - Use mean of rings 5, 6, 7 (the longitudinal analysis ROIs)
- **Interpretation:**
  - p ≥ 0.05: WM proportion is stable → **rules out tissue composition as confound**
  - p < 0.05: WM proportion changes → proceed to Analysis 2

### **Analysis 2: Test if WM proportion changes explain FA/MD patterns**
- Compare temporal trajectories:
  - Does WM proportion decrease in same pattern as FA?
  - If WM proportion stable/increases while FA decreases → contradicts reviewer's hypothesis
- Calculate correlations between change in WM proportion and change in FA/MD

### **Analysis 3: Control for WM proportion as covariate**
- Re-run FA/MD longitudinal LME models with WM proportion as covariate:
  - `FA ~ timepoint + WM_proportion + (1|patient_id)`
- If timepoint effect remains significant → FA changes are **independent** of tissue composition
- **This is the strongest statistical test**

### **Analysis 4: Compare craniectomy vs baseline ROIs**
- If reviewer's hypothesis is correct:
  - Craniectomy ROIs (ant/post) should show WM proportion changes due to herniation
  - Baseline ROIs (contralateral) should be stable
- Test interaction: `WM_prop ~ timepoint * ROI_type + (1|patient_id)`

### **Analysis 5: Effect size comparison**
- Quantify magnitude of changes:
  - How much does WM proportion change? (e.g., 5% decrease)
  - How much does FA change? (e.g., 30% decrease)
- Small WM proportion changes cannot explain large FA changes

---

## Files Created This Session

### 1. **`wm_proportion_longitudinal.py`** ✓ COMPLETE
- **Location:** `DTI_Processing_Scripts/wm_proportion_longitudinal.py`
- **Purpose:** Visualize WM proportion changes over time
- **Outputs:** Longitudinal trajectory plots for each region/ring
- **Status:** Working, legend modified to skip number 10 per user request

### 2. **`wm_proportion_lme_analysis.py`** ✓ CREATED (HPC ISSUES)
- **Location:** `DTI_Processing_Scripts/wm_proportion_lme_analysis.py`
- **Purpose:** Analysis 1 - Test null hypothesis via LME models
- **Issue:** Segmentation fault on HPC cluster due to rpy2 library conflicts
- **Solution:** User will run locally instead

### 3. **`wm_proportion_lme_analysis.R`** ⚠️ CREATION INTERRUPTED
- **Location:** Would be `DTI_Processing_Scripts/wm_proportion_lme_analysis.R`
- **Purpose:** Pure R version of Analysis 1 (no Python/rpy2 dependencies)
- **Status:** User interrupted creation, will handle locally

---

## Script Details: wm_proportion_lme_analysis.py

### What It Does
1. Loads `all_wm_proportion_analysis_10x4vox_NEW_filtered.csv`
2. For each region (ant, post, baseline_ant, baseline_post):
   - Calculates mean WM proportion across rings 5, 6, 7
   - Fits LME model: `WM_prop_mean ~ timepoint + (1|patient_id)`
   - Extracts p-value for timepoint effect
   - Generates trajectory plot (mean ± SEM with individual patient lines)
3. Creates summary table with p-values for all regions
4. Provides overall conclusion about tissue composition confounding

### Expected Outputs
- `DTI_Processing_Scripts/wm_proportion_lme_results/`
  - `lme_ant_rings_5-7.txt` - Detailed model output for anterior
  - `lme_post_rings_5-7.txt` - Detailed model output for posterior
  - `lme_baseline_ant_rings_5-7.txt` - Detailed model output for baseline anterior
  - `lme_baseline_post_rings_5-7.txt` - Detailed model output for baseline posterior
  - `wm_proportion_trajectory_ant.png` - Trajectory plot for anterior
  - `wm_proportion_trajectory_post.png` - Trajectory plot for posterior
  - `wm_proportion_trajectory_baseline_ant.png` - Trajectory plot for baseline anterior
  - `wm_proportion_trajectory_baseline_post.png` - Trajectory plot for baseline posterior
  - `wm_proportion_lme_summary.csv` - Summary table with all p-values

### Required R Packages
- `lme4` (for LME models)
- `lmerTest` (for p-values in LME models)

---

## Next Steps for New Claude Instance

### Immediate Actions
1. **Run Analysis 1 locally:**
   ```bash
   python DTI_Processing_Scripts/wm_proportion_lme_analysis.py
   ```
   - Check if R packages installed: `lme4`, `lmerTest`
   - Review outputs in `wm_proportion_lme_results/`

2. **Interpret Analysis 1 results:**
   - Check p-values in `wm_proportion_lme_summary.csv`
   - Look at trajectory plots to visualize patterns
   - Determine if WM proportion is stable or changing

### Conditional Next Steps

#### If WM proportion is STABLE (p ≥ 0.05 for all regions):
✓ **Mission accomplished!** Draft conclusion:
> "To address concerns about tissue composition confounding, we quantified WM proportion within the ROIs used for longitudinal analysis (rings 5-7). WM proportion did not significantly change over time in any region (all p ≥ 0.05, linear mixed effects models). This rules out changes in white-grey matter content as a potential explanation for the observed FA decreases."

#### If WM proportion CHANGES in some/all regions:
→ **Proceed to Analysis 2-5:**
1. **Analysis 2:** Calculate correlations between WM proportion changes and FA/MD changes
2. **Analysis 3:** Re-run FA/MD LME models with WM proportion as covariate (STRONGEST TEST)
3. **Analysis 4:** Compare craniectomy vs baseline ROI patterns
4. **Analysis 5:** Quantify effect sizes

---

## Technical Notes

### Timepoint Categories
Data uses standardized timepoint labels:
- `ultra-fast`: 0-2 days
- `fast`: 2-8 days
- `acute`: 8-42 days
- `3mo`: 42-179 days
- `6mo`: 179-278 days
- `12mo`: 278-540 days
- `24mo`: 540+ days

### ROI Configuration
- **10x4vox_NEW**: 10 concentric rings, 4 voxel radius each
- **Rings 5, 6, 7** used in longitudinal analysis (20-28 voxels from craniectomy site)
- **Regions:**
  - `ant`: Anterior craniectomy ROI
  - `post`: Posterior craniectomy ROI
  - `baseline_ant`: Anterior contralateral (control)
  - `baseline_post`: Posterior contralateral (control)

### Patient Cohorts
- LEGACY cohort: Numeric IDs (e.g., 12519, 13782)
- CENTER cohort: Alphanumeric IDs (e.g., 2ZFz639, 6DBs724)

---

## Key Questions to Answer

1. **Does WM proportion change over time?** (Analysis 1)
2. **If yes, does it correlate with FA/MD changes?** (Analysis 2)
3. **Do FA/MD changes remain significant after controlling for WM proportion?** (Analysis 3)
4. **Do craniectomy ROIs differ from baseline ROIs in WM proportion trajectory?** (Analysis 4)
5. **Are WM proportion changes large enough to explain FA changes?** (Analysis 5)

---

## References to Original Analysis

- Original DTI results showing FA decrease: Figure 6.17 (page 258)
- Original analysis scripts: `DTI_Processing_Scripts/dti_results_plotting_main.py`
- Harmonization: `DTI_Processing_Scripts/harmonisation.py`
- ROI extraction: `DTI_Processing_Scripts/roi_extract_wm.sh`

---

## Contact Points for Continuation

**Current status:** Analysis 1 script created but not yet run due to HPC cluster issues
**Blocker:** rpy2 segmentation fault on cluster
**Solution:** User running locally
**Waiting on:** Analysis 1 results to determine next steps

---

## End of Log

**Last updated:** 2025-11-04
**Session status:** Paused for local execution
**Resume point:** Run `wm_proportion_lme_analysis.py` and interpret results
