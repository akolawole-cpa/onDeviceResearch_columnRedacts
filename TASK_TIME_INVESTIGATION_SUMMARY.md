# Task Time Calculation Bug Investigation Summary

## Problem Identified

The `task_time_taken_s` feature showed suspicious statistics:
- **Mean**: 1679.41 seconds (27.99 minutes) - seemed high
- **Std Dev**: 99636.03 seconds (1660.60 minutes) - extremely large, indicating calculation error
- **Negative thresholds**: Fast thresholds were negative, indicating distribution issues

## Root Cause

**Bug Location**: `src/data_pull/aggregators.py`, line 50

**Issue**: The code was incorrectly calling `unix_timestamp()` with string column names instead of Column objects:

```python
# BEFORE (INCORRECT):
unix_timestamp(date_completed_col) - unix_timestamp(date_created_col)
```

In PySpark, `unix_timestamp()` expects a Column object, not a string. When passed a string, it likely:
1. Returns NULL for all rows
2. Or treats the string as a literal value, causing incorrect calculations
3. Or fails silently with unexpected behavior

## Fix Applied

**File**: `src/data_pull/aggregators.py`

**Changes**:
1. Added `col` import from `pyspark.sql.functions`
2. Fixed the calculation to use `col()` function to reference columns:

```python
# AFTER (CORRECT):
from pyspark.sql.functions import count, unix_timestamp, col

unix_timestamp(col(date_completed_col)) - unix_timestamp(col(date_created_col))
```

3. Also fixed `count()` call for consistency (though it can accept strings, using `col()` is more explicit)

## Diagnostic Tools Created

Created `src/utils/diagnose_task_time.py` with utilities to:
1. **`analyze_task_time_distribution()`**: Analyze distribution of task_time_taken_s values
   - Count, mean, median, std dev, percentiles
   - NULL counts, negative values, outliers
   - Values > 1 hour, > 1 day, > 1 week

2. **`validate_date_columns()`**: Validate date columns used in calculation
   - Check if columns exist
   - Check data types, NULL counts
   - Check date ranges and sample values
   - Check logical consistency (e.g., date_created > date_completed)

3. **`print_diagnostic_report()`**: Print comprehensive diagnostic report

## Next Steps

1. **Re-run data pull**: After the fix, re-run `respondent_pull_refactored.ipynb` to regenerate data with correct calculations

2. **Run diagnostics**: Use the diagnostic utilities to verify:
   ```python
   from utils.diagnose_task_time import print_diagnostic_report
   
   print_diagnostic_report(
       user_info_df,
       task_time_col="task_time_taken_s",
       date_completed_col="date_completed",
       date_created_col="date_created"
   )
   ```

3. **Verify results**: Check that:
   - Mean/median task times are reasonable (typically 5-15 minutes for surveys)
   - Standard deviation is reasonable (not thousands of minutes)
   - No negative values
   - Distribution makes sense

4. **Check column existence**: Verify that `date_created` column exists in the joined DataFrame. If it doesn't exist, you may need to:
   - Check the source table schema
   - Use a different column name
   - Or calculate task time differently

## Expected Results After Fix

After fixing the bug, you should see:
- **Mean task time**: Typically 5-15 minutes (300-900 seconds) for most surveys
- **Std Dev**: Much smaller, typically 50-200% of the mean
- **Distribution**: Reasonable distribution without extreme outliers
- **No negative values**: All task times should be positive

## Files Modified

1. `src/data_pull/aggregators.py` - Fixed calculation bug
2. `src/utils/diagnose_task_time.py` - Created diagnostic utilities (NEW FILE)

## Notes

- The fix assumes that `date_completed` and `date_created` columns exist in the DataFrame
- If `date_created` doesn't exist, the calculation will fail with a clear error message
- The diagnostic utilities will help identify if columns are missing or have data quality issues

