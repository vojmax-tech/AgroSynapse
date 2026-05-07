# UI Specification – AgroSynapse Dashboard

## Overview
The AgroSynapse dashboard provides agricultural survey intelligence through interactive visualizations, predictive analytics, and data exploration tools. It supports any CSV survey data and adapts to different field schemas automatically.

## Core Pages

### Dashboard
- **Hero Section**: Platform title and description
- **Key Metrics Cards**: Total farmers, average yield, average soil pH, total area
- **Charts**:
  - Farmer distribution map (by district/state)
  - Yield distribution histogram
  - Crop type breakdown (pie/bar chart)
  - Soil pH distribution
  - Fertilizer usage adoption rates
- **Pipeline Status**: Last run time and success indicator

### Model Performance
- **Metrics Display**: RMSE, MAE, R², training sample size
- **Feature Importance**: Bar chart of top predictive features
- **Prediction vs Actual**: Scatter plot with trend line
- **Model Readiness**: Qualitative assessment (experimental/improving/strong)

### Data Explorer
- **Data Table**: Paginated view of processed survey data
- **Column Statistics**: Summary stats for each field (mean, median, unique values)
- **Correlation Matrix**: Heatmap of numeric field correlations
- **Export Options**: Download filtered data as CSV

### Uploads
- **File Upload**: Drag-and-drop CSV upload interface
- **Pipeline Execution**: Run full ETL + training pipeline
- **Progress Indicators**: Real-time pipeline status updates
- **Error Handling**: Clear error messages for invalid data

### Settings
- **App Configuration**: Dashboard preferences (bins, limits, animations)
- **Data Paths**: Configure input/output file locations
- **Filters**: Global district/crop filters that apply across pages

## Global Features
- **Sidebar Navigation**: Clean page switching with platform branding
- **Global Filters**: District, crop, and category filters that persist across pages
- **Theme Support**: Light/dark mode adaptation
- **Responsive Design**: Works on desktop and tablet devices
- **Data Refresh**: Manual pipeline re-run with status feedback

## Visual Style
- **Clean Interface**: Data-focused layout with minimal clutter
- **Consistent Typography**: Poppins and Space Grotesk fonts
- **Agronomic Color Palette**: Greens and earth tones
- **Interactive Elements**: Hover tooltips, clickable legends
- **Accessibility**: High contrast ratios and keyboard navigation

## Data Processing
- **Automatic Field Detection**: Intelligent column type inference
- **Flexible Schema**: Adapts to any agricultural survey format
- **Real-time Updates**: Dashboard refreshes after pipeline runs
- **Error Recovery**: Graceful handling of missing or malformed data
