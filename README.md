# AI-TOOL - Professional Data Analytics Framework

**Version:** 2.2  
**Status:** Production-Ready with Claude Code Integration  
**Credits:** $250 Claude Code Budget  
**Framework Modules:** 28  

---

## Overview

AI-TOOL is a professional-grade data analytics framework integrating 28 specialized modules for:
- **R Statistical Analysis** (12 modules) - Regression, Classification, CART
- **Python Data Science** (2 modules) - Cleaning, Ecosystem Management
- **React Web Analytics** (6 modules) - Interactive Dashboards
- **Environment-Aware Routing** - Auto-detection & Dual-path strategies

### Quality Standards
- ✅ Zero AI Slop
- ✅ Complete Citations (ISLR, ESL, Penn State)
- ✅ Reproducible Methodology
- ✅ Production-Ready Code

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/aztectech82-cell/AI-TOOL.git
cd AI-TOOL

# Install Python dependencies
pip install -r requirements.txt

# Test installation
python main.py
```

### Claude Code Integration

```bash
# Initialize Claude Code
claude-code init

# Start building with AI assistance
claude-code "Add multiple linear regression module"
claude-code "Create data cleaning pipeline"
claude-code "Build interactive dashboard for emissions data"
```

---

## Framework Capabilities

### R Statistical Analysis
- Multiple Linear Regression (MLR)
- k-Nearest Neighbors (kNN) Classification
- Logistic Regression (Binary Classification)
- CART Decision Trees
- Classification Evaluation Metrics
- Professional Data Cleaning

### Python Data Science
- Production Data Cleaning Pipeline
- Package Ecosystem Management
- Statistical Computing
- Data Preprocessing & Validation

### React Web Analytics
- Interactive Dashboards
- Predictive Maintenance Visualizations
- Real-time Analytics
- Emissions Lab Data Processing

---

## Using Your $250 Claude Code Credit

### Starter Commands

**1. Data Analysis Tasks**
```bash
claude-code "Analyze this CSV file and provide summary statistics with visualizations"
claude-code "Build a logistic regression model for binary classification"
claude-code "Create a kNN classifier with k-fold cross-validation"
```

**2. Code Generation**
```bash
claude-code "Generate R script for multiple linear regression with VIF analysis"
claude-code "Create Python data cleaning pipeline following MODULE_Data_Cleaning_Production_Python"
claude-code "Build React dashboard for vehicle emissions data"
```

**3. Debugging & Optimization**
```bash
claude-code "Debug this R script and fix package installation errors"
claude-code "Optimize this pandas DataFrame operation for better performance"
claude-code "Add error handling and logging to this analysis function"
```

**4. Documentation**
```bash
claude-code "Generate docstrings for all functions in main.py"
claude-code "Create a comprehensive tutorial for the kNN module"
claude-code "Write test cases for the data cleaning pipeline"
```

**5. Integration Tasks**
```bash
claude-code "Integrate Power BI connection for this Python script"
claude-code "Add API endpoint for exposing model predictions"
claude-code "Create Docker container for this analytics workflow"
```

---

## Project Structure

```
AI-TOOL/
├── main.py                  # Core analytics framework
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── .clauderc               # Claude Code configuration
├── data/                   # Data directory
│   ├── raw/               # Raw input data
│   └── processed/         # Cleaned data
├── scripts/               # Analysis scripts
│   ├── regression/        # R regression scripts
│   ├── classification/    # R/Python classification
│   └── visualization/     # React dashboards
├── modules/               # Framework modules
│   ├── r_modules/        # R statistical modules
│   ├── python_modules/   # Python modules
│   └── react_modules/    # React components
├── outputs/              # Analysis results
│   ├── reports/         # Generated reports
│   ├── plots/           # Visualizations
│   └── models/          # Saved models
└── tests/               # Unit tests
```

---

## Example Usage

### Python Analytics

```python
from main import AnalyticsFramework

# Initialize framework
framework = AnalyticsFramework()

# Load data
df = framework.load_data('data/emissions_data.csv')

# Generate statistics
stats = framework.basic_stats(df)
print(stats)

# Clean data
clean_df = framework.clean_data(df, strategy='mean')

# Correlation analysis
corr_results = framework.correlation_analysis(clean_df, threshold=0.7)

# Export results
framework.export_results(clean_df, 'outputs/cleaned_data.csv')
```

### R Statistical Analysis

```r
# Load framework modules
source('modules/r_modules/MODULE_Multiple_Linear_Regression_Framework.R')

# Run regression analysis
results <- run_mlr_analysis(
  data = emissions_data,
  response = 'NOx',
  predictors = c('RPM', 'Load', 'Temperature'),
  seed = 256
)

# Generate professional report
generate_mlr_report(results, output = 'outputs/regression_report.pdf')
```

---

## Module Reference

### Core Framework (2 modules)
1. Analytics_Master_Instructions_v2.0.md
2. MODULE_Environment_Aware_Analytics.md

### R Statistical Analysis (12 modules)
3. MASTER_MODULE_Regression_Assignment_Complete.R
4. Advanced_Regression_Module.md
5. MODULE_Academic_Regression_Report_Writing.R
6. MODULE_Multiple_Linear_Regression_Framework.R
7. MODULE_kNN_Classification_Framework.R
8. MODULE_Logistic_Regression_Complete.R
9. MODULE_CART_Decision_Trees_Complete.R
10. MODULE_Classification_Evaluation_Complete.R
11. MODULE_Data_Cleaning_Production_R.R
12. MODULE_R_Package_Ecosystem.md
13. ONE_BUTTON_Complete_Homework_Automation.txt
14. MODULE_RStudio_Troubleshooting_UltraBeginner.md

### Python Data Science (2 modules)
15. MODULE_Python_Package_Ecosystem.md
16. MODULE_Data_Cleaning_Production_Python.py

### React Web Analytics (6 modules)
17-22. [React dashboard modules]

### Documentation (8 modules)
23-30. [Learning guides, quick starts, framework updates]

---

## References

### Statistical Learning
- **ISLR:** Introduction to Statistical Learning - http://www-bcf.usc.edu/~gareth/ISL/
- **ESL:** Elements of Statistical Learning - https://hastie.su.domains/ElemStatLearn/
- **Penn State STAT 501:** https://online.stat.psu.edu/stat501/

### Programming & Tools
- **R Documentation:** https://www.rdocumentation.org/
- **Python scikit-learn:** https://scikit-learn.org/
- **pandas:** https://pandas.pydata.org/
- **React Documentation:** https://react.dev/
- **Recharts:** https://recharts.org/

### Academic Standards
- **APA Style:** https://apastyle.apa.org/
- **ASA Guidelines:** https://www.amstat.org/

---

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Quality
```bash
# Format code
black main.py

# Lint code
pylint main.py

# Type checking
mypy main.py
```

### Contributing
1. Fork repository
2. Create feature branch: `git checkout -b feature/new-module`
3. Commit changes: `git commit -m 'Add new analytics module'`
4. Push: `git push origin feature/new-module`
5. Create Pull Request

---

## Troubleshooting

### Common Issues

**R Package Installation Errors:**
```r
# Use binary packages
install.packages("package_name", type = "binary")
```

**Python Environment Issues:**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Claude Code Connection:**
```bash
# Verify API key
claude-code config get api-key

# Reinitialize if needed
claude-code init --force
```

---

## License

MIT License - See LICENSE file for details

---

## Contact & Support

- **GitHub:** [@aztectech82-cell](https://github.com/aztectech82-cell)
- **Repository:** [AI-TOOL](https://github.com/aztectech82-cell/AI-TOOL)
- **Claude Code Docs:** https://docs.claude.ai/docs/claude-code

---

## Acknowledgments

Built with professional analytics frameworks and best practices:
- Anthropic Claude API & Claude Code
- 28-Module Analytics Master System
- Evidence-based statistical methodologies
- Enterprise-grade quality standards

---

**Version:** 2.2  
**Last Updated:** November 2025  
**Status:** ✅ Production-Ready  
**Quality:** Zero AI Slop, Professional Grade
