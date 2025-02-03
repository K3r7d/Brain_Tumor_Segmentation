# TumorSeg Computer Vision Project

 The project aims to identify tumor regions accurately within Medical Images using advanced techniques.
 
## Project Setup Guide

### Environment Requirements
**Python Version**: 
- 3.12.x >= for Window 
- 3.9.x - 3.12.x for MacOS or Linux

### Development Environment Setup

#### Prerequisites
- Python
- pip
- git

#### Clone the Repository
To get started, first clone the repository:

```bash
git clone https://github.com/K3r7d/MLProject_forfun.git
cd MLProject_forfun
```

##### Windows
1. Install Python from official website
2. Open Command Prompt
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

##### macOS/Linux
```bash
# Install Python via Homebrew (macOS) or package manager (Linux)
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
### Project Structure
```
MLProject_forfun/
├── data/          # Raw and processed data
├── models/        # Trained models
├── notebooks/     # Jupyter notebooks
├── doc/           # Project documentation
├── src/           # Source code
└── test/          # Unit tests
```