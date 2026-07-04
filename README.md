# SmartLender - AI-Powered Loan Eligibility Screening Platform

## 📌 Project Summary
**SmartLender** is an AI-powered loan eligibility prediction platform developed to automate the preliminary loan screening process using machine learning. Instead of manually reviewing every application, the system analyzes applicant information (such as income, employment status, credit history, loan amount, marital status, education, and property details) using trained machine learning models and predicts whether the applicant is eligible for a loan. 

Rather than replacing loan officers, Smart Lender acts as an intelligent decision-support tool that enables faster, more consistent, and more transparent lending decisions.

### ✨ Key Features
- **Instant loan eligibility prediction** with model confidence score
- **AI-based feature contribution analysis**
- **Real-time EMI & Debt-to-Income (DTI) calculation**
- **Batch prediction** through CSV upload
- **REST API support**
- **Interactive analytics dashboard & session-based history**
- **Responsive Web Interface** with Light/Dark Mode support

## 🏗️ Technical Architecture
The project follows a modular three-layer architecture:
1. **Machine Learning Layer:** Handles data preprocessing, feature engineering, and model training (Decision Tree, Random Forest, KNN, Logistic Regression, XGBoost). XGBoost is primarily used for its high accuracy.
2. **Backend Layer:** Implemented using **Flask**, managing HTTP requests, business logic, REST APIs, batch processing, and prediction generation.
3. **Presentation Layer:** Built with HTML5, CSS3, JavaScript, Jinja2 templates, and Bootstrap for a responsive user experience.

## 👥 Project Submitted By
**Shaik Shahil** - shaik.shahil9959@gmail.com

## 🚀 How to Run the Application

### Prerequisites
- Python 3.10 or higher installed and available in PATH
- `pip` package manager available

### Step 1: Navigate to the Project Folder
The complete source code of the application is located inside the `SmartLender_Project_codes` directory.
```bash
cd SmartLender_Project_codes
```

### Step 2: Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### Step 3: Install Project Dependencies
Install all required Python libraries.
```bash
pip install -r requirements.txt
```

### Step 4: Generate Machine Learning Model Files
Before launching the web application, execute the machine learning pipeline to generate and save the model artifacts into the `model/` directory.
```bash
python train_model.py
```
*This process takes about 30–60 seconds, loading the dataset, training models, selecting the best-performing one, and generating analytics.*

### Step 5: Launch the Flask Web Application
Start the server using:
```bash
python app.py
```

### Step 6: Open the Application
Visit the following URL in your web browser:
**http://127.0.0.1:5000**

## 🧪 Testing (Optional)
To run the automated test suite to ensure all backend functions, API endpoints, and prediction logic work correctly:
```bash
pytest tests/ -v
```

## 📂 Project Structure
- `1.Brainstroming and Ideation/` to `8. Project Demonstration/`: Documentation and project lifecycle phases.
- `SmartLender_Project_codes/`: The primary source code folder containing:
  - `app.py`: Flask web application.
  - `train_model.py`: ML pipeline (Data prep, training, model evaluation).
  - `data/`: Contains `train.csv` dataset.
  - `model/`: Stores generated model artifacts (`.pkl` files) and charts.
  - `templates/` & `static/`: Frontend HTML files and static assets (CSS/JS).
  - `tests/`: Unit and integration testing files.
  - `notebooks/`: Exploratory Data Analysis notebooks.
