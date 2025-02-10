# **Student Exam Performance Predictor**  

This project is a **Machine Learning-based Student Exam Performance Predictor** that estimates a student's Math Score based on various input features. It follows a **modular coding approach** to ensure clean, scalable, and maintainable code.  

## **Project Structure**  
The project is organized into multiple modules to separate concerns and enhance reusability.  

## **How to Run the Project**  
1. **Download** all project files into your local repository.  
2. **Create a virtual environment** in the same local repository named **venv** using the following command (ensure Anaconda Navigator is installed):  
   ```sh
   conda create -p venv python==3.8 -y
   ```  
3. **Activate** the environment by running the following command in the terminal:  
   ```sh
   conda activate venv/
   ```  
4. **Install dependencies**: Open the terminal and run:  
   ```sh
   pip install -r requirements.txt
   ```  
   *(This process will create necessary folders—this is expected.)*  
5. **Run data ingestion and model training**:  
   ```sh
   python src/components/data_ingestion.py
   ```
   *(This process will create necessary folders—this is expected.)*
   *(This step may take some time, as it involves data ingestion, model training, hyperparameter tuning, and selecting the best-performing model.)*  
6. **Start the application**:  
   ```sh
   python app.py
   ```  
   *(Two links will be generated—click on either one to access the application.)*  
7. **You're all set! 🚀 Your project is now running!**  

## **Key Features**  
✔ **User-Friendly Interface** – A clean and interactive UI for seamless predictions.  
✔ **Modular Structure** – Ensures better code maintainability and reusability.  
✔ **Machine Learning Model** – Accurately predicts student scores based on input attributes.  

This project is a well-structured and reusable **machine learning application**! 🚀  

