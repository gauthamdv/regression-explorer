import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


class DataPreprocessor:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.data = self.df.copy()
        self.label_encoders = {}
        self.standard_scalers = {}
        self.minmax_scalers = {}
        self.num_cols = None
        self.cat_cols = None
        self.date_cols = None
        self.missing = None
        self.missing_values = None
        self.total_rows = None
        self.cardinality_dict = None

    def data_explore(self):
        self.update_glob()
        print("\nInformation")
        print(self.df.info())
    
        print("\nMissing Values")
        print(self.missing_values)

        print('\nNumeric Data Summary')
        print(self.df.describe(include=[np.number]).T)
    
        print('\nNon-Numeric Data')
        print(self.df.describe(include=['object', 'category', 'datetime64[ns]']))
    
        print('\nUnique Values per Column')
        print(self.df.nunique().sort_values(ascending=True).head(30))
        
        dup_count = self.df.duplicated().sum()
        print(f'\nNumber of Duplicate Values : {dup_count}')
        if dup_count:
            self.df = self.df.drop_duplicates().reset_index(drop=True)

    
    def update_glob(self):
        self.num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = self.df.select_dtypes(include=['category', 'object']).columns.tolist()
        self.date_cols = self.df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
        self.missing = self.df.isnull().sum()
        self.missing_values = self.missing[self.missing > 0].sort_values(ascending=False)
        self.total_rows = len(self.df)
        
        
    def imputation_missing_num(self, threshold=50):
        self.update_glob()
        if len(self.missing_values) > 0:
            plt.figure(figsize=(10,4))
            plt.bar(self.missing_values.index, self.missing_values)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Missing count')
            plt.title('Missing values per column')
            plt.show()
        else:
            print("No missing values found.")      
        
        missing_pct = (self.missing / len(self.df)) * 100
        print(missing_pct[missing_pct > 0].sort_values(ascending=False))
        cols_to_drop = missing_pct[missing_pct > threshold].index
        
        if len(cols_to_drop) > 0:
            print(f"\nThe following columns have more than {threshold}% missing values:")
            print(list(cols_to_drop))
            user_choice = input("Do you want to drop these columns? (y/n): ").strip().lower()
            
            if user_choice == "y":
                self.df.drop(columns=cols_to_drop, inplace=True)
                print(f"Dropped columns: {list(cols_to_drop)}")
            else:
                print("Skipping column drop as per user choice.")
        else:
            print(f"No columns found with more than {threshold}% missing values.")
            
        for col in self.num_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna(self.df[col].median())
                
        self.update_glob()
            
            
    def imputation_missing_cat(self, threshold=50): 
        self.update_glob()
        for col in self.cat_cols:
            cardinality_pct = (self.df[col].nunique() / self.total_rows) * 100

            if cardinality_pct > threshold:
                print(f"\nColumn: '{col}' - Unique values: {self.df[col].nunique()} ({cardinality_pct:.1f}%) exceeds threshold of {threshold}%")

                while True:
                    choice = input(f"Do you want to drop the column '{col}'? (y/n): ").strip().lower()
                    if choice in ['y', 'n']:
                        break
                    else:
                        print("Please type 'y' or 'n'.")
                
                if choice == 'y':
                    self.df.drop(columns=[col], inplace=True)
                    print(f"Column '{col}' dropped. New shape: {self.df.shape}")
                else:
                    print(f"Column '{col}' retained.")
            else:
                print(f"\nColumn: '{col}' - Unique values: {self.df[col].nunique()} ({cardinality_pct:.1f}%) is under threshold. Skipping.")
        
        self.update_glob()
        for col in self.cat_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        self.update_glob()
                    
                    
    def encode_cat(self):
        self.update_glob()
        print(self.df.head())
        print(f"Categorical Columns: {self.cat_cols}")
        
        for col in self.cat_cols:
            n_unique = self.df[col].nunique()
            unique_pct = n_unique / self.total_rows * 100
        
            if unique_pct > 50:
                suggestion_text = f"High cardinality ({n_unique} unique values, {unique_pct:.1f}% of rows). Skip recommended."
            elif n_unique == 2:
                suggestion_text = f"Binary column ({n_unique} unique values). Label Encoding recommended."
            elif n_unique <= 10:
                suggestion_text = f"Few categories ({n_unique} unique values). One-Hot Encoding recommended."
            else:
                suggestion_text = f"{n_unique} unique values. Label Encoding recommended to avoid too many columns."
        
            print(f"\nColumn '{col}': {suggestion_text}")
        
            while True:
                choice = input(f"Choose encoding for '{col}' - One-Hot (o), Label (l), Skip (s): ").strip().lower()
                if choice in ['o', 'l', 's']:
                    break
                else:
                    print("Invalid input. Please type 'o', 'l', or 's'.")
        
            if choice == 'o':
                self.df = pd.get_dummies(self.df, columns=[col], prefix=col)
                print(f"Applied One-Hot Encoding to '{col}'.")
        
            elif choice == 'l':
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
                print(f"Applied Label Encoding to '{col}'.")
        
            elif choice == 's':
                print(f"Skipped column '{col}'.")    
        self.update_glob()
                
    
    def scale_num(self):
        self.update_glob()
        print("Numeric columns detected:", self.num_cols)

        for col in self.num_cols:
            while True:
                choice = input(f"\nColumn '{col}': Choose scaling - Standardize (s), Normalize (n), Skip (k): ").strip().lower()
                if choice in ['s', 'n', 'k']:
                    break
                else:
                    print("Invalid input. Please type 's', 'n', or 'k'.")
        
            if choice == 's':
                scaler = StandardScaler()
                self.df[col] = scaler.fit_transform(self.df[[col]])
                self.standard_scalers[col] = scaler 
                print(f"Standardized column '{col}'.")
        
            elif choice == 'n':
                scaler = MinMaxScaler()
                self.df[col] = scaler.fit_transform(self.df[[col]])
                self.minmax_scalers[col] = scaler
                print(f"Normalized column '{col}'.")

            elif choice == 'k':
                print(f"Skipped column '{col}'.")
                
        self.update_glob()


    def cardinality_plot(self):
        self.update_glob()
        self.cardinality_dict = {col: (self.df[col].nunique() / self.total_rows) * 100 for col in self.num_cols}
        cols = list(self.cardinality_dict.keys())
        cardinality_pct = list(self.cardinality_dict.values())
        
        plt.figure(figsize=(10, 5))
        plt.plot(cols, cardinality_pct, marker='o', linestyle='-', color='b')
        
        for i, val in enumerate(cardinality_pct):
            plt.text(i, val + 0.5, f"{val:.1f}%", ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.ylabel('Normalized Cardinality (%)')
        plt.title('Normalized Cardinality of Numeric Columns')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
        
    
    def outlier_handle(self, threshold_pct = 20, cardinality_threshold = 50):
        self.update_glob()
        for col in self.df.columns:
            normalized_cardinality = self.cardinality_dict.get(col, 0)
            outlier_pct = None
            condition_fails = False
            
            if col in self.num_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_count = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                outlier_pct = (outlier_count / self.total_rows) * 100
                
                if outlier_pct > threshold_pct or normalized_cardinality > cardinality_threshold:
                    condition_fails = True
            else:
                if normalized_cardinality > cardinality_threshold:
                    condition_fails = True
    
            print(f"\nColumn: '{col}':")
            if outlier_pct is not None:
                print(f"Outlier percentage is {outlier_pct:.1f}% (threshold {threshold_pct}%).")
            else:
                print("Outlier percentage is N/A (categorical column).")
            print(f"Normalized cardinality is {normalized_cardinality:.1f}% (threshold {cardinality_threshold}%).")
    
            if condition_fails:
                while True:
                    choice = input(f"Do you want to drop the column '{col}'? (y/n): ").strip().lower()
                    if choice in ['y', 'n']:
                        break
                    else:
                        print("Please type 'y' or 'n'.")
                
                if choice == 'y':
                    self.df.drop(columns=[col], inplace=True)
                    print(f"Column '{col}' dropped. New shape: {self.df.shape}")
                else:
                    print(f"Column '{col}' retained.")
            else:
                print(f"Column '{col}' is fine. Skipping without prompt.")
        self.update_glob()
                

    def plot_comparision(self):
        self.update_glob()
        numeric_old = self.data.select_dtypes(include=[float, int]).columns.tolist()
        numeric_new = self.df.select_dtypes(include=[float, int]).columns.tolist()
        
        common_numeric = list(set(numeric_old) & set(numeric_new))
        
        changed_numeric = [col for col in common_numeric 
                           if not self.data[col].equals(self.df[col])]
        
        print("Columns before preprocessing:", numeric_old)
        print("Columns after preprocessing:", numeric_new)
        print("Numeric columns that changed during preprocessing:", changed_numeric)
        
        scaler = MinMaxScaler()
        
        DATASET_norm = self.data[changed_numeric].copy()
        DATASET_norm[changed_numeric] = scaler.fit_transform(self.data[changed_numeric])
        
        df_norm = self.df[changed_numeric].copy()
        df_norm[changed_numeric] = scaler.fit_transform(self.df[changed_numeric])
        
        plt.figure(figsize=(12, 6))
        
        for col in changed_numeric:
            plt.hist(DATASET_norm[col], bins=30, density=True, alpha=0.5, linestyle='--', edgecolor='black', label=f'{col} - Old')
            plt.hist(df_norm[col], bins=30, density=True, alpha=0.5, linestyle='-', edgecolor='black', label=f'{col} - Clean')
        
        plt.title("Normalized Numeric Feature Signatures (Only Changed Features)")
        plt.xlabel("Normalized Value")
        plt.ylabel("Density")
        plt.legend()
        plt.show()
        
        
    def save_file(self, path = None):
        if path: 
            self.df.to_csv(path, index=False)
        else:
            print('Please enter a Valid path')
        
    
if __name__ == "__main__":
    global load_path, save_path
    
    load_path = "datasets/Housing.csv"
    save_path = "datasets/Housing-Clean-i.csv"
    
    proc = DataPreprocessor(load_path)
    
    proc.data_explore()
    
    print("\n--- Handling Numerical Missing Data ---")
    proc.imputation_missing_num()
    
    print("\n--- Handling Non-Numerical Missing Data ---")
    proc.imputation_missing_cat()
    
    print("\n--- Encoding Categorical Data ---")
    proc.encode_cat()
    
    print("\n--- Scaling Numerical Data ---")
    proc.scale_num()
    
    print("\n--- Plotting Cardinaloty to check possible outliers ---")
    proc.cardinality_plot()
    
    print("\n--- Handling outliers ---")
    proc.outlier_handle()
    
    print("\n--- Comparision between Original and Preprocessed Data ---")
    proc.plot_comparision()
    
    print(proc.df.head())
    
    save = input("\nDo you wish to save this data (y/n) \n")    
    if save == 'y':
        proc.save_file(save_path)
        print(f"File saved successfully at: {save_path}")
    else:
        print("File not saved, exiting...")
    
    

            
   
