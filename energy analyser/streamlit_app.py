# Energy Analyzer Project
# Author: Krish Garg

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

# File path
FILE_NAME = "energy_data.csv"

# Create a sample dataset if not found
if not os.path.exists(FILE_NAME):
    data = {
        "Appliance": ["AC", "Refrigerator", "Washing Machine", "TV", "Heater", "Computer"],
        "Energy_Consumed(kWh)": [450, 300, 150, 120, 400, 200],
        "Usage_Hours": [8, 24, 2, 5, 6, 8],
        "Cost_per_kWh": [8, 8, 8, 8, 8, 8]
    }
    df = pd.DataFrame(data)
    df.to_csv(FILE_NAME, index=False)
    print("Sample dataset created: energy_data.csv")

# Load dataset
df = pd.read_csv(FILE_NAME)

def show_menu():
    print("\n--- Energy Analyzer Menu ---")
    print("1. View Energy Data")
    print("2. Total and Average Energy Consumption")
    print("3. Top Energy Consuming Appliances")
    print("4. Energy Cost Estimation")
    print("5. Detect Anomalies (Unusual Usage)")
    print("6. Linear Regression Trend (Usage Prediction)")
    print("7. Exit")

# Main loop
while True:
    show_menu()
    choice = input("\nEnter your choice (1-7): ")

    if choice == '1':
        print("\nEnergy Consumption Data:\n")
        print(df.to_string(index=False))

    elif choice == '2':
        total_energy = df["Energy_Consumed(kWh)"].sum()
        avg_energy = df["Energy_Consumed(kWh)"].mean()
        print(f"\nTotal Energy Consumed: {total_energy} kWh")
        print(f"Average Energy Consumed: {avg_energy:.2f} kWh")

        plt.figure(figsize=(8,5))
        plt.bar(df["Appliance"], df["Energy_Consumed(kWh)"], color="orange")
        plt.title("Energy Consumption per Appliance")
        plt.xlabel("Appliance")
        plt.ylabel("Energy Consumed (kWh)")
        plt.show()

    elif choice == '3':
        top_appliances = df.sort_values(by="Energy_Consumed(kWh)", ascending=False).head(3)
        print("\nTop 3 Energy Consuming Appliances:\n")
        print(top_appliances.to_string(index=False))

        plt.figure(figsize=(8,5))
        plt.bar(top_appliances["Appliance"], top_appliances["Energy_Consumed(kWh)"], color="red")
        plt.title("Top 3 Energy Consuming Appliances")
        plt.xlabel("Appliance")
        plt.ylabel("Energy (kWh)")
        plt.show()

    elif choice == '4':
        df["Estimated_Cost"] = df["Energy_Consumed(kWh)"] * df["Cost_per_kWh"]
        print("\nEstimated Energy Costs:\n")
        print(df[["Appliance", "Energy_Consumed(kWh)", "Estimated_Cost"]])

        total_cost = df["Estimated_Cost"].sum()
        print(f"\nTotal Energy Cost: ₹{total_cost:.2f}")

    elif choice == '5':
        mean_usage = df["Energy_Consumed(kWh)"].mean()
        std_usage = df["Energy_Consumed(kWh)"].std()
        threshold = mean_usage + 1.5 * std_usage

        anomalies = df[df["Energy_Consumed(kWh)"] > threshold]
        if anomalies.empty:
            print("\nNo unusual energy usage detected.")
        else:
            print("\nAnomaly Detected! High Energy Usage:\n")
            print(anomalies.to_string(index=False))

    elif choice == '6':
        X = df["Usage_Hours"].values.reshape(-1, 1)
        y = df["Energy_Consumed(kWh)"].values
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        plt.figure(figsize=(8,5))
        plt.scatter(X, y, color='blue', label='Actual')
        plt.plot(X, y_pred, color='red', label='Regression Line')
        plt.title("Linear Regression: Energy vs Usage Hours")
        plt.xlabel("Usage Hours")
        plt.ylabel("Energy Consumed (kWh)")
        plt.legend()
        plt.show()

    elif choice == '7':
        print("Exiting Energy Analyzer. Goodbye!")
        break

    else:
        print("Invalid choice. Please try again.")