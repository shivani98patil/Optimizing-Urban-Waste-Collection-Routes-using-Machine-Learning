import pandas as pd
import numpy as np
import random
import os

# Step 1: Define parameters for synthetic data
num_areas = 50  # Number of city areas
days = 30  # Number of days for data simulation

# Step 2: Generate synthetic data
np.random.seed(42)
area_ids = range(1, num_areas + 1)
dates = pd.date_range("2024-01-01", periods=days)
traffic_conditions = ["Low", "Medium", "High"]
weather_conditions = ["Sunny", "Rainy", "Cloudy"]
road_types = ["Highway", "Residential", "Commercial"]
waste_types = ["Organic", "Recyclable", "General"]
priority_levels = ["High", "Medium", "Low"]

# Create data
data = []
for date in dates:
    for area in area_ids:
        waste_generated = round(random.uniform(50, 300), 2)  # Random waste generated in kg
        traffic = random.choice(traffic_conditions)  # Random traffic condition
        population_density = round(random.uniform(500, 2000), 2)  # Random population density per km²
        weather = random.choice(weather_conditions)  # Random weather condition
        distance_to_next = round(random.uniform(1, 5), 2)  # Random distance to the next area in km
        collection_frequency = random.randint(1, 7)  # Collection frequency in days
        fuel_efficiency = round(random.uniform(8, 15), 2)  # Fuel efficiency in km/l
        vehicle_capacity = random.randint(500, 2000)  # Vehicle capacity in kg
        traffic_density = round(random.uniform(10, 100), 2)  # Traffic density in vehicles/km
        road_type = random.choice(road_types)  # Road type
        waste_type = random.choice(waste_types)  # Waste type
        priority_level = random.choice(priority_levels)  # Area priority
        travel_time = round(random.uniform(5, 30), 2)  # Travel time in minutes

        data.append([
            date, area, waste_generated, traffic, population_density, weather,
            distance_to_next, collection_frequency, fuel_efficiency, vehicle_capacity,
            traffic_density, road_type, waste_type, priority_level, travel_time
        ])

# Step 3: Convert data to a DataFrame
columns = [
    "Date", "AreaID", "WasteGenerated(kg)", "TrafficCondition", "PopulationDensity",
    "WeatherCondition", "DistanceToNextArea(km)", "CollectionFrequency(days)",
    "FuelEfficiency(km/l)", "VehicleCapacity(kg)", "TrafficDensity(vehicles/km)",
    "RoadType", "WasteType", "PriorityLevel", "TravelTime(minutes)"
]
synthetic_data = pd.DataFrame(data, columns=columns)

# Step 4: Save the dataset locally
output_dir = "datasets"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "synthetic_waste_collection_data.csv")
synthetic_data.to_csv(output_file, index=False)

print(f"Synthetic dataset saved locally at: {output_file}")

# Step 5: Display sample data
print(synthetic_data.head())