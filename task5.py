import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Set up paths
data_dir = 'path/to/food-101/images'  # Update with your path to Food-101 images
calorie_data_path = 'path/to/calories.csv'  # You will need to create this CSV

# Create a CSV file for calorie information (You may need to adjust this)
# Here is a sample mapping, you can create a more comprehensive one
food_calories = {
    'apple': 95,
    'banana': 105,
    'burger': 354,
    'cake': 257,
    # Add more mappings...
}

# Create a DataFrame
food_items = []
calories = []
for food, cal in food_calories.items():
    food_items.append(food)
    calories.append(cal)

calorie_df = pd.DataFrame({'food_item': food_items, 'calories': calories})
calorie_df.to_csv(calorie_data_path, index=False)

# Load calorie data
calorie_data = pd.read_csv(calorie_data_path)

# Create a mapping of food items to their calorie content
calorie_dict = dict(zip(calorie_data['food_item'], calorie_data['calories']))

# Prepare image data and labels
def load_and_preprocess_images(data_dir, classes, target_size=(224, 224)):
    images = []
    labels = []
    
    for food in classes:
        food_path = os.path.join(data_dir, food)
        for img in os.listdir(food_path):
            img_path = os.path.join(food_path, img)
            img_array = cv2.imread(img_path)
            img_array = cv2.resize(img_array, target_size)
            img_array = img_array / 255.0  # Normalize
            images.append(img_array)
            labels.append(calorie_dict.get(food, 0))  # Default to 0 if food not found
    
    return np.array(images), np.array(labels)

# Load images and calories
classes = os.listdir(data_dir)  # This will get the food class names
X, y = load_and_preprocess_images(data_dir, classes)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base model layers

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1)  # Single output for calorie prediction
])

# Compile the model
model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test MAE: {test_mae}')

# Prediction function
def predict_calories(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize to fit model input
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    predicted_calories = model.predict(img)
    return predicted_calories[0][0]

# Example usage
image_to_predict = 'path/to/example_food_image.jpg'  # Replace with your image path
estimated_calories = predict_calories(image_to_predict)
print(f'Estimated Calories: {estimated_calories}')
