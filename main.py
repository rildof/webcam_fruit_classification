from textblob import Word
import cv2
import numpy as np
import tensorflow as tf
from tf_keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tf_keras.applications.mobilenet_v2 import decode_predictions
import argostranslate.package
import argostranslate.translate
from PIL import ImageFont, ImageDraw, Image
from ModelClass import FruitModel

#CONFIG_PATH = "fruit_classifier_v2_dropout.h5"
CONFIG_PATH = "fruit_classifier_fruit360.h5"
# Load MobileNetV2 model pre-trained on ImageNet
model = FruitModel(CONFIG_PATH)

# Set the desired window size
window_width = 1280
window_height = 720

# Function to translate text using Argos Translate
def translate_text(text, from_lang='en', to_lang='pt'):
    translated_text = text  # Default to the original text in case translation fails
    try:
        # Get installed translation languages
        installed_languages = argostranslate.translate.get_installed_languages()
        from_language = next(filter(lambda x: x.code == from_lang, installed_languages))
        to_language = next(filter(lambda x: x.code == to_lang, installed_languages))
        # Translate text
        translated_text = argostranslate.translate.translate(text, from_lang, to_lang)
    except Exception as e:
        print(f"Translation error: {e}")
    
    return translated_text

# Function to draw the confidence bar at the bottom of the frame
def draw_confidence_bar(frame, confidence, x=50, y=None, bar_width=1000, bar_height=40):
    if y is None:
        y = window_height - 100  # Place bar at the bottom of the window
    filled_width = int(bar_width * (confidence/100.0))  # Width filled based on confidence
    # Draw the background of the bar (gray)
    cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (50, 50, 50), -1)
    # Draw the filled part of the bar (green)
    cv2.rectangle(frame, (x, y), (x + filled_width, y + bar_height), (0, 255, 0), -1)
    # Draw the confidence percentage text above the bar
    #make any value from confidence from 0 to 100
    confidence = 100 if confidence > 100 else confidence
    cv2.putText(frame, f'{confidence:.2f}%', (x + bar_width + 20, y + bar_height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def draw_text_with_pillow(frame, text, position=(50, 50), font_path="C:\\Windows\\Fonts\\arial.ttf", font_size=35):
    # Convert the OpenCV image (frame) to a PIL image
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Create a drawing object
    draw = ImageDraw.Draw(frame_pil)
    
    # Load the font and draw the text
    font = ImageFont.truetype(font_path, font_size)
    # get color from text
    if text == "Nenhuma fruta específica detectada": #Red Text
        draw.text(position, text, font=font, fill=(255, 0, 0))
    else:
        draw.text(position, text, font=font, fill=(0, 255, 0))  # White text
    
    # Convert the PIL image back to an OpenCV image
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    return frame

# Function to classify specific fruits
def classify_fruit():
    # Initialize webcam
    cap = cv2.VideoCapture(1)
    
    # Set the desired resolution for the window
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)

    if not cap.isOpened():
        print("Error accessing the camera")
        return
    
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        #frame = cv2.imread("banana.jpeg")
        if not ret:
            print("Failed to grab frame")
            break
        
        # Resize the frame for the model input (224x224) for MobileNetV2 processing
        #img = cv2.resize(frame, (224, 224))
        #img = np.expand_dims(img, axis=0)
        #img = preprocess_input(img)  # Pre-process the image for MobileNetV2
        # Get predictions from the model
        fruit_found = model.predict(frame)
        # decoded_preds = decode_predictions(predictions, top=50)[0]  # Get top 3 predictions

        # # Filter the predictions to show only specific fruits (banana, lemon)
        # fruit_found = None
        # for pred in decoded_preds:
        #     fruit_name = pred[1]  # Class label (fruit name)
        #     confidence = pred[2] * 100  # Confidence level as percentage
            
        #     fruit_found = (fruit_name, confidence)

        frame = cv2.resize(frame, (window_width, window_height))
        label, confidence = fruit_found
        print(label,confidence)
        # If a fruit is detected, translate and display the result
        if True:
            translated_label = translate_text(label, 'en', 'pt')
            translated_text = f"Fruta: {translated_label}"

            # Draw the translated text (with non-ASCII characters) onto the frame using Pillow
            frame = draw_text_with_pillow(frame, translated_text, position=(50, 50), font_size=50)
            
            # Draw the confidence bar at the bottom
            frame = draw_text_with_pillow(frame, "Confiança:", position=(50, window_height - 150), font_size=35)
            draw_confidence_bar(frame, confidence, x=50)

        else:
            # Draw the "no fruit detected" message using Pillow (in Portuguese)
            frame = draw_text_with_pillow(frame, "Nenhuma fruta específica detectada", position=(50, 50), font_size=50)
        
        # Show the frame with predictions and confidence bar
        cv2.imshow('Classificador de Frutas com Tradução e Barra de Confiança - Pressione "q" para sair', frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Run the classifier with real-time translation and confidence bar
classify_fruit()