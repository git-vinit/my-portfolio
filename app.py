from flask import Flask, render_template, request , jsonify
from lstm_model import create_lstm_model, train_lstm_model, predict_stock_price, get_stock_data, get_latest_price
import matplotlib.pyplot as plt
import io
import base64
import cv2
import os
import requests
from PIL import Image
import pytesseract
import pickle
import joblib





filename = 'pickle.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/projects')
def projects():
    return render_template('projects.html')


@app.route('/stock_prediction')
def stock_prediction():
    return render_template('stock_prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['symbol']
    
    # Get stock data
    data = get_stock_data(symbol)
    num_records = len(data)
    X_train = data[:-1]
    y_train = data[1:]
    
    # Create and train the LSTM model
    model = create_lstm_model()
    train_lstm_model(model, X_train, y_train)
    
    # Perform prediction
    X = data[-1:]
    predicted_price = predict_stock_price(model, X)
    
    # Get latest price for plotting
    latest_price = get_latest_price(symbol)
    
    # Plot the graph
    plt.plot(range(num_records), data, label='Actual')
    plt.plot(num_records, latest_price, 'ro', label='Latest Price')
    plt.plot(num_records, predicted_price, 'go', label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    
    # Save the plot to a buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    
    return render_template('stock_predicted.html', symbol=symbol, predicted_price=predicted_price[0][0],
                           plot_data=plot_data)




# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/face_detection')
def facedetection():
    return render_template('face_detection.html')



@app.route('/detect', methods=['POST'])
def detect_faces():
    # Get the image file from the request
    image_file = request.files['image']
    
    # Save the image to a temporary directory
    image_path = 'static/temp_image.jpg'
    image_file.save(image_path)
    
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Save the image with the detected faces
    result_path = 'static/result_image.jpg'
    cv2.imwrite(result_path, image)

    # Remove the temporary image file
    os.remove(image_path)

    # Convert image to base64 string
    with open(result_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    return jsonify(encoded_string)


# OpenWeatherMap API key
API_KEY = 'b0e56b32407dc48b139b50aa72ac29bf'

def get_weather_data(city):
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric'
    response = requests.get(url)
    if response.status_code == 200:
        weather_data = response.json()
        return weather_data
    else:
        return None

@app.route('/weather_one')
def weather_one():
    return render_template('weather_one.html')




@app.route('/result', methods=['POST'])
def result():
    city = request.form['city']
    weather_data = get_weather_data(city)
    return render_template('weather_res.html', weather_data=weather_data)




@app.route('/imagetotext')
def imagetotext():
    return render_template('imagetotext.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return render_template('imagetotext.html', message='No image file selected')
    
    image = request.files['image']
    
    if image.filename == '':
        return render_template('imagetotext.html', message='No image file selected')
    
    if image and allowed_file(image.filename):
        text = extract_text(image)
        return render_template('imagetotext.html', message='Success', extracted_text=text)
    else:
        return render_template('imagetotext.html', message='Invalid file type')

def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def extract_text(image):
    img = Image.open(image)
    text = pytesseract.image_to_string(img)
    return text


@app.route('/spamclass')
def spamclass():
	return render_template('spamclass.html')

@app.route('/predict_spam',methods=['POST'])
def predict_spam():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('spamresult.html',prediction = my_prediction)



if __name__ == '__main__':
    app.run(debug=False,'0.0.0.0')
