# MLTrafficVision - Real-Time Traffic Analytics Platform

MLTrafficVision is a comprehensive real-time traffic detection and analytics system built with Django, OpenCV, and YOLO. The platform provides advanced visualization capabilities for traffic monitoring, vehicle detection, and traffic pattern analysis.

## 🚀 Features

- **Real-Time Traffic Detection**
  - Vehicle detection and classification
  - Speed estimation
  - Traffic density analysis
  - Congestion level monitoring

- **Interactive Analytics Dashboard**
  - Traffic volume over time
  - Vehicle type distribution
  - Average speed trends
  - Peak hours visualization
  - Real-time updates every 10 seconds

- **Advanced Visualization**
  - Interactive charts using Chart.js
  - Color-coded congestion levels
  - Responsive design with Tailwind CSS
  - Dynamic data updates

## 🛠️ Technology Stack

- **Backend**: Django, Python
- **Computer Vision**: OpenCV, YOLO (Ultralytics)
- **Frontend**: Chart.js, Tailwind CSS
- **Data Processing**: NumPy, Pandas
- **Time Handling**: Moment.js

## 📋 Prerequisites

- Python 3.8 or higher
- uv (Modern Python package installer)
- Virtual environment (recommended)
- Video input device or video files for testing

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MLTrafficVision.git
   cd MLTrafficVision
   ```

2. Install uv if you haven't already:
   ```bash
   pip install uv
   ```

3. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies using uv:
   ```bash
   uv pip install -r requirements.txt
   ```

5. Set up environment variables:
   Create a `.env` file in the root directory and add:
   ```
   DEBUG=True
   SECRET_KEY=your_secret_key
   ALLOWED_HOSTS=localhost,127.0.0.1
   ```

6. Run migrations:
   ```bash
   python manage.py migrate
   ```

7. Create a superuser (optional):
   ```bash
   python manage.py createsuperuser
   ```

## 🚀 Running the Application

1. Start the development server:
   ```bash
   python manage.py runserver
   ```

2. Access the application:
   - Main dashboard: http://localhost:8000/
   - Analytics dashboard: http://localhost:8000/analytics/
   - Video upload: http://localhost:8000/upload/

## 📊 Features in Detail

### Traffic Detection
- Real-time vehicle detection and tracking
- Multiple vehicle class recognition
- Speed estimation for each vehicle
- Traffic density calculation

### Analytics Dashboard
- Real-time traffic statistics
- Historical data visualization
- Interactive charts and graphs
- Customizable time ranges
- Export capabilities

### Data Management
- Automatic data aggregation
- Rolling 1-hour window for performance
- Efficient database queries
- Timezone support (Asia/Manila)

## 🔧 Configuration

Key settings can be modified in `traffic_vision/settings.py`:
- `TIME_ZONE`: Currently set to 'Asia/Manila'
- `DETECTION_INTERVAL`: Frequency of detection updates
- `DATA_RETENTION_PERIOD`: How long to keep historical data

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLO by Ultralytics
- OpenCV community
- Django framework
- Chart.js contributors

## 📫 Contact

For questions and support, please open an issue in the GitHub repository.

---

Made with ❤️ by [lanzy-lanzy]
