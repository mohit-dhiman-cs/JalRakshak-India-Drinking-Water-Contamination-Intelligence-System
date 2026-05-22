import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os
from dotenv import load_dotenv

load_dotenv()

class CPCBDataLoader:
    """Handles data ingestion from CPCB National Water Monitoring Program."""
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("CPCB_API_KEY")
        self.base_url = "https://api.cpcb.gov.in/v1/water-quality" # Placeholder URL

    def fetch_live_data(self, station_id):
        """Placeholder for real CPCB API call."""
        if not self.api_key:
            return self.get_synthetic_data(station_id)
        
        # Real implementation would go here:
        # response = requests.get(f"{self.base_url}/{station_id}", headers={"X-API-KEY": self.api_key})
        # return pd.DataFrame(response.json())
        return self.get_synthetic_data(station_id)

    def get_synthetic_data(self, station_id="Sample-Station"):
        """Generates synthetic CPCB data for development."""
        dates = [datetime(2025, 1, 1) + timedelta(days=30*i) for i in range(12)]
        data = []
        # Start at a healthy WQI (~75) and degrade down to (~31) over 12 months.
        # This guarantees both classes (>=50 and <50) are present in the target variable,
        # preventing classification training errors, while retaining a strong downward trend.
        for i, date in enumerate(dates):
            wqi = 75.0 - (i * 4.0) + np.random.normal(0, 2.0)
            data.append({
                'Date': date,
                'Station': station_id,
                'WQI': max(0.0, min(100.0, wqi)),
                'pH': np.random.uniform(6.5, 8.5),
                'TDS': np.random.uniform(100, 1000),
                'Turbidity': np.random.uniform(0.1, 10),
                'DO': np.random.uniform(4, 10),
                'BOD': np.random.uniform(1, 20),
                'Nitrates': np.random.uniform(0, 50)
            })
        return pd.DataFrame(data)

class SocialMediaLoader:
    """Handles fetching citizen complaints from Twitter/X and Google Reviews."""
    def __init__(self, bearer_token=None):
        self.bearer_token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN")

    def fetch_tweets(self, query="water contamination India"):
        """Placeholder for Tweepy/Twitter API call."""
        if not self.bearer_token:
            return self.get_synthetic_complaints()
        
        # Real implementation using tweepy:
        # client = tweepy.Client(bearer_token=self.bearer_token)
        # tweets = client.search_recent_tweets(query=query)
        return self.get_synthetic_complaints()

    def get_synthetic_complaints(self):
        complaints = [
            "Water in our area smells like sewage since yesterday. #Varanasi #WaterCrisis",
            "Many kids in the neighborhood are falling sick after drinking tap water.",
            "The water looks brownish today. Is it even treated?",
            "Thank you municipal corporation for fixing the pipeline.",
            "Frequent water cuts are making life difficult in Patna.",
            "Yellowish water supply in Sector 4. Please look into it @MunCorp",
            "Safe drinking water is a basic right. Why is the WQI so low here?",
            "Diarrhea outbreak reported in several houses in our colony.",
            "Water pressure is low but quality seems okay for now.",
            "Avoid drinking tap water directly, it's highly contaminated."
        ]
        labels = [1, 1, 1, 0, 0, 1, 1, 1, 0, 1]
        return pd.DataFrame({'Text': complaints, 'Label': labels})

class InfrastructureDataLoader:
    """Fetches district-level infrastructure and census data from data.gov.in."""
    def fetch_district_data(self):
        # Expanded list of major Indian cities/districts for a more 'national' feel
        india_cities = [
            {'District': 'Mumbai', 'lat': 19.0760, 'lon': 72.8777},
            {'District': 'Delhi', 'lat': 28.6139, 'lon': 77.2090},
            {'District': 'Bangalore', 'lat': 12.9716, 'lon': 77.5946},
            {'District': 'Hyderabad', 'lat': 17.3850, 'lon': 78.4867},
            {'District': 'Ahmedabad', 'lat': 23.0225, 'lon': 72.5714},
            {'District': 'Chennai', 'lat': 13.0827, 'lon': 80.2707},
            {'District': 'Kolkata', 'lat': 22.5726, 'lon': 88.3639},
            {'District': 'Surat', 'lat': 21.1702, 'lon': 72.8311},
            {'District': 'Pune', 'lat': 18.5204, 'lon': 73.8567},
            {'District': 'Jaipur', 'lat': 26.9124, 'lon': 75.7873},
            {'District': 'Lucknow', 'lat': 26.8467, 'lon': 80.9462},
            {'District': 'Kanpur', 'lat': 26.4499, 'lon': 80.3319},
            {'District': 'Nagpur', 'lat': 21.1458, 'lon': 79.0882},
            {'District': 'Indore', 'lat': 22.7196, 'lon': 75.8577},
            {'District': 'Thane', 'lat': 19.2183, 'lon': 72.9781},
            {'District': 'Bhopal', 'lat': 23.2599, 'lon': 77.4126},
            {'District': 'Visakhapatnam', 'lat': 17.6868, 'lon': 83.2185},
            {'District': 'Pimpri-Chinchwad', 'lat': 18.6298, 'lon': 73.7997},
            {'District': 'Patna', 'lat': 25.5941, 'lon': 85.1376},
            {'District': 'Vadodara', 'lat': 22.3072, 'lon': 73.1812},
            {'District': 'Ghaziabad', 'lat': 28.6692, 'lon': 77.4538},
            {'District': 'Ludhiana', 'lat': 30.9010, 'lon': 75.8573},
            {'District': 'Agra', 'lat': 27.1767, 'lon': 78.0081},
            {'District': 'Nashik', 'lat': 19.9975, 'lon': 73.7898},
            {'District': 'Faridabad', 'lat': 28.4089, 'lon': 77.3178},
            {'District': 'Meerut', 'lat': 28.9845, 'lon': 77.7064},
            {'District': 'Rajkot', 'lat': 22.3039, 'lon': 70.8022},
            {'District': 'Kalyan-Dombivli', 'lat': 19.2403, 'lon': 73.1305},
            {'District': 'Vasai-Virar', 'lat': 19.3919, 'lon': 72.8397},
            {'District': 'Varanasi', 'lat': 25.3176, 'lon': 82.9739},
            {'District': 'Srinagar', 'lat': 34.0837, 'lon': 74.7973},
            {'District': 'Aurangabad', 'lat': 19.8762, 'lon': 75.3433},
            {'District': 'Dhanbad', 'lat': 23.7957, 'lon': 86.4304},
            {'District': 'Amritsar', 'lat': 31.6340, 'lon': 74.8723},
            {'District': 'Navi Mumbai', 'lat': 19.0330, 'lon': 73.0297},
            {'District': 'Allahabad', 'lat': 25.4358, 'lon': 81.8463},
            {'District': 'Ranchi', 'lat': 23.3441, 'lon': 85.3096},
            {'District': 'Howrah', 'lat': 22.5726, 'lon': 88.3639},
            {'District': 'Coimbatore', 'lat': 11.0168, 'lon': 76.9558},
            {'District': 'Jabalpur', 'lat': 23.1815, 'lon': 79.9864},
            {'District': 'Gwalior', 'lat': 26.2124, 'lon': 78.1772},
            {'District': 'Vijayawada', 'lat': 16.5062, 'lon': 80.6480},
            {'District': 'Jodhpur', 'lat': 26.2389, 'lon': 73.0243},
            {'District': 'Madurai', 'lat': 9.9252, 'lon': 78.1198},
            {'District': 'Raipur', 'lat': 21.2514, 'lon': 81.6296},
            {'District': 'Chandigarh', 'lat': 30.7333, 'lon': 76.7794},
            {'District': 'Guwahati', 'lat': 26.1445, 'lon': 91.7362}
        ]
        
        data = []
        for city in india_cities:
            data.append({
                'District': city['District'],
                'lat': city['lat'],
                'lon': city['lon'],
                'Avg_WQI': np.random.uniform(30, 90),
                'Outbreak_Freq': np.random.randint(0, 10),
                'Pipe_Age_Index': np.random.uniform(1, 50),
                'Pop_Density': np.random.randint(500, 10000),
                'Sewage_Coverage': np.random.uniform(10, 95),
                'Monsoon_Rainfall': np.random.uniform(500, 3000)
            })
        return pd.DataFrame(data)

# Helper functions for backward compatibility with app.py
def get_cpcb_sample_data():
    return CPCBDataLoader().get_synthetic_data()

def get_tweet_complaints():
    return SocialMediaLoader().get_synthetic_complaints()

def get_district_risk_data():
    return InfrastructureDataLoader().fetch_district_data()
