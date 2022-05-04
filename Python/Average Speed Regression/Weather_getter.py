import pandas as pd
from geographiclib.geodesic import Geodesic
import requests
import json

# dictionary for alternative place names
altNames = {}

def bearing(lat1, long1, lat2, long2):
    return Geodesic.WGS84.Inverse(lat1, long1, lat2, long2)['azi1']

def getWeather(data):
    # dictionary to hold weather data for each destination and date
    weatherData = {}
    weatherData['Date_Location'] = []

    # getting weather for every row in data
    for i in range(len(data)):
        # getting the destination and date from current line
        line = data.iloc[i]
        destination = line["Location"]
        date = line["Date"]

        print(destination, date)

        # calling getWeather function
        response = requests.get(
            "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/" + destination + "/" + date + "API KEY GOES HERE")
        print(response.status_code)

        # The API can't find some of the locations
        # So I will input them myself for the ones it can't find
        while response.status_code != 200:
            print("Error for location", destination)
            print("Searching for alternative name")

            # checking dictionary for alternative place name
            if destination in altNames:
                newDestination = altNames.get(destination)
            else:
                newDestination = input("Location wasn't found, try a different one:")

            # getting new response
            print("New Values:")
            print(newDestination, date)
            response = requests.get(
            "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/" + newDestination + "/" + date + "API KEY GOES HERE")

            # adding alternative name
            if response.status_code == 200:
                altNames[destination] = newDestination

        # Getting the response and appending weatherData dictionary
        jsonData = response.json()
        weatherData["Date_Location"].append({
            'destination': destination,
            'date': date,
            'data': jsonData,
        })

    #API call to get historical weather
    return weatherData

#getting data from csv
data = pd.read_csv('tdf_stages_regression.csv', encoding='latin1')


#                           #
#DOESNT NEED TO BE RUN AGAIN#
#                           #
# originData = data[['Origin', 'Date']]
# originData = originData.rename(columns={"Origin": "Location"})
#
# destinationData = data[['Destination', 'Date']]
# destinationData = destinationData.rename(columns={"Destination": "Location"})
#
# originWeatherData = getWeather(originData)
# destinationWeatherData = getWeather(destinationData)

# #outputting to JSON file
# with open('originWeatherData.json', 'w') as f:
#     json.dump(originWeatherData, f)
#
# #outputting to JSON file
# with open('destinationWeatherData.json', 'w') as f:
#     json.dump(destinationWeatherData, f)
#
# with open('altNames.json', 'w') as f:
#     json.dump(altNames, f)

#opening json files
originJson = open('originWeatherData.json')
originWeatherData = json.load(originJson)
originWeatherData = originWeatherData["Date_Location"]

destinationJson = open('destinationWeatherData.json')
destinationWeatherData = json.load(destinationJson)
destinationWeatherData = destinationWeatherData["Date_Location"]

#arrays to hold weather data
temperatures = []
precipitations = []
bearings = []
windspeeds = []
winddirections = []
humidities = []


#getting data from the json files in to the data dataframe
for i in range(len(data)):
    #traversing starting location json
    origin = originWeatherData[i]
    originData = origin["data"]
    lat1 = originData["latitude"]
    long1 = originData["longitude"]

    #traversing finishing location json
    dest = destinationWeatherData[i]
    destData = dest["data"]
    lat2 = destData["latitude"]
    long2 = destData["longitude"]

    #weather at finishing location
    destDataDays = destData['days'][0]
    temperatures.append(destDataDays['temp'])
    precipitations.append(destDataDays['precip'])
    windspeeds.append(destDataDays['windspeed'])
    winddirections.append(destDataDays['winddir'])
    humidities.append(destDataDays['humidity'])

    averageBearing = bearing(lat1, long1, lat2, long2)
    bearings.append(averageBearing)

data['Bearing'] = bearings
data['Temperature'] = temperatures
data['Precipitation'] = precipitations
data['Wind Speed'] = windspeeds
data['Wind Direction'] = winddirections
data['Humidity'] = humidities


#calculating wind angle from bearing and wind direction
relativeWind = []

Bearings = data['Bearing'].to_numpy()
WindDirections = data['Wind Direction'].to_numpy()
WindSpeeds = data['Wind Speed'].to_numpy()

for i in range(len(Bearings)):
    #start and finish location is the same
    if(Bearings[i] == 0):
        angle = 1
    else:
        angle = Bearings[i] - WindDirections[i]
        angle = abs((angle+180) % 360 - 180)

    relativeWindToAngle = (angle / 180) * WindSpeeds[i]
    relativeWind.append(relativeWindToAngle)

data["Relative Wind Speed"] = relativeWind

#removing NA rows
data = data.dropna()
#removing column which shouldn't be there, just an issue with read csv most likely
data = data.drop('Unnamed: 0', axis=1)

data = data[["Year", "Distance", "Type", "Stage" ,"Temperature", "Precipitation", "Humidity", "Wind Speed","Relative Wind Speed", "Average_speed"]]

data.to_csv('tdf_stages_weather.csv')

