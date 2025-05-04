import pandas as pd
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb
import folium
import joblib
from tqdm import tqdm
from folium.plugins import HeatMap
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

testing = False         # True, wenn nur ein kleiner Test-Datensatz geladen werden soll, False, wenn alle Daten geladen werden sollen
all_accidents = False   # True, wenn alle Unfälle geladen werden sollen // False, wenn nur die Unfälle mit Radfahrerbeteiligung geladen werden sollen

###################################################
## 0. Importe, Konfigurationen, Vorbereitungen  ##
###################################################

# === Daten laden ===
if not testing:
    # Verzeichnis, in dem die Dateien liegen
    directory = "../data/citibike/"

    # Liste aller CSV-Dateien im Verzeichnis
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

    dfs = []

    # Lade alle Dateien und speichere sie in einer Liste von DataFrames
    for file in tqdm(csv_files, desc="Processing Citibike files"):
        file_path = os.path.join(directory, file)

        # df = pd.read_csv(file_path, parse_dates=["started_at", "ended_at"])  # alles Laden, wenn man kann, Größenordnung paar GB

        # Mein MB Air M1 packt das aber nicht, daher nehm ich nur eine Teilmenge, i.e. jeden 10. entry
        chunks = pd.read_csv(file_path, parse_dates=["started_at", "ended_at"], chunksize=10000)
        sampled_chunks = []
        for chunk in chunks:
            sampled_chunk = chunk.iloc[::10]
            sampled_chunks.append(sampled_chunk)
        df = pd.concat(sampled_chunks, ignore_index=True)
        dfs.append(df)

    # Alle DataFrames in einem einzigen DataFrame zusammenführen
    citibike_df = pd.concat(dfs, ignore_index=True)

else:
    print("#### Testing mode: using a single file ####")
    citibike_df = pd.read_csv("../data/citibike/202301-citibike-tripdata_1.csv", parse_dates=["started_at", "ended_at"])


# Lade NYPD Unfälle und paar Transformationen
accidents_df = pd.read_csv("../data/nypd/Motor_Vehicle_Collisions_-_Crashes_20250502.csv")
accidents_df["CRASH DATE"] = pd.to_datetime(accidents_df["CRASH DATE"], format='%m/%d/%Y', errors='coerce')
accidents_df["CRASH TIME"] = pd.to_datetime(accidents_df["CRASH TIME"], format='%H:%M', errors='coerce')
accidents_df = accidents_df.rename(columns={"LATITUDE": "latitude", "LONGITUDE": "longitude"})

if not all_accidents:
    print("#### Filtering accidents for cyclists ####")
    # Spalten definieren aus denen wir die Fahrradrelevanz holen
    vehicle_cols = [f"VEHICLE TYPE CODE {i}" for i in range(1, 6)]
    factor_cols = [f"CONTRIBUTING FACTOR VEHICLE {i}" for i in range(1, 6)]

    # Fahrradbezug in VEHICLE TYPE oder CONTRIBUTING FACTOR
    bike_keywords = r"\bbike\b|\bbicycle\b|\bcitibike\b|\bcyclist\b|\bpedal\b|\bpedal bike\b|\bpedal bicycle\b"

    bicycle_in_vehicle = accidents_df[vehicle_cols].apply(
        lambda row: row.astype(str).str.lower().str.contains(bike_keywords).any(), axis=1
    )

    bicycle_in_factors = accidents_df[factor_cols].apply(
        lambda row: row.astype(str).str.lower().str.contains(bike_keywords).any(), axis=1
    )

    # Cyclist-Involvierung aus den numerischen Spalten
    cyclist_involved = (
        (accidents_df['NUMBER OF CYCLIST INJURED'] > 0) |
        (accidents_df['NUMBER OF CYCLIST KILLED'] > 0)
    )

    # Kombinierter Filter
    bike_related = cyclist_involved | bicycle_in_vehicle | bicycle_in_factors

    # Gefiltertes DataFrame
    relevant_accidents = accidents_df[bike_related].copy()

    print(f"Anzahl Fahrrad-relevanter Unfälle: {len(relevant_accidents)}")
    print(f"Anzahl aller Unfälle: {len(accidents_df)}")
    print(f"Anzahl relevanter Fahrradunfälle mit Verletzten oder Toten: {len(relevant_accidents[cyclist_involved])}")

    accidents_df = relevant_accidents


print(f"Loaded {len(citibike_df)} CitiBike trips and {len(accidents_df)} accidents.")
print(citibike_df.head())
print(accidents_df.head())


# === Daten vorbereiten ===

# Citibike: Fahrtdauer in Minuten berechnen
citibike_df["duration_min"] = (citibike_df["ended_at"] - citibike_df["started_at"]).dt.total_seconds() / 60
citibike_df["hour"] = citibike_df["started_at"].dt.hour
citibike_df["date"] = citibike_df["started_at"].dt.date

# NYPD Unfälle: Koordinaten bereinigen
accidents_df["latitude"] = pd.to_numeric(accidents_df["latitude"], errors="coerce")
accidents_df["longitude"] = pd.to_numeric(accidents_df["longitude"], errors="coerce")
accidents_df.dropna(subset=["latitude", "longitude"], inplace=True)

# -> Wir haben jetzt im 0. Schritt unsere zwei DataFrames, die wir brauchen: citibike_df und accidents_df, i.e. log-files zu den Fahrten und davon getrennt die Unfälle.


###################################################
##### 1. Exploriative Datenanalyse  #####
###################################################
## Hier schauen wir uns die Daten an, um ein Gefühl für die Verteilung der Daten zu bekommen und um zu sehen, ob es irgendwelche Auffälligkeiten gibt.
# === 1.1 Citibike Daten ===

# 1. Histogramm: Fahrdauer
plt.figure(figsize=(10, 5))
plt.hist(citibike_df[(citibike_df["duration_min"] > 0) & (citibike_df["duration_min"] < 100)]["duration_min"], bins=100)
mean_duration = citibike_df["duration_min"].median()
plt.axvline(mean_duration, color='red', linestyle='dashed', linewidth=1, label=f'Median: {mean_duration:.2f} min')
plt.legend()
plt.title("Verteilung der Fahrdauer")
plt.xlabel("Fahrdauer (Minuten)")
plt.ylabel("Anzahl Fahrten")
plt.tight_layout()
plt.savefig("../results/01_explorative_data_analysis/duration_histogram.png")

# 1.1 Durchschnittliche Fahrtzeit
print("\nAverage Fahrtzeit (min):", citibike_df["duration_min"].mean()) #Macht natürlich eher weniger Sinn, aber dennoch immer wieder interessant zu sehen!
print("Median Fahrtzeit (min):", citibike_df["duration_min"].median())
print("Modus Fahrtzeit (min):", citibike_df["duration_min"].mode()[0])

# 2. Verteilung über Tageszeit: Start
citibike_df["hour_start"] = citibike_df["started_at"].dt.hour
plt.figure(figsize=(10, 5))
plt.hist(citibike_df["hour"], bins=range(25), align='left', density=True)
plt.title("Anzahl der Ausleihen nach Tageszeit (Start)")
plt.xlabel("Stunde (0–23)")
plt.ylabel("Relative Anzahl Fahrten")
plt.tight_layout()
plt.savefig("../results/01_explorative_data_analysis/rides_per_hour_start.png")

# 2.1 Verteilung über Tageszeit: Ende
citibike_df["hour_end"] = citibike_df["ended_at"].dt.hour
plt.figure(figsize=(10, 5))
plt.hist(citibike_df["hour"], bins=range(25), align='left', density=True)
plt.title("Anzahl der Ausleihen nach Tageszeit (Ende)")
plt.xlabel("Stunde (0–23)")
plt.ylabel("Relative Anzahl Fahrten")
plt.tight_layout()
plt.savefig("../results/01_explorative_data_analysis/rides_per_hour_end.png")

# # 2.2 Verteilung über Tageszeit: Start und Ende
# plt.figure(figsize=(10, 5))
# plt.hist(citibike_df["hour_start"], bins=range(25), align='left', alpha=0.5, label='Start')
# plt.hist(citibike_df["hour_end"], bins=range(25), align='left', alpha=0.5, label='Ende')
# plt.title("Anzahl der Ausleihen nach Tageszeit (Start vs Ende)")
# plt.xlabel("Stunde (0–23)")
# plt.ylabel("Relative Anzahl Fahrten")
# plt.legend()
# plt.tight_layout()
# plt.savefig("../results/01_explorative_data_analysis/rides_per_hour_start_end.png")

# 3. Verteilung über Monate
citibike_df["month"] = citibike_df["started_at"].dt.month
plt.figure(figsize=(10, 5))
# sns.countplot(x="month", data=citibike_df)
plt.hist(citibike_df["month"], bins=12, align='left')
plt.xticks(range(1, 13), ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"])
plt.title("Anzahl der Ausleihen nach Monat")
plt.xlabel("Monat")
plt.ylabel("Anzahl Fahrten")
plt.tight_layout()
plt.savefig("../results/01_explorative_data_analysis/rides_per_month.png")

# 4. Verteilung nach Wochentag
citibike_df["weekday"] = citibike_df["started_at"].dt.day_name()
weekdays_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plt.figure(figsize=(10, 5))
plt.hist(citibike_df["weekday"], bins=7, align='left')
plt.title("Fahrten nach Wochentag")
plt.xlabel("Wochentag")
plt.ylabel("Anzahl Fahrten")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../results/01_explorative_data_analysis/rides_per_weekday.png")

# 5. Nutzergruppenanalyse: member vs. casual
plt.figure(figsize=(10, 5))
plt.hist(citibike_df["member_casual"], align='mid', bins=2)
plt.xticks([0.25, 0.75], ['Casual', 'Member'])
plt.title("Nutzergruppen (Member vs Casual)")
plt.xlabel("Typ")
plt.ylabel("Fahrten")
plt.tight_layout()
plt.savefig("../results/01_explorative_data_analysis/user_groups.png")

# 5.1 Nutzergruppenanalyse: electric_bike vs. classic_bike
plt.figure(figsize=(10, 5))
plt.hist(citibike_df["rideable_type"], align='mid', bins=2)
plt.xticks([0.25, 0.75], ["Classic Bike", "E-Bike"])
plt.title("Fahrten nach Fahrradtyp (E-Bike vs Classic Bike)")
plt.xlabel("Fahrradtyp")
plt.ylabel("Fahrten")
plt.tight_layout()
plt.savefig("../results/01_explorative_data_analysis/bike_types.png")

# 5.2 Nutzergruppenanalyse: nach Wochentag
plt.figure(figsize=(10, 5))
sns.countplot(x="weekday", hue="member_casual", data=citibike_df, order=weekdays_order)
plt.title("Fahrten nach Wochentag (Member vs Casual)")
plt.xlabel("Wochentag")
plt.ylabel("Anzahl Fahrten")
plt.xticks(rotation=45)
plt.legend(title="Nutzergruppe")
plt.tight_layout()
plt.savefig("../results/01_explorative_data_analysis/rides_per_weekday_user_groups.png")

# 6. Analyse der meistgenutzten Startstationen
plt.figure(figsize=(10, 5))
top_start_stations = citibike_df["start_station_name"].value_counts().head(15)
sns.barplot(x=top_start_stations.values, y=top_start_stations.index)
plt.title("Top 15 meistgenutzte Startstationen")
plt.xlabel("Anzahl Fahrten")
plt.ylabel("Startstation")
plt.tight_layout()
plt.savefig("../results/01_explorative_data_analysis/top_start_stations.png")

# 7. Analyse der meistgenutzten Endstationen
plt.figure(figsize=(10, 5))
top_end_stations = citibike_df["end_station_name"].value_counts().head(15)
sns.barplot(x=top_end_stations.values, y=top_end_stations.index)
plt.title("Top 15 meistgenutzte Endstationen")
plt.xlabel("Anzahl Fahrten")
plt.ylabel("Endstation")
plt.tight_layout()
plt.savefig("../results/01_explorative_data_analysis/top_end_stations.png")

# 8. Analyse der wenigsten genutzten Startstationen
plt.figure(figsize=(10, 5))
bottom_start_stations = citibike_df["start_station_name"].value_counts().tail(15)
sns.barplot(x=bottom_start_stations.values, y=bottom_start_stations.index)
plt.title("15 wenigsten genutzte Startstationen")
plt.xlabel("Anzahl Fahrten")
plt.ylabel("Startstation")
plt.tight_layout()
plt.savefig("../results/01_explorative_data_analysis/bottom_start_stations.png")

# 9. Analyse der wenigsten genutzten Endstationen
plt.figure(figsize=(10, 5))
bottom_end_stations = citibike_df["end_station_name"].value_counts().tail(15)
sns.barplot(x=bottom_end_stations.values, y=bottom_end_stations.index)
plt.title("15 wenigsten genutzte Endstationen")
plt.xlabel("Anzahl Fahrten")
plt.ylabel("Endstation")
plt.tight_layout()
plt.savefig("../results/01_explorative_data_analysis/bottom_end_stations.png")

# 10. Analyse der Fahrtstrecken (Luftlinie), auch wenn das vermutlich nicht viel anders aussieht als die Fahrtdauer durch eine im Schnitt konstante Durchschnittsgeschwindigkeit, schauen wir da mal rein.
# cf.: https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Erdradius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

# Strecke berechnen in km
citibike_df["distance_km"] = haversine(
    citibike_df["start_lat"],
    citibike_df["start_lng"],
    citibike_df["end_lat"],
    citibike_df["end_lng"]
)

plt.figure(figsize=(10, 5))
plt.hist(citibike_df[(citibike_df["distance_km"] > 0) & (citibike_df["distance_km"] < 10)]["distance_km"], bins=100)
mean_duration = citibike_df["distance_km"].median()
plt.axvline(mean_duration, color='red', linestyle='dashed', linewidth=1, label=f'Median: {mean_duration:.2f} km')
plt.legend()
plt.title("Verteilung der Fahrtstrecken (Luftlinie)")
plt.xlabel("Fahrtstrecke (km)")
plt.ylabel("Anzahl Fahrten")
plt.tight_layout()
plt.savefig("../results/01_explorative_data_analysis/distance_histogram.png")

# === 1.2 NYPD Daten ===

## Heatmap der Unfälle

# Karte zentrieren (Manhattan ungefähr)
m = folium.Map(location=[40.75, -73.98], zoom_start=12)

# Heatmap Punkte hinzufügen
heat_data = list(zip(accidents_df["latitude"], accidents_df["longitude"]))
HeatMap(heat_data[:1000]).add_to(m)  # Limitiert auf 1000 Punkte zur Performance

# Karte speichern
m.save("../results/01_explorative_data_analysis/accidents_heatmap.html")

print("\nHeatmap gespeichert als accidents_heatmap.html")

# === paar interessante Statistiken über Unfälle ===

### Plot der Unfälle über Wochentage
plt.figure(figsize=(10, 5))
plt.hist(accidents_df["CRASH DATE"].dt.day, bins=7, align='left')
plt.title("Unfälle nach Wochentag")
plt.xlabel("Wochentag")
plt.ylabel("Anzahl Unfälle")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../results/01_explorative_data_analysis/accidents_per_weekday.png")

# # Plot der Unfälle über Monate
plt.figure(figsize=(10, 5))
plt.hist(accidents_df["CRASH DATE"].dt.month, bins=range(1,13), align='left')
plt.title("Unfälle nach Monat")
plt.xlabel("Monat")
plt.ylabel("Anzahl Unfälle")
plt.xticks(range(1, 13), ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"])
plt.tight_layout()
plt.savefig("../results/01_explorative_data_analysis/accidents_per_month.png")

# # Plot der Unfälle über Tageszeit
plt.figure(figsize=(10, 5))
plt.hist(accidents_df["CRASH TIME"].dt.hour, bins=range(25), align='left')
plt.title("Unfälle nach Tageszeit")
plt.xlabel("Stunde (0–23)")
plt.ylabel("Anzahl Unfälle")
plt.tight_layout()
plt.savefig("../results/01_explorative_data_analysis/accidents_per_hour.png")



###########################################################################
##### 2. Was können wir aus der Verknüpfung der Daten neues lernen? #####
###########################################################################

# === 2.1 Unfälle und Citibike-Fahrten zusammenführen ===
# Koordinaten der Citibike-Stationen
# Alle eindeutigen Stationen sammeln
stations = pd.concat([
    citibike_df[["start_station_name", "start_lat", "start_lng"]].rename(
        columns={"start_station_name": "station", "start_lat": "lat", "start_lng": "lng"}
    ),
    citibike_df[["end_station_name", "end_lat", "end_lng"]].rename(
        columns={"end_station_name": "station", "end_lat": "lat", "end_lng": "lng"}
    )
]).drop_duplicates("station").dropna()

# Stationen & Unfälle als Arrays (in RADIANS für BallTree)
stations_coords_rad = np.radians(stations[["lat", "lng"]].to_numpy())
crashes_coords_rad = np.radians(accidents_df[["latitude", "longitude"]].to_numpy())

# BallTree auf Unfalldaten
tree = BallTree(crashes_coords_rad, metric='haversine')

# Radius in Kilometer, umgewandelt in Winkelmaß
radius_km = 0.25 # Design choice: ich interessiere mich für alle Unfälle in einem 250m Umkreis
radius_rad = radius_km / 6371.0  # Erdradius in km

# Für jede Station: wie viele Unfälle im Radius
counts = tree.query_radius(stations_coords_rad, r=radius_rad, count_only=True)

# Ergebnis als neue Spalte speichern
stations["crashes_nearby_250m"] = counts

# Welches sind denn die gefährlichsten Stationen?
top_risk_stations = stations.sort_values("crashes_nearby_250m", ascending=False).head(int(len(stations) * 0.02)) # get top 2 % of stations

plt.figure(figsize=(10, 6))
sns.barplot(x="crashes_nearby_250m", y="station", data=top_risk_stations)
plt.title("Top 2% Verleihstationen mit den meisten Unfällen im Umkreis (250 m)")
plt.xlabel("Anzahl Unfälle")
plt.ylabel("Station")
plt.tight_layout()
plt.savefig("../results/02_insights/top_risk_stations.png")

# Risiko exposure, also Risiko/Verleih-Verhältnis
ride_counts = citibike_df["start_station_name"].value_counts()
stations["rides_from_station"] = stations["station"].map(ride_counts)

# Unfälle pro Fahrten in arbitrary units: wir haben keine informationen darüber, wie sich die Unfälle aller Fahrradfahrer auf unsere Anzahl Fahrten verteilen, daher ist dies hier eine _relative_ Information, um z.B. Stationen untereinander abzuwägen. Also nicht Unfälle pro 1000 Fahrten, aber wir wissen Station A ist gefährlicher als B. 
stations["crashes_per_rides"] = (stations["crashes_nearby_250m"] / stations["rides_from_station"])
stations = stations.dropna(subset=["crashes_per_rides"])

# Top 10 riskanteste Stationen
top_stations = stations.nlargest(10, "crashes_per_rides")
# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x="crashes_per_rides", y="station", data=top_stations)
plt.title("Top 10 Verleihstationen mit den meisten Unfällen pro Ausleihen")
plt.xlabel("Unfälle pro Fahrten (arbitrary units)")
plt.ylabel("Station")
plt.tight_layout()
plt.savefig("../results/02_insights/top_risk_stations_per_rides.png")

### -> Zwischenergebnis: Für die Kooperation bedeutet das: 
# Citibike weiß jetzt, wo die Gefahren sind: Es kann Sicherheitswarnungen für Nutzer beim Ausleihen geben und gegenüber der Versicherung zeigen, dass man die Nutzer sensibilisiert. Hypothese: Die Fahrradfahrer passen auf und die Unfälle sinken.
# Risikozuschläge/höhere Prämie, die die Versicherung noch zu bestimmen hat.
# Standortanalysen für CitiBike, zusammen mit den beliebtesten Stationen könnte man schauen, ob man die überhaupt braucht oder lieber drauf verzichtet.
############################################################


## Analyse der Unfallhäufigkeit nach Tageszeit (nur weil nachmittags die meisten Unfälle sind, heisst das ja offensichtlich nicht, dass die Unfallgefahr am höchsten ist)
accidents_df["crash_hour"] = accidents_df["CRASH TIME"].dt.hour # Extrahiere die Stunde (0 bis 23)

# Rel. Anzahl Unfälle pro Stunde
accidents_per_hour = accidents_df["crash_hour"].value_counts(normalize=True).sort_index()

citibike_df["start_hour"] = pd.to_datetime(citibike_df["started_at"]).dt.hour
rides_per_hour = citibike_df["start_hour"].value_counts(normalize=True).sort_index()

# Relatives Risiko
risk_per_hour = (accidents_per_hour / rides_per_hour)

plt.figure(figsize=(10, 5))
plt.plot(risk_per_hour.index, risk_per_hour.values, marker='o')
plt.title("Relatives Risiko pro Stunde")
plt.xlabel("Stunde (0–23)")
plt.ylabel("Relatives Risiko (arbitrary units)")
plt.xticks(range(0, 24))
plt.tight_layout()
plt.savefig("../results/02_insights/relative_risk_per_hour.png")

## -> wir haben jetzt hier zwei Merkmale herausgearbeitet: Die gefährlichsten Stationen und die gefährlichsten Uhrzeiten.

## Schauen wir mal, ob es unterschiede zwischen den Nutzergruppen gibt. Das konnten wir auch schon ohne die NYPD Daten machen, dafür helfen sie uns ja nicht, aber ich möchte sehen, wer z.B. tendenziell länger fährt.
colors = {'member': 'blue', 'casual': 'orange'}
medians = {}

plt.figure(figsize=(10, 5))
for user_type in ['member', 'casual']:
    data = citibike_df[citibike_df["member_casual"] == user_type]
    
    # Normalisiere das Histogramm auf die Fläche (Integral = 1), sonst wirds schwer die Verteilungen zu vergleichen
    plt.hist(data[(data["duration_min"] > 0) & (data["duration_min"] < 100)]["duration_min"], 
             bins=100, alpha=0.5, label=f'{user_type.capitalize()}', 
             color=colors[user_type], density=True)

    # Berechne und plotte die Median für jede Gruppe
    median_duration = data["duration_min"].median()
    medians[user_type] = median_duration
    plt.axvline(median_duration, color=colors[user_type], linestyle='dashed', linewidth=1, 
                label=f'Median {user_type.capitalize()}: {median_duration:.2f} min')

# Berechne den prozentualen Unterschied der Mediane
median_diff_percent_duration = ((medians['casual'] - medians['member']) / medians['member']) * 100
print(f"Die median duration ist {median_diff_percent_duration:.2f}% höher für casual Nutzer als für member Nutzer.")
# -> Der Unterschied ist schon deutlich, möglicherweise ein Indiz für höhere Unfallgefahr, wenn sie länger fahren. Für mich an diesem Punkt jetzt eine Annahme/Hypothese.
plt.legend()
plt.title("Verteilung der Fahrdauer nach Nutzergruppen")
plt.xlabel("Fahrdauer (Minuten)")
plt.ylabel("Relative Anzahl Fahrten")
plt.tight_layout()
plt.savefig("../results/02_insights/duration_histogram_by_user_type.png")

# Das gleiche, aber für die Fahrtstrecke
plt.figure(figsize=(10, 5))
for user_type in ['member', 'casual']:
    data = citibike_df[citibike_df["member_casual"] == user_type]
    
    # Normalisiere das Histogramm auf die Fläche (Integral = 1)
    plt.hist(data[(data["distance_km"] > 0) & (data["distance_km"] < 10)]["distance_km"], 
             bins=100, alpha=0.5, label=f'{user_type.capitalize()}', 
             color=colors[user_type], density=True)

    # Berechne und plotte die Median für jede Gruppe
    median_distance = data["distance_km"].median()
    medians[user_type] = median_distance
    plt.axvline(median_distance, color=colors[user_type], linestyle='dashed', linewidth=1, 
                label=f'Median {user_type.capitalize()}: {median_distance:.2f} km')
# Berechne den prozentualen Unterschied der Mediane
median_diff_percent_distance = ((medians['casual'] - medians['member']) / medians['member']) * 100
print(f"Die median distance ist {median_diff_percent_distance:.2f}% höher für casual Nutzer als für member Nutzer.")

plt.legend()
plt.title("Verteilung der Fahrtstrecke nach Nutzergruppen")
plt.xlabel("Fahrtstrecke (km)")
plt.ylabel("Relative Anzahl Fahrten")
plt.tight_layout()
plt.savefig("../results/02_insights/distance_histogram_by_user_type.png")

if median_diff_percent_duration/median_diff_percent_distance  > 1.2:
    print("Die casual Nutzer fahren entweder im Durchschnitt einfach langsamer oder stehen rum. Oder aber sie fahren im Kreis, z.b. Sightseeing, das sehen wir in Daten natürlich nicht.")


## -> Insgesamt haben wir jetzt 3 Dinge gelernt: Welche Uhrzeiten und Stationen gefährlich sind, und dass casual Nutzer im Schnitt länger fahren als Member. Man kann immer mehr machen um zu verfeinern, aber das reicht für den ersten Schritt.


############################################################
##### 3. Bau eines Modells, um die Unfälle vorherzusagen. #####
############################################################

# Beispiel für einen Risikofaktor, den wir aus unseren drei Erkenntnissen konstruieren.
# Ich verwende hier explizit nur die Informationen, die wir *vor* einem Fahrtantritt haben, also nicht die Dauer oder die Zielstation, cf. README.md
citibike_df['risk_hour'] = citibike_df['hour'].map(risk_per_hour)  # Risiko basierend auf der Stunde
citibike_df['risk_station'] = citibike_df['start_station_name'].map(stations.set_index('station')['crashes_per_rides'])  # Risiko basierend auf der Station
citibike_df['risk_user_type'] = citibike_df['member_casual'].map({'member': 1, 'casual': 1.2})  # Casual-Nutzer haben einen höheren Risikofaktor, weil sie länger fahren als member

# Normalisierung der Einzelrisikofaktoren, die einzelnen Risiken sind auf arbiträren Skalen, sonst macht die weighted Sum ja kein Sinn. Normalisierung auf [0, 1], einfach (wert-min)/(max-min)
# Separater Scaler für Features
scaler_features = MinMaxScaler()
citibike_df[['risk_hour', 'risk_station', 'risk_user_type']] = scaler_features.fit_transform(
    citibike_df[['risk_hour', 'risk_station', 'risk_user_type']]
)

# Risiko-Berechnung basierend auf normalisierten Werten. Die relative Gewichtung ist hier ein educated guess, z.B. dass der user type nicht ganz so wichtig ist. Selbstverständlich ist das zu tunen.
citibike_df['risk_factor'] = (
    citibike_df['risk_hour'] * 0.4 +
    citibike_df['risk_station'] * 0.4 +
    citibike_df['risk_user_type'] * 0.2
)

# Nan raus
citibike_df = citibike_df.dropna(subset=['risk_hour', 'risk_station', 'risk_user_type', 'risk_factor'])

scaler_target = MinMaxScaler()
citibike_df['normalized_risk_factor'] = scaler_target.fit_transform(citibike_df[['risk_factor']])

## Control plot, Verteilung der Risikofaktoren, nur aus Interesse/double-check Gründen, dass die nicht Auffälligkeiten zeigen
plt.figure(figsize=(10, 5))
plt.hist(citibike_df["risk_factor"], bins=100)
plt.title("Verteilung der Risikofaktoren")
plt.xlabel("Risikofaktor")
plt.ylabel("Anzahl")
plt.tight_layout()
plt.savefig("../results/03_model/risk_factor_distribution.png")

# Features und Ziel definieren
features = ['risk_hour', 'risk_station', 'risk_user_type']
X = citibike_df[features]
y = citibike_df['normalized_risk_factor']

# Daten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)

# Modell initialisieren und trainieren
print("Trainiere Decision Tree Regressor...")
model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=100, random_state=1337)
model.fit(X_train, y_train)

# Vorhersage
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R^2: {r2:.4f}")


plt.figure(figsize=(12, 6))
plot_tree(model, feature_names=features, filled=True, rounded=True)
plt.savefig("../results/03_model/decision_tree_model.png")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Ideale Linie
plt.xlabel("Wahrer Risiko-Faktor")
plt.ylabel("Vorhergesagter Risiko-Faktor")
plt.title("Vorhersage vs. Wahrheit")
plt.grid(True)
plt.tight_layout()
plt.savefig("../results/03_model/dtr_risk_prediction_scatterplot.png")

residuals = y_test - y_pred
plt.figure(figsize=(6, 4))
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Vorhergesagter Risiko-Faktor")
plt.ylabel("Residuen")
plt.title("Residuenplot")
plt.grid(True)
plt.tight_layout()
plt.savefig("../results/03_model/dtr_residuals_plot.png")


scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print("Einzelergebnisse:", scores)
print(f"Durchschnittlicher R^2: {scores.mean():.4f}")
print(f"Standardabweichung: {scores.std():.4f}")


# check if the model is overfitting, wenn train r^2 viel höher ist als test r^2, dann overfitting
y_train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_pred)
print(f"Train R^2: {train_r2:.4f}")
print(f"Test  R^2: {test_r2:.4f}")


## Probieren wir noch ein anderes Modell, Gradient Boosting
# Modell initialisieren und trainieren
print("Trainiere Gradient Boosting Regressor...")
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=1337
)
model.fit(X_train, y_train)

# Vorhersage
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R^2: {r2:.4f}")

# Cross-Validation (optional)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"CV R^2 scores: {cv_scores}")
print(f"CV R^2 mean: {cv_scores.mean():.4f}")

# Scatterplot Vorhersage vs. Wahrheit
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel("Wahrer Risiko-Faktor")
plt.ylabel("Vorhergesagter Risiko-Faktor")
plt.title("Vorhersage vs. Wahrheit")
plt.grid(True)
plt.tight_layout()
plt.savefig("../results/03_model/gbr_risk_prediction_scatterplot.png")

# Residuenplot
residuals = y_test - y_pred
plt.figure(figsize=(6, 4))
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Vorhergesagter Risiko-Faktor")
plt.ylabel("Residuen")
plt.title("Residuenplot")
plt.grid(True)
plt.tight_layout()
plt.savefig("../results/03_model/gbr_residuals_plot.png")


# Speichern des Modells, Scalers und der Risikomappings
joblib.dump(model, "../models/risk_model_gbr.pkl")
joblib.dump(scaler_features, "../models/risk_scaler_features.pkl")
joblib.dump(scaler_target, "../models/risk_scaler_target.pkl")  #normalized output
joblib.dump(risk_per_hour, "../models/risk_per_hour.pkl")
joblib.dump(stations.set_index('station')['crashes_per_rides'].to_dict(), "../models/risk_per_station.pkl")
