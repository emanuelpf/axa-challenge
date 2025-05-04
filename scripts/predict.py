import joblib
import pandas as pd
import numpy as np

# Modelle & Mapping laden
model = joblib.load("../models/risk_model_gbr.pkl")
scaler_features = joblib.load("../models/risk_scaler_features.pkl")
risk_per_hour = joblib.load("../models/risk_per_hour.pkl")
risk_per_station = joblib.load("../models/risk_per_station.pkl")

def predict_risk(hour, station_name, user_type):
    # 1. Feature-Mapping
    risk_hour = risk_per_hour.get(hour, np.nan)
    risk_station = risk_per_station.get(station_name, np.nan)
    risk_user_type = {'member': 1.0, 'casual': 1.2}.get(user_type, np.nan)

    # Fehlerbehandlung bei ungültigen Eingaben
    if np.isnan(risk_hour) or np.isnan(risk_station) or np.isnan(risk_user_type):
        raise ValueError(f"Ungültiger Eingabewert. Gegeben: hour={hour}, station='{station_name}', user_type='{user_type}'")

    # 2. In DataFrame packen
    input_df = pd.DataFrame([{
        'risk_hour': risk_hour,
        'risk_station': risk_station,
        'risk_user_type': risk_user_type
    }])

    # 3. Skalieren
    input_scaled = scaler_features.transform(input_df)

    # 4. Vorhersage
    predicted_risk = model.predict(input_scaled)[0]

    return predicted_risk


# Beispielaufruf
if __name__ == "__main__":
    try:
        pred = predict_risk(14, "Vesey St & Church St", "member")
        print(f"Vorhergesagter Risiko-Faktor: {pred:.4f}")
        pred = predict_risk(4, "Vesey St & Church St", "casual")
        print(f"Vorhergesagter Risiko-Faktor: {pred:.4f}")
    except ValueError as e:
        print(f"Fehler: {e}")
