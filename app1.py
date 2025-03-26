import pickle
from sklearn.preprocessing import LabelEncoder

# Example locations
locations = ["USA", "Canada", "UK", "Germany", "Australia"]

# Create and fit LabelEncoder
label_enc = LabelEncoder()
label_enc.fit(locations)

# Save the encoder
with open("label_encoders.pkl", "wb") as file:
    pickle.dump({"Location": label_enc}, file)