import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

class EnhancedJokeRecommender:
    def __init__(self):
        self.dense_joke_ids = [5, 7, 8, 13, 15, 16, 17, 18, 19, 20]
        
    def prepare_data(self, ratings_df, jokes_df):
        """Prépare les données avec une meilleure normalisation."""
        self.unique_jokes = jokes_df.drop_duplicates(subset=['joke_id']).set_index('joke_id')
        
        # Sélectionner et normaliser les évaluations
        ratings_matrix = ratings_df.iloc[:, 1:101]
        self.dense_ratings = ratings_matrix.iloc[:, [id-1 for id in self.dense_joke_ids]]
        
        # Normalisation par utilisateur pour mieux capturer les préférences individuelles
        self.dense_ratings.replace(99, np.nan, inplace=True)
        
        # Normalisation centrée réduite par utilisateur
        user_means = self.dense_ratings.mean(axis=1)
        user_stds = self.dense_ratings.std(axis=1)
        normalized_ratings = self.dense_ratings.sub(user_means, axis=0).div(user_stds, axis=0)
        
        # Remplacer les NaN par 0 après la normalisation
        return normalized_ratings.fillna(0)
    
    def build_model(self, input_dim):
        """Construit un modèle plus sophistiqué avec des couches supplémentaires."""
        input_layer = tf.keras.layers.Input(shape=(input_dim,))
        
        # Encoder
        x = tf.keras.layers.Dense(128, activation='relu')(input_layer)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        encoded = tf.keras.layers.Dense(32, activation='relu')(x)
        
        # Decoder
        x = tf.keras.layers.Dense(64, activation='relu')(encoded)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        decoded = tf.keras.layers.Dense(input_dim, activation='tanh')(x)
        
        return tf.keras.Model(inputs=input_layer, outputs=decoded)
    
    def train_model(self, train_data, test_data, epochs=20):
        """Entraîne le modèle avec une configuration optimisée."""
        self.model = self.build_model(train_data.shape[1])
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5
        )
        
        history = self.model.fit(
            train_data, 
            train_data,
            epochs=epochs,
            batch_size=128,
            validation_data=(test_data, test_data),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def get_recommendations(self, user_index, predicted_ratings, original_ratings, num_recommendations=3):
        """Génère des recommandations personnalisées."""
        user_ratings = predicted_ratings[user_index]
        original_user_ratings = original_ratings[user_index]
        
        # Créer un masque pour les blagues non notées
        unrated_mask = original_user_ratings == 0
        
        if not any(unrated_mask):
            return []
        
        # Calculer un score personnalisé
        user_preference_score = user_ratings * (1 + np.abs(user_ratings - np.mean(user_ratings)))
        recommendation_scores = user_preference_score * unrated_mask
        
        best_indices = np.argsort(recommendation_scores)[::-1][:num_recommendations]
        return [self.dense_joke_ids[idx] for idx in best_indices]
    
    def get_joke_text(self, joke_ids):
        """Récupère les textes des blagues."""
        return self.unique_jokes.loc[joke_ids][['text']]

def main():
    # Chargement des données
    ratings_df = pd.read_csv("all_ratings.csv")
    jokes_df = pd.read_csv("all_jokes.csv")
    
    # Initialisation et préparation
    recommender = EnhancedJokeRecommender()
    processed_data = recommender.prepare_data(ratings_df, jokes_df)
    
    # Division des données
    train_data, test_data = train_test_split(processed_data, test_size=0.2, random_state=42)
    
    # Entraînement
    history = recommender.train_model(train_data.values, test_data.values)
    
    # Prédictions et recommandations
    predicted_ratings = recommender.model.predict(test_data.values)
    
    # Test avec différents utilisateurs
    test_users = [0, 6436, 10000]
    for user_idx in test_users:
        print(f"\nRecommandations pour l'utilisateur {user_idx}:")
        recommended_ids = recommender.get_recommendations(
            user_idx, 
            predicted_ratings, 
            test_data.values
        )
        recommendations = recommender.get_joke_text(recommended_ids)
        for joke_id, row in zip(recommended_ids, recommendations.itertuples()):
            print(f"\nBlague {joke_id}:")
            print(row.text)
            print("-" * 50)

if __name__ == "__main__":
    main()