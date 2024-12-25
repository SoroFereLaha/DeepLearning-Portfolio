import pandas as pd
import os
from bs4 import BeautifulSoup, Comment
import zipfile
import warnings
warnings.filterwarnings('ignore')

class JesterDataProcessor:
    def __init__(self, base_path):
        self.base_path = base_path
        # Liste des blagues retirées (mentioned as removed in the documentation)
        self.removed_jokes = {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 20, 27, 
                            31, 43, 51, 52, 61, 73, 80, 100, 116}
        # Gauge set jokes
        self.gauge_set = {7, 8, 13, 15, 16, 17, 18, 19}

    def extract_joke_text(self, html_content):
        """Extrait le texte d'une blague depuis le HTML."""
        soup = BeautifulSoup(html_content, 'html.parser')
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        joke_text = ""
        
        for idx, comment in enumerate(comments):
            if "begin of joke" in comment:
                current = comment.next_element
                while current:
                    if isinstance(current, Comment) and "end of joke" in current:
                        break
                    if isinstance(current, str) and current.strip():
                        joke_text += current.strip() + " "
                    current = current.next_element
                break
        
        return joke_text.strip()

    def process_dataset_1(self):
        """Traite le Dataset 1 (4.1 million ratings)."""
        print("Processing Dataset 1...")
        
        # Traiter les textes des blagues
        jokes_folder = os.path.join(self.base_path, "dataset_1/jokes")
        jokes_data = []
        
        for i in range(1, 101):
            try:
                with open(os.path.join(jokes_folder, f"init{i}.html"), "r", encoding="utf-8") as f:
                    joke_text = self.extract_joke_text(f.read())
                    jokes_data.append({
                        'joke_id': i,
                        'text': joke_text,
                        'dataset': 1,
                        'removed': i in self.removed_jokes,
                        'gauge_set': i in self.gauge_set
                    })
            except Exception as e:
                print(f"Error processing joke {i}: {str(e)}")
        
        # Traiter les évaluations
        ratings_parts = []
        for i in range(1, 4):
            try:
                df = pd.read_excel(os.path.join(self.base_path, f"dataset_1/ratings_{i}.xls"))
                df['dataset'] = 1
                df['part'] = i
                ratings_parts.append(df)
            except Exception as e:
                print(f"Error processing ratings part {i}: {str(e)}")
        
        return pd.DataFrame(jokes_data), pd.concat(ratings_parts, ignore_index=True)

    def process_dataset_3(self):
        """Traite le Dataset 3 (2.3 million ratings)."""
        print("Processing Dataset 3...")
        
        # Traiter les textes des blagues
        jokes_data = []
        jokes_df = pd.read_excel(os.path.join(self.base_path, "dataset_3/jokes.xls"))
        
        for index, row in jokes_df.iterrows():
            jokes_data.append({
                'joke_id': index + 1,
                'text': row[0],
                'dataset': 3,
                'removed': (index + 1) in self.removed_jokes,
                'gauge_set': (index + 1) in self.gauge_set
            })
        
        # Traiter les évaluations
        ratings_df = pd.read_excel(os.path.join(self.base_path, "dataset_3/ratings.xls"))
        ratings_df['dataset'] = 3
        
        return pd.DataFrame(jokes_data), ratings_df

    def process_dataset_4(self):
        """Traite le Dataset 4 (100k new ratings)."""
        print("Processing Dataset 4...")
        
        # Traiter les textes des blagues
        jokes_data = []
        jokes_df = pd.read_excel(os.path.join(self.base_path, "dataset_4/jokes.xls"))
        
        for index, row in jokes_df.iterrows():
            jokes_data.append({
                'joke_id': index + 1,
                'text': row[0],
                'dataset': 4,
                'removed': (index + 1) in self.removed_jokes,
                'gauge_set': (index + 1) in self.gauge_set
            })
        
        # Traiter les évaluations
        ratings_df = pd.read_excel(os.path.join(self.base_path, "dataset_4/ratings.xls"))
        ratings_df['dataset'] = 4
        
        return pd.DataFrame(jokes_data), ratings_df

    def process_all_datasets(self):
        """Combine tous les datasets."""
        all_jokes = []
        all_ratings = []
        
        # Traiter chaque dataset
        for processor in [self.process_dataset_1, self.process_dataset_3, self.process_dataset_4]:
            jokes_df, ratings_df = processor()
            all_jokes.append(jokes_df)
            all_ratings.append(ratings_df)
        
        # Combiner les données
        combined_jokes = pd.concat(all_jokes, ignore_index=True)
        combined_ratings = pd.concat(all_ratings, ignore_index=True)
        
        # Nettoyer les évaluations (remplacer 99 par NaN)
        combined_ratings = combined_ratings.replace(99, pd.NA)
        
        return combined_jokes, combined_ratings

    def save_processed_data(self, jokes_df, ratings_df, output_dir):
        """Sauvegarde les données traitées."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Sauvegarder les données
        jokes_df.to_csv(os.path.join(output_dir, "all_jokes.csv"), index=False)
        ratings_df.to_csv(os.path.join(output_dir, "all_ratings.csv"), index=False)
        
        # Créer un rapport des statistiques
        with open(os.path.join(output_dir, "statistics.txt"), "w") as f:
            f.write("Jester Dataset Statistics\n")
            f.write("=======================\n\n")
            
            f.write("Jokes Statistics:\n")
            f.write(f"Total number of unique jokes: {len(jokes_df)}\n")
            f.write(f"Jokes per dataset:\n{jokes_df['dataset'].value_counts().to_string()}\n\n")
            
            f.write("Ratings Statistics:\n")
            f.write(f"Total number of ratings: {len(ratings_df)}\n")
            f.write(f"Ratings per dataset:\n{ratings_df['dataset'].value_counts().to_string()}\n")
            
            # Vérification de l'existence de 'user_id' dans ratings_df
            if 'user_id' in ratings_df.columns:
                f.write(f"Number of unique users: {ratings_df['user_id'].nunique()}\n")
            else:
                f.write(f"Number of unique users: {ratings_df.shape[0]}\n")
            
            f.write("\nGauge Set Jokes: {}\n".format(sorted(self.gauge_set)))
            f.write("Removed Jokes: {}\n".format(sorted(self.removed_jokes)))

# Utilisation
if __name__ == "__main__":
    base_path = os.path.join(os.getcwd(), "base_path") # À modifier selon votre configuration
    output_dir = "processed_jester_data"
    
    processor = JesterDataProcessor(base_path)
    jokes_df, ratings_df = processor.process_all_datasets()
    processor.save_processed_data(jokes_df, ratings_df, output_dir)
