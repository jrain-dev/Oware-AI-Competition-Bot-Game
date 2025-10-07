import csv

class DataLogger:
    def __init__(self, filename="training_log.csv"):
        self.filename = filename
        try:
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                # Defines header
                writer.writerow([
                    "Episode",
                    "Winner",
                    "Final_Score_P0",
                    "Final_Score_P1",
                    "Exploration_Rate"
                ])
        except IOError as e:
            print(f"Error initializing log file: {e}")
            raise

    def log_episode(self, episode_data):
        # Append episode data to the CSV file
        try:
            with open(self.filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode_data.get("episode", ""),
                    episode_data.get("winner", ""),
                    episode_data.get("score_p0", ""),
                    episode_data.get("score_p1", ""),
                    episode_data.get("epsilon", "")
                ])
        except IOError as e:
            print(f"Error writing to log file: {e}")