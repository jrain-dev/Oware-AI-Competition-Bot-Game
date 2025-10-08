import csv

class DataLogger:
    def __init__(self, filename="training_log.csv", columns=None, data_keys=None):
        
        self.filename = filename
        # Default header 
        if columns is None:
            self.columns = [
                "Episode",
                "Winner",
                "Final_Score_P0",
                "Final_Score_P1",
                "Exploration_Rate",
            ]
        else:
            self.columns = columns

        if data_keys is None:
            self.data_keys = ["episode", "winner", "score_p0", "score_p1", "epsilon"]
        else:
            self.data_keys = data_keys

        try:
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.columns)
        except IOError as e:
            print(f"Error initializing log file: {e}")
            raise

    def log_episode(self, episode_data):
        try:
            with open(self.filename, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [episode_data.get(k, "") for k in self.data_keys]
                writer.writerow(row)
        except IOError as e:
            print(f"Error writing to log file: {e}")