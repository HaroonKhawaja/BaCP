import os

class Logger(object):
    def __init__(self, model_name, learning_type):
        self.log_records_folder = './log_records'
        self.model_name = model_name
        
        self.learning_type = learning_type
        self.log_dir = os.path.join(self.log_records_folder, self.model_name, self.learning_type)   

        os.makedirs(self.log_dir, exist_ok=True)    # ./log_records/model_name/learning_type
    
    def create_log(self):
        existing_files = [file_name for file_name in os.listdir(self.log_dir) if file_name.startswith('run')]
        run_count = len(existing_files) + 1

        train_log_file_path = os.path.join(self.log_dir, f'run_{run_count}.log')    
        self.train_log_file_path = train_log_file_path  # ./log_records/model_name/learning_type/run_{run_count}.log
        
        with open(train_log_file_path, 'w') as log_file:
            log_file.write(f"Model : {self.model_name} - Learning Type: {self.learning_type}\n")

        print(f"[LOGGER] Log file created at location: {self.train_log_file_path}")

    def log_hyperparameters(self, log_info):
        config = 'Configuration:\n'
        for key, value in log_info.items():
            config += f'{key}: {value}\n'
        config += '\n'
        with open(self.train_log_file_path, 'a') as log_file:
                log_file.write(config)
                        
    def log_epochs(self, log_epoch):
        with open(self.train_log_file_path, 'a') as log_file:
                log_file.write(log_epoch)
             