from IPython import display 
from collections import deque

#Used to keep track of how the training is going
#What current score and so forth.

class Logger:
    
    def __init__(self, n_rows, header=None):
        self.log_queue = deque(maxlen=n_rows)
        self.header = header
    
    def log(self, text):
        self.log_queue.append(text)
        display.clear_output(wait=True)
        if self.header != None:
            print(self.header)
        print("\n".join(self.log_queue), flush=True)
