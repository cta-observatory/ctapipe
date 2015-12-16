from time import sleep
import threading

    
class ListTelda():
    def __init__(self,configuration=None):
        self.configuration = configuration
        
        
    def init(self):
        print("--- ListTelda init ---")

    def run(self,event):
        sleep(.2)
        res = list(event.dl0.tels_with_data)
        print("--- ListTeleData res ---" , res)
        return res

    
    def finish(self):
        print("--- ListTelda finish ---")