from time import sleep
import threading

    
class ListTelda():
    def __init__(self,configuration=None):
        self.configuration = configuration
        
        
    def init(self):
        print("--- ListTelda init ---")

    def run(self,event):
        sleep(1)
        res = list(event.dl0.tels_with_data)
        print("--- LsitTeleData res ---" , res)
        return res

        #self.next_instance.do_it(res,event.dl0.event_id)
    
    def finish(self):
        print("--- ListTelda finish ---")