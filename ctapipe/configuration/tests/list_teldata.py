from time import sleep
import threading

    
class ListTelda(threading.Thread):
    def __init__(self,_conf):
        super(ListTelda, self).__init__()
        self.next_instance = None
        self.conf = _conf
        
    def init(self):
        print("--- ListTelda init ---")
        self.next_instance = self.__dict__["next_instance"]

    def run(self):
        event = (yield)
        print("---ListTelData do_it start event id",
               event.dl0.event_id , "---")
        sleep(1)
        res = list(event.dl0.tels_with_data)
        print("---ListTelData do_it stop---")
        print(res)

        #self.next_instance.do_it(res,event.dl0.event_id)
    
    def finish(self):
        print("--- ListTelda finish ---")
        self.next_instance.finish()