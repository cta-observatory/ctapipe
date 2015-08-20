class ListTelda():
    
    def __init__(self,_conf):
        self.next_instance = None
        self.conf = _conf
        
    def init(self):
        print("--- ListTelda init ---")
        self.next_instance = self.__dict__["next_instance"]

    def do_it(self,event):
        res = list(event.dl0.tels_with_data)
        self.next_instance.do_it(res)
    
    def finish(self):
        print("--- ListTelda finish ---")
        self.next_instance.finish()