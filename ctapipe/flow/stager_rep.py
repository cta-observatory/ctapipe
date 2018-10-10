
class StagerRep():
    """
    class representing steps status.
    Parameters
    ----------
    name  : str
    next_steps : list(str)
    running : bool
    nb_job_done : int
    """
    STAGER = 1
    PRODUCER = 2
    CONSUMER = 3

    def __init__(self, name, next_steps=None, running=0,
                 nb_job_done=0, queue_length=0, 
                 nb_process=1, step_type=STAGER):
        self.type = step_type
        self.name = name
        self.next_steps = next_steps or []
        self.running = running
        self.nb_job_done = nb_job_done
        self.queue_length = queue_length
        self.nb_process = nb_process

    def __repr__(self):
        """  called by the repr() built-in function and by string conversions
        (reverse quotes) to compute the "official" string representation of
        an object.  """
        return (self.name + ' running: ' +
                str(self.running) + '-> nb_job_done: ' +
                str(self.nb_job_done) + '-> next_steps:' +
                str(self.next_steps) + '-> queue_length:' +
                str(self.queue_length) + '-> nb_process:' +
                str(self.nb_process))

    def get_statistics(self):
        """
        return
        ======
        str containing step name (without its process extension) and the number
        of jobs it did.
        """
        return (
            self.name.split('$$process')[0] +
            ' number of jobs done: ' +
            str(self.nb_job_done)
        )
