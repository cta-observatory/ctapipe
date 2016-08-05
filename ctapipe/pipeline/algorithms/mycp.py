from time import sleep
import threading
import subprocess
import os
from ctapipe.core import Component
from traitlets import Unicode


class MyCp(Component):
    """`MyCp` class represents a Producer for pipeline.
        It lists all files prensent in source_dir directory.
        It add out_extention and copies each file to outdir_dir
    """
    output_dir = Unicode('/tmp/test/in', help='directory receving produced data').tag(
    config=True, allow_none=False)
    out_extension = Unicode('type1', help='directory receving produced data').tag(
    config=True, allow_none=False)

    def init(self):
        if self.output_dir == None or self.out_extension == None:
            self.log.error("MyCp :configuration error ")
            self.log.error('output_dir: {} out_extension: {}'
            .format(self.output_dir,self.out_extension))
            return False

        if not os.path.exists(self.output_dir):
            try:
                os.makedirs(self.output_dir)
            except OSError as e:
                self.log.error(
                    "{} : could not create output directory {}: {}".format(self.section_name,  self.output_dir, e))
                return False
        return True

    def run(self,input_info):
        input_path, input_file  = input_info
        self.log.info('Mycp start {}'.format(input_file))
        output_file = self.output_dir+"/"+input_file+self.out_extension
        cmd = ['cp',input_path+"/"+input_file,output_file]
        proc = subprocess.Popen(cmd)
        proc.wait()
        return output_file



    def finish(self):
        self.log.info("--- MyCp finish ---")
