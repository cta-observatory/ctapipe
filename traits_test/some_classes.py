#!/usr/bin/env python
from traitlets import *
from ctapipe.core import Tool

class A(HasTraits):
    foo = Int(default_value=7)

class B(A):
    bar = Long(default_value=10)


class MyTool(Tool):
    name = "mytool"
    description = "do some things and stuff"
    classes = List([A, B])

    def setup(self):
        print(self.config)

    def start(self):
        print('in start')

    def finish(self):
        print('in finish')


def main():
    print('in main')
    tool = MyTool()
    tool.run()

if __name__ == "__main__":
    main()
