#/bin/env python3

'''
Created on Apr 13, 2015

@author: jacquem

https://docs.python.org/2/library/unittest.html

A testcase is created by subclassing unittest.TestCase. 
The three individual tests are defined with methods whose names start with the letters test.
This naming convention informs the test runner about which methods represent tests.

The crux of each test is a call to assertEqual() to check for an expected result;
assertTrue() or assertFalse() to verify a condition; or assertRaises()
to verify that a specific exception gets raised. These methods are used instead 
of the assert statement so the test runner can accumulate all test results and produce a report.

The setUp() and tearD from StringIO import StringIO

    saved_stdout = sys.stdout
    try:
        out = StringIO()
        sys.stdout = out
        foo()
        output = out.getvalue().strip()
        assert output == 'hello world!'
    finally:
        sys.stdout = saved_stdoutown() methods allow you to define instructions that will be
 executed before and after each test method. They are covered in more details in
  the section Organizing test code.
'''

import unittest
import sys
from io import StringIO
from configuration.configuration import Configuration, ConfigurationException
#from sys import implementation

class C():
    pass
    
class Test(unittest.TestCase):

    def testArgsParser(self):

        # This method test argument receive on command line
        # To emulate argument on command line:
        # 1/ backup sys.argv
        # 2/ Set sys.argv with program name and arguments
        # 3/ restore sys.argv         
        backup =  sys.argv
        sys.argv="executable --name CTA -R --sum".split()
        print("--- Test command line arguments parsing")
        conf = Configuration()

        conf.add_argument("--name", dest="name", action='store') 
        conf.add_argument("-R", dest="rtrue", action='store_true')
        conf.add_argument("--sum", dest="sfalse", action='store_false')
        conf.add_argument("--miss", dest="miss_false", action='store_false')
    
        c=C()
        conf.parse_args(namespace=c)
        self.assertEqual(conf.get("name"),"CTA","Configuration.get error")
        self.assertEqual(conf.get("rtrue"),True,"Configuration.get error")
        self.assertEqual(conf.get("sfalse"),False,"Configuration.get error")
        
        # DEFAULT SECTION entry can be accessed directly as a class member
        self.assertEqual(conf.name ,"CTA","Configuration.get error")   

        conf.add_argument("--false", dest="test", action='store',required=True)
        try: 
            conf.parse_args()
            assert()  # Must prodices an error because --false is required and not supplied
        except:
            pass
        print("--- command line arguments parsing listing")
        
        conf.list()
        
        # test copy constructor
        copy_conf = Configuration(conf)
        print("--- copy construction  arguments parsing listing")
        self.assertEqual(copy_conf.get("name"),"CTA","Configuration.get error")
        self.assertEqual(copy_conf.get("rtrue"),True,"Configuration.get error")
        self.assertEqual(copy_conf.get("sfalse"),False,"Configuration.get error")
        self.assertEqual(copy_conf.get("miss_false"),True,"Configuration.get error")
        sys.argv = backup

        
       
    def testName(self):
        print("--- Test manulally add/get option")
        conf = Configuration()
        # test with section
        conf.add("key","value",comment="Mon commentaire", section="section")
        self.assertEqual(conf.get("key","section"),"value","Configuration.get error")
        self.assertEqual(conf.getComment("key","section"),"Mon commentaire","Configuration.get error")
        self.assertEquals(conf.has_key("key","section"),True)
        
        #test with DEFAULT section
        conf.add("key","value")
        self.assertEqual(conf.get("key"),"value","Configuration.get error")
        
        #test with float value
        conf.add("height",1.76)
        self.assertEqual(float(conf.get("height")),1.76,"Configuration error")
        
        #test with float exponential value
        conf.add("height_expo",1.76e32)
        self.assertEqual(float(conf.get("height_expo")),1.76e32,"Configuration.get error")

        # test to access a none existing value
        self.assertEquals(conf.get("none existing"),None)
        self.assertEquals(conf.has_key("none existing"),False)
        
        # test duplicate key
        try: 
            conf.add("height_expo",1.76e32)
            assert()
        except:
            pass
        
        # Test copy constructor
        print("--- List manulally add/get option")
        conf.list()
        
        print("--- Test copy constructor")
        copy_config = Configuration(conf)
        print("--- List copied constructor conf")
        copy_config.list()
        self.assertEqual(copy_config.get("key"),"value","Configuration error")
        self.assertEqual(float(copy_config.get("height")),1.76,"Configuration error")
        self.assertEqual(float(copy_config.get("height_expo")),1.76e32,"Configuration.get error")
         

    def testConfigParser(self):
        print("--- Test Read/Write Conf")
        conf = Configuration()
        conf._entries['DEFAULT'] = {'ServerAliveInterval': ('45','comment'),
                      'Compression': ('yes',''),
                      'CompressionLevel': ('9','comment')}
        conf._entries['bitbucket.org'] = {}

        conf._entries['bitbucket.org']['User'] = ('hg','')
        conf._entries['topsecret.server.com'] = {}
        topsecret = conf._entries['topsecret.server.com']
        topsecret['Port'] = ('50022', '')     # mutates the parser
        topsecret['ForwardX11'] = ('no','')  # same here
        conf._entries['DEFAULT']['ForwardX11'] = ('yes', '')
        conf.add("AnotherKey","AnotherValue",comment="Another comment")
        # Write to text file

        print("--- Test Write Conf to INI format")
        conf.write('example.ini',Configuration.INI)
        readed_conf = Configuration()
        readed_conf.read('example.ini',Configuration.INI)  
        print("--- List read Conf to INI format")
        readed_conf.list()
        self.assertEqual(readed_conf.get("ServerAliveInterval"),'45',"Configuration.get error")
        self.assertEqual(readed_conf.getComment("ServerAliveInterval"),'comment',"Configuration.get error")

        # test Fits Data implemeation
        print("--- Test Write Conf to FITS data table format")
        conf.write('example.fits')
        readed_conf = Configuration()
        readed_conf.read('example.fits')
        self.assertEqual(readed_conf.get("ServerAliveInterval"),'45',"Configuration.get error")
        self.assertEqual(readed_conf.getComment("ServerAliveInterval"),'comment',"Configuration.get error")
        print("--- List read Conf to Fits table data format")
        readed_conf.list()
        
   
        # test Fits Header implemeation
        print("--- Test Write Conf to Header fits  format")
        conf.write('example_HEADER.fits',implementation=Configuration.HEADERIMPL)
        readed_conf = Configuration()
        readed_conf.read('example_HEADER.fits',implementation=Configuration.HEADERIMPL)
        self.assertEqual(readed_conf.get("ServerAliveInterval"),'45',"Configuration.get error")
        self.assertEqual(readed_conf.getComment("ServerAliveInterval"),'comment',"Configuration.get error")
        print("--- List read Conf to Fits Header format")
        readed_conf.list()
   
   


        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    
    unittest.main()
    # The following 3 lines allow command line arguments to be pass to
    # Configuration constructor
    
    #runner = unittest.TextTestRunner()
    #itersuite = unittest.TestLoader().loadTestsFromTestCase(Test)
    #runner.run(itersuite)

    
    