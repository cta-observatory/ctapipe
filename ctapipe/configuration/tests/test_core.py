import pytest
import sys

from ..core import Configuration, ConfigurationException

class C():
  pass

def test_argsParser():

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
    assert conf.get("name") == "CTA"
    assert conf.get("rtrue") == True
    assert conf.get("sfalse") == False

    # DEFAULT SECTION entry can be accessed directly as a class member
    assert conf.name == "CTA"   

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
    assert copy_conf.get("name") == "CTA"
    assert copy_conf.get("rtrue") == True
    assert copy_conf.get("sfalse") == False
    assert copy_conf.get("miss_false") == True
    sys.argv = backup

      
     
def test_name():
    print("--- Test manulally add/get option")
    conf = Configuration()
    # test with section
    conf.add("key","value",section="section",comment="Mon commentaire")
    conf.list()
    assert conf.get("key","section") == "value"
    assert conf.getComment("key","section") == "Mon commentaire"
    assert conf.has_key("key","section") == True
    
    #test with DEFAULT section
    conf.add("key","value")
    assert conf.get("key") == "value"
    
    #test with float value
    conf.add("height",1.76)
    assert float(conf.get("height")) == 1.76
    
    #test with float exponential value
    conf.add("height_expo",1.76e32)
    assert float(conf.get("height_expo")) == 1.76e32

    # test to access a none existing value
    assert conf.get("none existing") == None
    assert conf.has_key("none existing") == False
    
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
    assert copy_config.get("key") =="value"
    assert float(copy_config.get("height")) ==1.76
    assert float(copy_config.get("height_expo")) ==1.76e32
       

def test_configParser():
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
    assert readed_conf.get("ServerAliveInterval") =='45'
    assert readed_conf.getComment("ServerAliveInterval") =='comment'

    # test Fits Data implemeation
    print("--- Test Write Conf to FITS data table format")
    conf.write('example.fits')
    readed_conf = Configuration()
    readed_conf.read('example.fits')
    assert readed_conf.get("ServerAliveInterval") == '45'
    assert readed_conf.getComment("ServerAliveInterval") == 'comment'
    print("--- List read Conf to Fits table data format")
    readed_conf.list()
    

    # test Fits Header implemeation
    print("--- Test Write Conf to Header fits  format")
    conf.write('example_HEADER.fits',implementation=Configuration.HEADERIMPL)
    readed_conf = Configuration()
    readed_conf.read('example_HEADER.fits',implementation=Configuration.HEADERIMPL)
    assert readed_conf.get("ServerAliveInterval") == '45'
    assert readed_conf.getComment("ServerAliveInterval") == 'comment'
    print("--- List read Conf to Fits Header format")
    readed_conf.list()


if __name__ == '__main__':
  print("main")
