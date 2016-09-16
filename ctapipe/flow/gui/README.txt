Open port for GUI if pipeline is running on remote system
prompt> sudo iptables -A INPUT -p tcp  --dport 5565 -j ACCEPT


Create images_rc.py from images.qrc:

prompt%> pyrcc4 -py3 -o images_rc.py images.qrc
