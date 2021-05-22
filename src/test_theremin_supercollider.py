from pythonosc import udp_client
client = udp_client.SimpleUDPClient("127.0.0.1", 57120) #default ip and port for SC
client.send_message("/freq", 220) # set the frequency at 440
client = udp_client.SimpleUDPClient("127.0.0.1", 57120) #default ip and port for SC
client.send_message("/amp", 0.4) # set the frequency at 440
