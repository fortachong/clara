from pythonosc import udp_client
client = udp_client.SimpleUDPClient("127.0.0.1", 57121) #default ip and port for SC
client.send_message("/main/f", 220) # set the frequency at 440
client = udp_client.SimpleUDPClient("127.0.0.1", 57121) #default ip and port for SC
client.send_message("/main/a", 0.4) # set the frequency at 440
