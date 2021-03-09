import factorio_rcon

client = factorio_rcon.RCONClient("127.0.0.1", 7777, "zzz")
response = client.send_command("/sc rcon.print(game.tick) game.ticks_to_run = 12 rcon.print(game.tick)")

print (response)
