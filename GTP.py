import sys
from Engine import Engine
from Board import *


# Checkt ob Großbuchstabe übergeben wurde, berechnet int Wert der x-Koordinate, sonst mit kleinem Buchstaben
# I wird dabei (in beiden Fällen) ausgeschlossen
def coords_from_str(string):
    if 91 > ord(string[0]) > 64:
        x = ord(string[0]) - ord('A')
    else:
        x = ord(string[0]) - ord('a')
    if x >= 9:
        x -= 1
    y = int(string[1:]) - 1
    return x, y

def colour_from_str(string):
    if 'W' in string or 'w' in string:
        return Stone.White
    else:
        return Stone.Black

# zurücktransformieren der Koordinaten in Buchstaben
def str_from_coords(x, y):
    if x >= 8:
        x += 1
    return chr(ord('A')+x) + str(y+1)

class GTP:
    def __init__(self, engine):
        self.engine = engine


    def tell_client(self, s):
        sys.stdout.write('= ' + s + '\n\n')
        sys.stdout.flush()

    def error_client(self, s):
        sys.stdout.write('? ' + s + '\n\n')
        sys.stdout.flush()

    def list_commands(self):
        commands = ["protocol_version", "name", "version", "boardsize", "clearboard", "komi", "play", "genmove",
                    "list_commands", "quit"]
        self.tell_client("\n".join(commands))

    def set_board_size(self, line):
        try:
            boardsize = int(line.split()[1])
        except ValueError:
            print("No valid number")
            return
        self.engine.create_board(boardsize)

    def clear_board(self):
        self.engine.board.clear()

    def set_komi(self, line):
        try:
            komi = float(line.split()[1])
        except ValueError:
            print("No valid number")
            return
        self.engine.set_komi(komi)

    def stone_played(self, line):
        stone = colour_from_str(line.split()[1])
        if "pass" in line.split()[2]:
            self.engine.player_passed(stone)
        else:
            x, y = coords_from_str(line.split()[2])
            self.engine.stone_played(x, y, stone)

    def gen_move(self, line):
        try:
            stone = colour_from_str(line.split()[1])
        except ValueError:
            print("No valid input")
            return
        self.engine.play_legal_move(self.engine.board, stone)

    def show_board(self):
        self.engine.board.show()




    def quit(self):
        exit(0)




    def loop(self):
        while True:
            line = sys.stdin.readline().strip()
            print("Client sent: " + line)

            if line.startswith("protocol_version"):  # GTP protocol version
                self.tell_client("2")
            elif line.startswith("name"):  # Engine name
                self.tell_client(self.engine.name())
            elif line.startswith("version"):  # Engine version
                self.tell_client(self.engine.version())
            elif line.startswith("list_commands"):  # List supported commands
                self.list_commands()
            elif line.startswith("quit"):  # Quit
                self.quit()
            elif line.startswith("boardsize"):  # Board size
                self.set_board_size(line)
            elif line.startswith("clear_board"):  # Clear board
                self.clear_board()
            elif line.startswith("komi"):  # Set komi
                self.set_komi(line)
            elif line.startswith("play"):  # A stone has been placed
                self.stone_played(line)
            elif line.startswith("genmove"):  # We must generate a move
                self.gen_move(line)
            elif line.startswith("showboard"):
                self.show_board()
            else:
                self.error_client("Didn't recognize")

def test():
    engine = Engine(5)
    gtp = GTP(engine)
    # gtp.list_commands()
    gtp.loop()

test()