import sys
from Engine import Engine, IntelligentEngine, FilterEngine
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
    def __init__(self, engine, logfile):
        self.engine = engine
        self.fclient = sys.stdout
        sys.stdout = sys.stderr = open(logfile, 'w')
        print("GTP: Redirected stdout to logfile.")

    def tell_client(self, s):
        self.fclient.write('= ' + s + '\n\n')
        self.fclient.flush()
        print("GTP: Told client: " + s)

    def error_client(self, s):
        self.fclient.write('? ' + s + '\n\n')
        self.fclient.flush()
        print("GTP: Sent error message to client: " + s)

    def list_commands(self):
        commands = ["protocol_version", "name", "version", "boardsize", "clearboard", "komi", "play", "genmove",
                    "list_commands", "known_commands", "quit"]
        self.tell_client("\n".join(commands))

    def known_commands(self):
        commands = ["protocol_version", "name", "version", "boardsize", "clearboard", "komi", "play", "genmove",
                    "list_commands", "known_commands", "quit"]
        self.tell_client("\n".join(commands))

    def set_board_size(self, line):
        try:
            boardsize = int(line.split()[1])
        except ValueError:
            self.error_client("Unsupported board size")
            return
        print("GTP: setting board size to", boardsize)
        self.engine.create_board(boardsize)
        self.tell_client("")

    def clear_board(self):
        self.engine.board.clear()
        print("GTP: clearing board")
        self.tell_client("")

    def set_komi(self, line):
        try:
            komi = float(line.split()[1])
        except ValueError:
            print("No valid number")
            self.tell_client("")
            return
        print("GTP: setting komi to", komi)
        self.engine.set_komi(komi)
        self.tell_client("")

    def stone_played(self, line):
        stone = colour_from_str(line.split()[1])
        if ("pass" in line.split()[2]):
            print("GTP: ", colour_names[stone], " passed")
            self.engine.player_passed(stone)
        else:
            x, y = coords_from_str(line.split()[2])
            print("GTP: ", colour_names[stone], " has played at (%d,%d)" % (x, y))
            self.engine.stone_played(x, y, stone)
        self.tell_client("")

    def gen_move(self, line):
        try:
            stone = colour_from_str(line.split()[1])
        except ValueError:
            print("No valid input")
            self.tell_client("")
            return
        print("GTP: asked to generate move for", colour_names[stone])

        coords = self.engine.play_legal_move(self.engine.board, stone)
        if coords != "pass":
            x, y = coords
            print("GTP: engine generated move (%d,%d) for" % (x, y), colour_names[stone])
            self.tell_client(str_from_coords(x, y))
        else:
            print("GTP: engine passed")
            self.tell_client("pass")

    def show_board(self):
        self.engine.board.show()

    def quit(self):
        print("GTP: Quitting")
        self.tell_client(" ")
        self.fclient.close()  # Close log file
        exit(0)

    def loop(self):
        while True:
            line = sys.stdin.readline().strip()
            line = line.lower()
            if len(line) == 0: return
            print("Client sent: " + line)

            if line.startswith("protocol_version"):  # GTP protocol version
                self.tell_client("2")
            elif line.startswith("name"):  # Engine name
                self.tell_client(self.engine.name())
            elif line.startswith("version"):  # Engine version
                self.tell_client(self.engine.version())
            elif line.startswith("list_commands"):  # List supported commands
                self.list_commands()
            elif line.startswith("known_commands"):  # List known commands
                self.known_commands()
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
                self.error_client("command unknown: " + line)

def run():
    engine = FilterEngine(9)
    logfile = "log_3.txt"
    gtp = GTP(engine, logfile)
    # gtp.list_commands()
    gtp.loop()

run()