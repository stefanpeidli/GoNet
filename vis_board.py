# This script converts a board into a Latex-Visual:

from Board import Board
import numpy as np
""" 
The Latex code should look like:

begin{figure}[htp]
	centering
	begin{tikzpicture}[scale=0.5]
        \draw[shift={(0.5,0.5)}] (0,0) grid (9,9);
        foreach x in {1,...,7,8,9} {
            node[left] at (0,x) {$x$};
        }
        node at (1,10) {A};
        node at (2,10) {B};
        node at (3,10) {C};
        node at (4,10) {D};
        node at (5,10) {E};
        node at (6,10) {F};
        node at (7,10) {G};
        node at (8,10) {H};
        node at (9,10) {K};
        foreach x in {2,3,4,5} {
            foreach \y in {3,4} {
                fill (x,\y) circle(0.3);
                fill (\y,x) circle(0.3);
            }
        }
        \draw (8,7) circle(0.3);
	\end{tikzpicture}
\end{figure}

"""


def vis_latex(board):
    if type(board) is Board:
        b = board.vertices
    else:
        b = board * 1
    print(b)
    a = "\ "
    a = a[0]
    for row in range(9):
        for col in range(9):
            s = b[row, col]
            if s == 1:
                print("\draw (" + str(col+1) + "," + str(9-row) + ") circle(0.3);")
            elif s == -1:
                print(a + "fill (" + str(col+1) + "," + str(9-row) + ") circle(0.3);")


#b = Board(9)
#b.play_stone(0, 1, 1)
b = np.round(np.random.uniform(-1, 1, (9, 9)), 0)
vis_latex(b)
