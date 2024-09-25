import argparse
import sys
import pdb
import copy
import heapq
#====================================================================================

char_single = '2'

class Piece:
    """
    This represents a piece on the Hua Rong Dao puzzle.
    """

    def __init__(self, is_2_by_2, is_single, coord_x, coord_y, orientation):
        """
        :param is_2_by_2: True if the piece is a 2x2 piece and False otherwise.
        :type is_2_by_2: bool
        :param is_single: True if this piece is a 1x1 piece and False otherwise.
        :type is_single: bool
        :param coord_x: The x coordinate of the top left corner of the piece.
        :type coord_x: int
        :param coord_y: The y coordinate of the top left corner of the piece.
        :type coord_y: int
        :param orientation: The orientation of the piece (one of 'h' or 'v') 
            if the piece is a 1x2 piece. Otherwise, this is None
        :type orientation: str
        """

        self.is_2_by_2 = is_2_by_2
        self.is_single = is_single
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.orientation = orientation
        
        self.occupied = set()
        if self.is_2_by_2:
            self.occupied.add((self.coord_y, self.coord_x))
            self.occupied.add((self.coord_y + 1, self.coord_x))
            self.occupied.add((self.coord_y, self.coord_x + 1))
            self.occupied.add((self.coord_y + 1, self.coord_x + 1))
        elif not self.is_single:
            if self.orientation == 'h':
                self.occupied.add((self.coord_y, self.coord_x))
                self.occupied.add((self.coord_y, self.coord_x + 1))
            elif self.orientation == 'v':
                self.occupied.add((self.coord_y, self.coord_x))
                self.occupied.add((self.coord_y + 1, self.coord_x))
        else:
            self.occupied.add((self.coord_y, self.coord_x))



    def set_coords(self, coord_x, coord_y):
        """
        Move the piece to the new coordinates. 

        :param coord: The new coordinates after moving.
        :type coord: int
        """

        self.coord_x = coord_x
        self.coord_y = coord_y

        self.occupied.clear()
        if self.is_2_by_2:
            self.occupied.add((self.coord_y, self.coord_x))
            self.occupied.add((self.coord_y + 1, self.coord_x))
            self.occupied.add((self.coord_y, self.coord_x + 1))
            self.occupied.add((self.coord_y + 1, self.coord_x + 1))
        elif not self.is_single:
            if self.orientation == 'h':
                self.occupied.add((self.coord_y, self.coord_x))
                self.occupied.add((self.coord_y, self.coord_x + 1))
            elif self.orientation == 'v':
                self.occupied.add((self.coord_y, self.coord_x))
                self.occupied.add((self.coord_y + 1, self.coord_x))
        else:
            self.occupied.add((self.coord_y, self.coord_x))

    def __repr__(self):
        return '{} {} {} {} {}'.format(self.is_2_by_2, self.is_single, \
            self.coord_x, self.coord_y, self.orientation)
    

class Board:
    """
    Board class for setting up the playing board.
    """

    def __init__(self, height, pieces):
        """
        :param pieces: The list of Pieces
        :type pieces: List[Piece]
        """

        self.width = 4
        self.height = height
        self.pieces = pieces

        # self.grid is a 2-d (size * size) array automatically generated
        # using the information on the pieces when a board is being created.
        # A grid contains the symbol for representing the pieces on the board.
        self.grid = []
        self.blanks = []
        self.__construct_grid()



    # customized eq for object comparison.
    def __eq__(self, other):
        if isinstance(other, Board):
            return self.grid == other.grid
        return False

    def __hash__(self):
        return hash(grid_to_string(self.grid))
    
    def __construct_grid(self):
        """
        Called in __init__ to set up a 2-d grid based on the piece location information.

        """

        for i in range(self.height):
            line = []
            for j in range(self.width):
                line.append('.')
            self.grid.append(line)

        for piece in self.pieces:
            if piece.is_2_by_2:
                self.grid[piece.coord_y][piece.coord_x] = '1'
                self.grid[piece.coord_y][piece.coord_x + 1] = '1'
                self.grid[piece.coord_y + 1][piece.coord_x] = '1'
                self.grid[piece.coord_y + 1][piece.coord_x + 1] = '1'
            elif piece.is_single:
                self.grid[piece.coord_y][piece.coord_x] = char_single
            else:
                if piece.orientation == 'h':
                    self.grid[piece.coord_y][piece.coord_x] = '<'
                    self.grid[piece.coord_y][piece.coord_x + 1] = '>'
                elif piece.orientation == 'v':
                    self.grid[piece.coord_y][piece.coord_x] = '^'
                    self.grid[piece.coord_y + 1][piece.coord_x] = 'v'

        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i][j] == '.':
                    self.blanks.append((i, j))
      
    def display(self):
        """
        Print out the current board.

        """
        for i, line in enumerate(self.grid):
            for ch in line:
                print(ch, end='')
            print()
        print()
        

class State:
    """
    State class wrapping a Board with some extra current state information.
    Note that State and Board are different. Board has the locations of the pieces. 
    State has a Board and some extra information that is relevant to the search: 
    heuristic function, f value, current depth and parent.
    """

    def __init__(self, board, hfn, f, depth, parent=None):
        """
        :param board: The board of the state.
        :type board: Board
        :param hfn: The heuristic function.
        :type hfn: Optional[Heuristic]
        :param f: The f value of current state.
        :type f: int
        :param depth: The depth of current state in the search tree.
        :type depth: int
        :param parent: The parent of current state.
        :type parent: Optional[State]
        """
        self.board = board
        self.hfn = hfn
        self.f = f
        self.depth = depth
        self.parent = parent

    def __eq__(self, other):
        if isinstance(other, State):
            return self.board == other.board
        return False
    
    def __hash__(self):
        return hash(self.board)

def read_from_file(filename):
    """
    Load initial board from a given file.

    :param filename: The name of the given file.
    :type filename: str
    :return: A loaded board
    :rtype: Board
    """

    puzzle_file = open(filename, "r")

    line_index = 0
    pieces = []
    final_pieces = []
    final = False
    found_2by2 = False
    finalfound_2by2 = False
    height_ = 0

    for line in puzzle_file:
        height_ += 1
        if line == '\n':
            if not final:
                height_ = 0
                final = True
                line_index = 0
            continue
        if not final: #initial board
            for x, ch in enumerate(line):
                if ch == '^': # found vertical piece
                    pieces.append(Piece(False, False, x, line_index, 'v'))
                elif ch == '<': # found horizontal piece
                    pieces.append(Piece(False, False, x, line_index, 'h'))
                elif ch == char_single:
                    pieces.append(Piece(False, True, x, line_index, None))
                elif ch == '1':
                    if found_2by2 == False:
                        pieces.append(Piece(True, False, x, line_index, None))
                        found_2by2 = True
        else: #goal board
            for x, ch in enumerate(line):
                if ch == '^': # found vertical piece
                    final_pieces.append(Piece(False, False, x, line_index, 'v'))
                elif ch == '<': # found horizontal piece
                    final_pieces.append(Piece(False, False, x, line_index, 'h'))
                elif ch == char_single:
                    final_pieces.append(Piece(False, True, x, line_index, None))
                elif ch == '1':
                    if finalfound_2by2 == False:
                        final_pieces.append(Piece(True, False, x, line_index, None))
                        finalfound_2by2 = True
        line_index += 1
        
    puzzle_file.close()
    board = Board(height_, pieces)
    goal_board = Board(height_, final_pieces)
    return board, goal_board



def grid_to_string(grid):
    string = ""
    for i, line in enumerate(grid):
        for ch in line:
            string += ch
        string += "\n"
    return string

def print_solution(state):
    """
    Print the solution.

    :param state: The solution state.
    :type state: State
    """
    sys.stdout = open(args.outputfile, 'w')
    solution_path = []
    while state is not None:
        solution_path.append(state)
        state = state.parent
    solution_path.reverse()
    for state in solution_path:
        state.board.display()
        # print("\n")


def dfs(initial_state, goal_board):
    """
    Perform a depth-first search to find a solution.

    :param board: The initial board.
    :type board: Board
    :param goal_board: The goal board.
    :type goal_board: Board
    :return: The solution state.
    :rtype: State
    """

    frontier = [initial_state]
    visited = set()

    while frontier:
        state = frontier.pop()

        if state not in visited:
            visited.add(state)
            if state.board == goal_board:
                return state
            next_steps = generate_next_steps(state)

            for successor in next_steps:
                if successor not in visited:
                    frontier.append(successor)
    return None
        

        
def astar(initial_state, goal_board):
    """
    Perform an A* search to find a solution.

    :param board: The initial board.
    :type board: Board
    :param goal_board: The goal board.
    :type goal_board: Board
    :return: The solution state.
    :rtype: State
    """
    frontier = [(initial_state.hfn(initial_state, goal_board), 0, initial_state)]
    heapq.heapify(frontier)
    visited = set()
    tiebreaker = 1
    while frontier:
        state = heapq.heappop(frontier)[2]
        if state not in visited:
            visited.add(state)
            if state.board == goal_board:
                return state
            next = generate_next_steps(state)
            for successor in next:
                if successor not in visited:
                    # breakpoint()
                    heapq.heappush(frontier, (successor.f, tiebreaker, successor))
                    tiebreaker += 1
    return None

def heuristic(state, goal_board):
    """
    Calculate the heuristic value of the current state.

    :param state: The current state.
    :type state: State
    :param goal_board: The goal board.
    :type goal_board: Board
    :return: The heuristic value.
    :rtype: int
    """
    distance = 0
    # breakpoint()
    unmatched = goal_board.pieces[:]
    # print(grid_to_string(state.board.grid))
    for piece in state.board.pieces:
        min_distance = float('inf')
        closest_piece = None
        for goal_piece in unmatched:
            if piece.is_2_by_2 and goal_piece.is_2_by_2:
                curr_distance = abs(piece.coord_x - goal_piece.coord_x) + abs(piece.coord_y - goal_piece.coord_y)
            elif piece.is_single and goal_piece.is_single:
                curr_distance = abs(piece.coord_x - goal_piece.coord_x) + abs(piece.coord_y - goal_piece.coord_y)
            elif not piece.is_single and not goal_piece.is_single:
                curr_distance = abs(piece.coord_x - goal_piece.coord_x) + abs(piece.coord_y - goal_piece.coord_y)
            else:
                continue

            if curr_distance < min_distance:
                min_distance = curr_distance
                closest_piece = goal_piece
        if closest_piece is not None:

            unmatched.remove(closest_piece)
            distance += min_distance
    # breakpoint()
    return distance
    


def generate_next_steps(state):
    """
    Generate next possible states from the current state.

    :param state: The current state.
    :type state: State
    :return: A list of next possible states.
    :rtype: List[State]
    """
    blank = state.board.blanks
    next_states = []
    # breakpoint()
    adjacent_piece = set()
    # blank_info = (False, 'none')
    # maj = None

    # if blank[0][0] == blank[1][0] and abs(blank[0][1] - blank[1][1]) == 1:
    #     blank_info = (True, 'h')
    #     if blank[0][1] < blank[1][1]:
    #         maj = blank[0]
    #     else:
    #         maj = blank[1]
    # elif blank[0][1] == blank[1][1] and abs(blank[0][0] - blank[1][0]) == 1:
    #     blank_info = (True, 'v')
    #     if blank[0][0] < blank[1][0]:
    #         maj = blank[0]
    #     else:
    #         maj = blank[1]
    
    
    width, height = state.board.width, state.board.height
    for space in blank:
            # check if the blank is at the edge on the x axis on the left side
        y, x = space
        potential_moves = [
            (y, x - 1, "h2") if x > 0 else None,
            (y, x + 1, "h1") if x < width - 1 else None,
            (y - 1, x, "v2") if y > 0 else None,
            (y + 1, x, "v1") if y < height - 1 else None
        ]

        potential_moves = [move for move in potential_moves if move is not None]

        for move in potential_moves:
            if state.board.grid[move[0]][move[1]] != '.':
                for piece in state.board.pieces:
                    if (move[0], move[1]) in piece.occupied:
                        adjacent_piece.add((piece, move[2]))
                        break
    # breakpoint()
    for piece in adjacent_piece:
        # if (piece[0].is_2_by_2 and blank_info[0] == False):
        #     continue
        # elif (not piece[0].is_single and blank_info[0] == False and piece[1].startswith(piece[0].orientation)):
        #     continue
        a = apply_move(state, piece[0], piece[1])
        if a is not None:
            next_states.append(a)
    # breakpoint()
    return next_states

        

def apply_move(state, piece, direction):
    """
    Apply a move to a piece on the board.

    :param state: The current state.
    :type state: State
    :param piece: The piece to be moved.
    :type piece: Piece
    :param direction: The direction of the move.
    :type direction: str
    """
    original_x = piece.coord_x
    original_y = piece.coord_y
    old_occupied = set()
    

    for i in piece.occupied:
        old_occupied.add(i)

    if direction == "h1":
        piece.set_coords(original_x - 1, original_y)

    elif direction == "h2":
        piece.set_coords(original_x + 1, original_y)

    elif direction == "v1":
        piece.set_coords(original_x, original_y - 1)

    elif direction == "v2":
        piece.set_coords(original_x, original_y + 1)

    
    for i in piece.occupied:
        if state.board.grid[i[0]][i[1]] != '.' and i not in old_occupied:
            piece.set_coords(original_x, original_y)
            return None
    # new_pieces = copy.deepcopy(state.board.pieces)
    new_pieces = []
    new_piece = Piece(piece.is_2_by_2, piece.is_single, piece.coord_x, piece.coord_y, piece.orientation)
    for old_piece in state.board.pieces:
        if old_piece == piece:
            new_pieces.append(new_piece)
        else: 
            new_pieces.append(old_piece)
    
    new_board = Board(state.board.height, new_pieces)
    piece.set_coords(original_x, original_y)


    new_state = State(new_board, state.hfn, state.f, state.depth + 1, state)
    new_state.f = new_state.hfn(new_state, goal_board) + new_state.depth

    return new_state

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzles."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=['astar', 'dfs'],
        help="The searching algorithm."
    )
    args = parser.parse_args()
    # read the board from the file
    board, goal_board = read_from_file(args.inputfile)
    initial_state = State(board, heuristic, 0, 0)

    print("Initial board:")
    print(grid_to_string(board.grid))


    if args.algo == 'dfs':
        a = dfs(initial_state, goal_board)
        if a is None:
            open(args.outputfile, 'w').write("No solution found.")
            print("AAAAAAAAAAAA")
        print_solution(a)

    elif args.algo == 'astar':
        # pdb.set_trace()
        a = astar(initial_state, goal_board)
        if a is None:
            open(args.outputfile, 'w').write("No solution found.")
            print("AAAAAAAAAAAA")
        else:
            print_solution(a)
    
    #An example of how to write solutions to the outputfile. (This is not a correct solution, of course).
    #with open(args.outputfile, 'w') as sys.stdout:
    #    board.display()
    #    print("")
    #    goal_board.display()

