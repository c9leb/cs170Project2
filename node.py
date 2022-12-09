class Node:
    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.depth = 0
        self.heur = 0
        self.childup = 0
        self.childdown = 0
        self.childleft = 0
        self.childright = 0
    
           
        