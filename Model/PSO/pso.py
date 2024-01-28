import numpy as np
import json
import sys

class atom:
    def __init__(self, pos, vec, fitness_value) -> None:
        self.pos = pos
        self.v = vec
        self.fitness_value = fitness_value
        
class pso:
    def __init__(self, number: int, D: int) -> None:
        self.number = number
        self.atoms = []
        self.D = D
        self.out_json: str
        
    def outputs(self):
        self.json_file = open("File/test.json", "w+", self.out_json, encoding='utf-8')
        self.json_file.close()
        