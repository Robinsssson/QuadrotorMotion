import json
import types

class SingleBuilding:
    def __init__(self, build_dict: dict) -> types.NoneType:
        self.json_str = ["label", "pos", "size", "angle", "height","attribute"]
        self.dic = build_dict
        self.label = build_dict["label"]
        self.pos = build_dict["pos"]
        self.size = build_dict["size"]
        self.angle = build_dict["angle"]
        self.height = build_dict["height"]
        self.attribute = build_dict["attribute"]
        
    def __str__(self) -> str:
        return f"label: {self.label}, pos: {self.pos}, size: {self.size}, angle: {self.angle}, height: {self.height}, attribute: {self.attribute}"
        
        
class BuildingParser:
    def __init__(self, path: str) -> None:
        self.f = open(path, "r")
        self.json_object = json.loads(self.f.read())
        self.builds = []
        for build in self.building_get():
            self.builds.append(SingleBuilding(build))
        
    def building_get(self) -> dict:
        return self.json_object["building"]
        
    
