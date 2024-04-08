import json
import types

class SingleBuilding:
    def __init__(self, build_dict: dict) -> types.NoneType:
        self.json_str = ["label", "pos", "size", "angle" "attribute"]
        self.dic = build_dict
        self.pos = build_dict["pos"]
        self.size = build_dict["size"]
        self.angle = build_dict["angle"]
        self.attribute = build_dict["attribute"]
        self.label = build_dict["label"]
        
        
class BuildingParser:
    def __init__(self, path: str) -> None:
        self.f = open(path, "r")
        self.json_object = json.loads(self.f.read())
        
    def building_get(self) -> dict:
        return self.json_object["building"]
        
parser = BuildingParser("/home/robinson/projects/QuadrotorMotion/File/building.json")
builds = []
for build in parser.building_get():
    builds.append(SingleBuilding(build))

print(builds[0].label)
    
