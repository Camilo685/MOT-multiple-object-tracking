import json
import random

comp = []
data = {}
data["frames"] = []
ano1 = []
ano1.append({'dco':True, 'height':random.randint(0,1000), 'width':random.randint(0,1000), 'id':random.randint(0,1000), 'y': random.randint(0,1000), 'x': random.randint(0,1000)})
ano2 = []
ano2.append({'dco':True, 'height':random.randint(0,1000), 'width':random.randint(0,1000), 'id':random.randint(0,1000), 'y': random.randint(0,1000), 'x': random.randint(0,1000)})
ano2.append({'dco':True, 'height':random.randint(0,1000), 'width':random.randint(0,1000), 'id':random.randint(0,1000), 'y': random.randint(0,1000), 'x': random.randint(0,1000)})
data["frames"].append({'timestamp':random.randint(0,1000), 'num': random.randint(0,1000), 'class': 'frame', 'annotations':ano1})
data["frames"].append({'timestamp':random.randint(0,1000), 'num': random.randint(0,1000), 'class': 'frame', 'annotations':ano2})
data["class"] = 'video'
data["filename"] = '/home/camilo685/Desktop/Tesis/Code'
comp.append(data)

y = json.dumps(comp)
print(y)
