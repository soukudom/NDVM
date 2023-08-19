from Data import Data

data = Data(selection=["australian", "spambase"])
datasets = data.load()
print(len(datasets))

data = Data(selection=("all", ["balanced", "binary"]))
datasets = data.load()
print(len(datasets))

data = Data(selection=("any", ["balanced", "binary"]))
datasets = data.load()
print(len(datasets))
