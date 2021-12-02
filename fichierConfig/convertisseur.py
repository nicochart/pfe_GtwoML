import json


def loadData(file):
    with open(file) as f:
        data = json.load(f)
    return data

def numberNeuronCumul(data):
    nbNeuronCumul = []
    nbNeuronCumul.append(data[0]["nbNeuron"])
    for i in range(1,len(data)):
        nbNeuronCumul.append(data[i]["nbNeuron"]+nbNeuronCumul[i-1])
    return nbNeuronCumul

def distribNeuronCumul(data):
    repartitionNeuronCumul = []
    for i in range(0,len(data)):
        repartitionNeuronCumul.append([data[i]["typeNeuron"][0]["nbNeuron"]/data[i]["nbNeuron"]])
        for j in range(1,data[i]["nbTypeNeuron"]):
            repartitionNeuronCumul[i].append(data[i]["typeNeuron"][j]["nbNeuron"]/data[i]["nbNeuron"]+repartitionNeuronCumul[i][j-1])
    return repartitionNeuronCumul

def probaConnection(data):
    probaConnect = [[] for i in range(len(data))]
    for i in range(len(data)):
        for j in range(data[i]["nbTypeNeuron"]):
            probaConnect[i].append([])
            for k in range(len(data)):
                probaConnect[i][j].append(data[i]["typeNeuron"][j]["nbConnection"]*data[i]["distribution"][k]/data[i]["nbNeuron"])
    return probaConnect

data = loadData("configTest.json")
print(numberNeuronCumul(data))
print(distribNeuronCumul(data))
print(probaConnection(data))
