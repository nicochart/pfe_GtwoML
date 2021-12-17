import json

def writeData(out):
    #Ecriture
    f = open("param.cfg","w")
    f.write(out)
    f.close()


def loadData(file):
    with open(file) as f:
        data = json.load(f)
    return data

def numberNeuronCumul(data):
    nbNeuronCumul = []
    nbNeuronCumul.append(data[0]["nbNeuron"]/2)
    for i in range(1,len(data)):
        nbNeuronCumul.append(data[i]["nbNeuron"]/2+nbNeuronCumul[i-1])
    for i in range(len(data)):
        nbNeuronCumul.append(data[i]["nbNeuron"]/2+nbNeuronCumul[len(data)+i-1])
    return nbNeuronCumul

def distribNeuronCumul(data):
    repartitionNeuronCumul = []
    for i in range(len(data)*2):
        repartitionNeuronCumul.append([data[i%len(data)]["typeNeuron"][0]["nbNeuron"]/data[i%len(data)]["nbNeuron"]])
        for j in range(1,data[i%len(data)]["nbTypeNeuron"]):
            repartitionNeuronCumul[i].append(data[i%len(data)]["typeNeuron"][j]["nbNeuron"]/data[i%len(data)]["nbNeuron"]+repartitionNeuronCumul[i][j-1])

    return repartitionNeuronCumul

def probaConnection(data):
    probaConnect = [[] for i in range(len(data)*2)]
    for i in range(len(data)):
        probaConnect[i] = [[] for n in range(data[i]["nbTypeNeuron"])]
        probaConnect[i+len(data)] = [[] for n in range(data[i]["nbTypeNeuron"])]
        for j in range(data[i]["nbTypeNeuron"]):
            probaConnect[i][j] = [0 for n in range(len(data)*2)]
            probaConnect[i+len(data)][j] = [0 for n in range(len(data)*2)]
            for k in range(len(data)):
                probaConnect[i][j][k] = (1-data[i]["connectionOpposite"])*data[i]["distribution"][k]*data[i]["typeNeuron"][j]["nbConnection"]/data[i]["nbNeuron"]
                probaConnect[i][j][k+len(data)] = data[i]["connectionOpposite"]*data[i]["distribution"][k]*data[i]["typeNeuron"][j]["nbConnection"]/data[i]["nbNeuron"]
                probaConnect[i+len(data)][j][k] = data[i]["connectionOpposite"]*data[i]["distribution"][k]*data[i]["typeNeuron"][j]["nbConnection"]/data[i]["nbNeuron"]
                probaConnect[i+len(data)][j][k+len(data)] = (1-data[i]["connectionOpposite"])*data[i]["distribution"][k]*data[i]["typeNeuron"][j]["nbConnection"]/data[i]["nbNeuron"]
    return probaConnect

data = loadData("configCortex.json")
out = ""
nnc = numberNeuronCumul(data)
for i in range(len(nnc)):
    out += str(nnc[i])+" "
out+= "-1 "
dnc = distribNeuronCumul(data)
for i in range(len(dnc)):
    out += str(dnc[i]) + " "
out += "-1 "
pc = probaConnection(data)
for i in range(len(pc)):
    out += str(pc[i]) + " "


"""
out = "neuronByPart = "+str(numberNeuronCumul(data))+"\n"
out += "neuronType = "+str(distribNeuronCumul(data))+"\n"
out += "probaConnection = "+str(probaConnection(data))+"\n"
"""
writeData(out)
