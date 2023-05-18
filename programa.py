import weka.core.jvm as jwm
from weka.core.converters import Loader
from weka.classifiers import Classifier, Evaluation
from weka.filters import Filter
from weka.core.classes import Random
import matplotlib.pyplot as plt



jwm.start()

filename = input("Sartu .arff fitxategiaren izena: ")

loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(dfile = filename)
data.class_is_last()

list_zeintzuk = []

zenbat = data.num_attributes
list_names = data.attribute_names()
lista = []
for i in range(0,len(list_names)):
    lista.append([i+1,list_names[i]])
lista.pop()

jarraitu = True

asm_tot = 0
while jarraitu:
    jarraitu = False
    prob = 0.0
    for i in range(1,len(lista)+1):
        listb = [j for j in lista]
        aldb = listb.pop(i-1)
        zer = str(listb[0][0])
        for i in range(1,len(listb)):
            zer = zer + ',' + str(listb[i][0])

        remove = Filter(classname="weka.filters.unsupervised.attribute.Remove", options=["-R", zer])

        remove.inputformat(data)
        filtered = remove.filter(data)

        cls = Classifier(classname="weka.classifiers.trees.RandomForest", options=["-P", "100", "-I", "100", "-num-slots", "1", "-K", "0", "-M", "1.0", "-V", "0.001", "-S", "1"])
        cls.build_classifier(filtered)

        evl = Evaluation(filtered)
        evl.crossvalidate_model(cls, filtered, 10, Random(1))

        asma_tasa = evl.percent_correct

        if asma_tasa > prob:
            prob = asma_tasa
            ald = aldb

    if prob > asm_tot:
        asm_tot = prob
        jarraitu = True
        list_zeintzuk.append(ald)
        list_zeintzuk[-1].append(prob)
        print("Aldagai altuena: ", list_zeintzuk[-1][1])
        print("Bere asmatze tasa: ", prob)
        ind = lista.index(list_zeintzuk[-1])
        #print(ind)
        lista.pop(ind)
    else:
        list_zeintzuk.append(ald)
        list_zeintzuk[-1].append(asma_tasa)
        print("Aldagai altuena: ", list_zeintzuk[-1][1])
        print("Bere asmatze tasa: ", prob)

    
list_ald = [list_zeintzuk[i][1] for i in range(0,len(list_zeintzuk))]
list_prob = [list_zeintzuk[i][2] for i in range(0,len(list_zeintzuk))]
print("Aldagaiak: ", list_ald)
print("Asmatze tasa: ", asm_tot)

plt.plot(list_ald, list_prob, marker='o', linestyle='--')
plt.ylim(bottom = 0, top = 100)
plt.show()

jwm.stop()
