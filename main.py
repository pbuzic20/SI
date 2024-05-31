from pathlib import Path
import pandas as panda
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as graf
import numpy as nmpy
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
import time
import os


def pripremiModelZaKoristenje(populatedDataframe: DataFrame):
    
    # Uklanjanje nepotrebnih svojstava skupa podataka
    populatedDataframe = populatedDataframe.drop(['sessionIndex', 'rep'], axis=1)

    # Razdvajanje skupa podataka na X i Y
    X = populatedDataframe.drop(columns=['subject'])
    Y = populatedDataframe['subject']

    # Podjela podataka na trening i test skupove
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Normalizacija skupova podataka
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def trenirajModel(trainDF_Y, trainDF_X, model: RandomForestClassifier):
    pocetnoVrijeme = time.time()
    model.fit(trainDF_X, trainDF_Y.values.ravel())
    vrijemeZavrsetka = time.time()
    vrijemeTreniranja = vrijemeZavrsetka - pocetnoVrijeme
    return model, vrijemeTreniranja

def testirajModel(model, y_test, testDF_X):
    pocetnoVrijeme = time.time()
    predikcija = model.predict(testDF_X)
    vrijemeZavrsetka = time.time()
    vrijemeTestiranja = vrijemeZavrsetka - pocetnoVrijeme
    matricaKonfuzije = confusion_matrix(y_test, predikcija)
    return matricaKonfuzije, predikcija, vrijemeTestiranja

def izracunajStatistikuPodataka(matricaKonfuzije, y_test, predikcija):
    tPozitiv, fPozitiv, fNegativ, tNegativ = izracunajMatricuKonfuzije(matricaKonfuzije)
    tocnost = izracunajTocnost(y_test, predikcija)
    opoziv = izracunajOpoziv(tPozitiv, fNegativ)
    stopaLaznihPozitiva = izracunajStopuLaznihPozitiva(fPozitiv, tNegativ)
    stopaPravihNegativa = izracunajStopuPravihNegativa(tNegativ, fPozitiv)
    preciznost = izracunajPreciznost(y_test, predikcija)
    fMjera = izracunajFMjeru(tocnost, opoziv)

    statistickiPodaci = {
        'truePositive': tPozitiv,
        'falsePositive': fPozitiv,
        'falseNegative': fNegativ,
        'trueNegative': tNegativ,
        'precision': tocnost,
        'recall': opoziv,
        'falsePositiveRate': stopaLaznihPozitiva,
        'trueNegativeRate': stopaPravihNegativa,
        'accuracy': preciznost,
        'fMeassure': fMjera
    }
    return statistickiPodaci

def izracunajMatricuKonfuzije(matricaKonfuzije):
    tPozitiv = nmpy.diag(matricaKonfuzije)
    fPozitiv = matricaKonfuzije.sum(axis=0) - nmpy.diag(matricaKonfuzije)
    fNegativ = matricaKonfuzije.sum(axis=1) - nmpy.diag(matricaKonfuzije)
    tNegativ = matricaKonfuzije.sum() - (fPozitiv + fNegativ + tPozitiv)
    return tPozitiv, fPozitiv, fNegativ, tNegativ

def izracunajTocnost(y_test, predikcija):
    return precision_score(y_test, predikcija, average='weighted') * 100

def izracunajOpoziv(tPozitiv, fNegativ):
    return tPozitiv / (tPozitiv + fNegativ) * 100

def izracunajStopuLaznihPozitiva(fPozitiv, tNegativ):
    return fPozitiv / (fPozitiv + tNegativ) * 100

def izracunajStopuPravihNegativa(tNegativ, fPozitiv):
    return tNegativ / (tNegativ + fPozitiv) * 100

def izracunajPreciznost(y_test, predikcija):
    return accuracy_score(y_test, predikcija) * 100

def izracunajFMjeru(tocnost, opoziv):
    return 2 * ((tocnost * opoziv) / (tocnost + opoziv))

def iscrtajGraf(dataframe, title, filename):
    ax = dataframe.plot(kind='bar')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title(title)
    ax.legend().remove()
    ax.figure.savefig(filename, bbox_inches='tight')

def grafOpoziv(statistickiPodaci, listaSubjekata):
    opozivDF = panda.DataFrame(list(statistickiPodaci.get("recall")), index=listaSubjekata)
    iscrtajGraf(opozivDF, 'Opoziv po kategorijama', 'grafOpoziva.png')

def grafFPR(statistickiPodaci, listaSubjekata):
    falsePositiveValuesDF = panda.DataFrame(statistickiPodaci.get("falsePositiveRate"), index=listaSubjekata)
    iscrtajGraf(falsePositiveValuesDF, 'Stopa pogrešnih klasifikacija po subjektima', 'stopaPogresnihKlasifikacija.png')

def grafTNR(statistickiPodaci, listaSubjekata):
    trueNegativeValuesDF = panda.DataFrame(statistickiPodaci.get("trueNegativeRate"), index=listaSubjekata)
    iscrtajGraf(trueNegativeValuesDF, 'Stopa točnih klasifikacija normalnih zapisa po subjektima', 'stopaTocnihKlasifikacija.png')

def grafFMjera(statistickiPodaci, listaSubjekata):
    fMeassureValuesDF = panda.DataFrame(statistickiPodaci.get("fMeassure"), index=listaSubjekata)
    iscrtajGraf(fMeassureValuesDF, 'F-mjera modela po subjektima', 'fMjera.png')

def pripremiOkvirJednostavnihStatistika(vrijemeTreniranja, vrijemeTestiranja, statistickiPodaci):
    okvirJednostavnihStatistika = panda.DataFrame(index=['Vrijeme treniranja', 'Vrijeme testiranja', 'Preciznost', 'Tocnost'])
    okvirJednostavnihStatistika["Jednostavni pokazatelji"] = [vrijemeTreniranja, vrijemeTestiranja, statistickiPodaci.get("accuracy"), statistickiPodaci.get("precision")]
    return okvirJednostavnihStatistika

def pripremiOkvirKlasifikacijskihStatistika(statistickiPodaci, listaSubjekata):
    okvirKlasifikacijskihStatistika = panda.DataFrame(index=listaSubjekata)
    okvirKlasifikacijskihStatistika['TP'] = statistickiPodaci.get("truePositive")
    okvirKlasifikacijskihStatistika['FP'] = statistickiPodaci.get("falsePositive")
    okvirKlasifikacijskihStatistika['FN'] = statistickiPodaci.get("falseNegative")
    okvirKlasifikacijskihStatistika['TN'] = statistickiPodaci.get("trueNegative")
    okvirKlasifikacijskihStatistika['FPR'] = statistickiPodaci.get("falsePositiveRate")
    okvirKlasifikacijskihStatistika['TNR'] = statistickiPodaci.get("trueNegativeRate")
    okvirKlasifikacijskihStatistika['Opoziv'] = statistickiPodaci.get("recall")
    okvirKlasifikacijskihStatistika['F-mjera'] = statistickiPodaci.get("fMeassure")
    return okvirKlasifikacijskihStatistika

def kreirajExcelDatoteku(statistickiPodaci, vrijemeTreniranja, vrijemeTestiranja, listaSubjekata):
    okvirJednostavnihStatistika = pripremiOkvirJednostavnihStatistika(vrijemeTreniranja, vrijemeTestiranja, statistickiPodaci)
    okvirKlasifikacijskihStatistika = pripremiOkvirKlasifikacijskihStatistika(statistickiPodaci, listaSubjekata)

    nazivDatoteke = 'Evaluacija modela.xlsx'
    pisac = panda.ExcelWriter(nazivDatoteke, engine='xlsxwriter', engine_kwargs={'options':{'strings_to_formulas': False}})
    radnaKnjiga = pisac.book
    radniList = radnaKnjiga.add_worksheet("Statistika")
    radniList.set_column(0, 0, 20)
    radniList.set_column(1, 1, 18)
    radniList.set_column(3, 7, 10)
    radniList.set_column(8, 11, 12)
    pisac.sheets["Statistika"] = radniList
    okvirJednostavnihStatistika.to_excel(pisac, sheet_name="Statistika", startrow=0, startcol=0)
    okvirKlasifikacijskihStatistika.to_excel(pisac, sheet_name="Statistika", startrow=0, startcol=3)
    pisac.close()

# 2. Dohvacanje podataka iz datoteke DSL - Strong Password.csv

trenutnaMapa = os.path.dirname(__file__)
putanjaSkupaPodataka = os.path.join(trenutnaMapa, "Dataset.csv")

# Spremanje skupa podataka u "Dataframe"
okvirSetaPodataka = panda.read_csv(putanjaSkupaPodataka)

# Spremanje liste subjekata za kasnije koristenje s 
# statistickim podacima
listaSubjekata = okvirSetaPodataka['subject'].drop_duplicates().tolist()

# 3. Priprema skupa podataka #

# Koristenje napravljene metode za pripremu skupa podataka 
# za unos u model strojng ucenja
X_train, X_test, y_train, y_test = pripremiModelZaKoristenje(okvirSetaPodataka)

# 4. Unos skupa podataka u model strojnog ucenja #

# Inicijalizacija modela strojnog ucenja
modelStrojnogUcenja = RandomForestClassifier(n_estimators = 30)

# Koristenje napravljene metode za rad nad modelom strojnog ucenja
modelStrojnogUcenja, vrijemeTreniranja = trenirajModel(y_train, X_train, modelStrojnogUcenja)

matricaKonfuzije, predikcija, vrijemeTestiranja = testirajModel(modelStrojnogUcenja, y_test, X_test)

# 5. Izracun statistickih pokazatelja modela strojnog ucenja
#    i spremanje statistickih pokazatelja u Excel datoteku#

# Metoda sluzi za izracun statisticki pokazatelja, odnosno
# sluzi za izracun performansi modela strojnog ucenja
statistickiPodaci = izracunajStatistikuPodataka(matricaKonfuzije, y_test, predikcija)

# Metoda kojom se iscrtavaju grafovi odabranih pokazatelja kako bi 
# smo kasnije detaljno analizirali rad modela
grafOpoziv(statistickiPodaci, listaSubjekata)
grafFPR(statistickiPodaci, listaSubjekata)
grafTNR(statistickiPodaci, listaSubjekata)
grafFMjera(statistickiPodaci, listaSubjekata)

# Metoda koja sluzi za perzistenciju podataka. Izracunate statisticke
# pokazatelje spremamo u Excel datoteku
kreirajExcelDatoteku(statistickiPodaci, vrijemeTreniranja, vrijemeTestiranja, listaSubjekata)