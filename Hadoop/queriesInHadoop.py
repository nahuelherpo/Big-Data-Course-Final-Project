# -*- coding: utf-8 -*-
"""

@author: Nahuel Herpo

@release: Sep 21, 2021


----


RESOLUTION OF QUERIES WITH HADOOP

Stage 1 of the final project of the Big Data course of the
National  University  of La Plata. The stage 1 consists of
resolving  two  queries  to  the  star  database using the
map reduce paradigm.

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TSGTRjetlhg9DHBIlS33GEYg1y3lX_RN

"""

"""# Imports"""

import string
import math
import sys
from Emulador_MapReduce import Job
sys.setrecursionlimit(10000)


"""**Attached functions**"""

def typeNumber(star):
  """
  This function returns the number of the star type. The bigger
  the star, the higher the number.

  Args:
    star (str): A star category (Enana, Gigante, so on...).

  Returns:
    int: Star category (-1 if it is an unrecognized type).
  """

  kind = star.split(' ')[0].upper()

  if kind == 'ENANAS':
    return 0
  elif kind == 'GIGANTES':
    return 1
  elif kind == 'SUPERGIGANTES':
    return 2
  elif kind == 'HIPERGIGANTES':
    return 3
  else:
    return -1   #unrecognized type
  
def calcAbsMag(period_in_days):
  """
  Calculate the absolute magnitude of the star using the period of days. The
  period of days is the number of days it takes for the star to restore its
  brightness. To calculate the real brightness of the object, the absolute
  magnitude must be calculated with the following formula:
  M = –1.43 – 2.81 x log(P).

  Args:
    period_in_days (float): The days period of the star.

  Returns:
    float: Absolute magnitude.
  """

  return ((-1.43) + (-2.81) * math.log(period_in_days))


def calcDistInPC(min_mag, max_mag, abs_mag):
  """
  Calculates the distance in parsec (astronomical measurement), for this it
  receives the minimum visual magnitude, the maximum visual magnitude and
  the absolute magnitude (real brightness of the object). The formula for
  the calculation is: m - M = -5 + 5log(D), where m is the mean visual
  magnitude and D is the distance in persecs.

  Args:
    min_mag (float): Minimum visual magnitude.
    max_mag (float): Maximum visual magnitude.
    abs_mag (float): Absolute magnitude.

  Returns:
    float: Distance in parsec (from Earth).
  """

  # Calculate the mean
  magnitud_mean = (min_mag + max_mag) / 2 #calculo la media

  # Formula with D cleared (the base of the logarithm is 10.)
  return (10 ** ((magnitud_mean - abs_mag + 5) / 5))

def main():

  """**Job 1 Filter**"""

  #  JOB 1 -->> Filtra por el tipo de estrella.

  def map1Filter(k1, v1, context):    # k1 = id_estrella  v1 = edad, tipo
    tipo = typeNumber(v1.split('\t')[1])
    if tipo == context['tipoDeEstrellas']: # Si la observacion es de una estrella de tipo ENANA entonces escribo la tupla
      context.write(int(k1), tipo)

  def reduce1(k2, v2, context):   # k2 = id_estrella  v2 = tipo
      context.write(k2, v2.next())

  # Outout:   id_estrella, tipo


  """**Job 2 Join**"""

  #  JOB 2 -->> Realiza un join y calcula el promedio por estrella.

  def map2_Obs(k1, v1, context):  # k1 = id_estrella  v1 = nombre, dni, mag_min, mag_max, periodo, fechaObs.
    values = v1.split('\t')
    distance = calcDistInPC(float(values[2]), float(values[3]), calcAbsMag(float(values[4])))
    context.write((int(k1), 'OBS'), ('OBS', distance, 1))

  def map2_Est(k1, v1, context):  # k1 = id_estrella  v1 = tipo.
    context.write((int(k1), 'EST'), ('TIP', v1))

  def combiner2(k2, v2, context):  # k2 = (id_estrella, 'EST' | 'OBS')  v2 = (('EST', tipo) | ('OBS', distancia, 1))
    count = 0
    distance_sum = 0.0
    for v in v2:
      if v[0] == 'OBS':
        count += v[2]
        distance_sum += v[1]
      elif v[0] == 'TIP':
        context.write(k2, v2)
    if count != 0:  # Si por lo menos hay una observacion
      context.write(k2, ('OBS', distance_sum, count))

  def reduce2(k2, v2, context):  # k2 = (id_estrella, 'EST' | 'OBS')  v2 = (('EST', tipo) | ('OBS', distancia, 1))
    count = 0
    distance_sum = 0.0
    first = v2.next()
    if first[0] == 'TIP':
      for v in v2:
        count += v[2]
        distance_sum += v[1]
      if count != 0: # Si por lo menos hay una observacion
        context.write(k2[0], distance_sum / count)

  def shuffle2(key1, key2):
    if key1[0] == key2[0]:
      return 0
    elif key1[0] < key2[0]:
      return -1
    else:
      return 1

  def sort2(key1, key2):
    if key1[1] == key2[1]:
      return 0
    elif key1[1] == 'EST':
      return -1
    else:
      return 1

  # Output:   id_estrella, prom_distancia

  """**Consulta 1:**"""

  #JOB_1 - Filter
  inputDirectory1 = root_path + '/TP1_TP2/Dataset_Estrellas/'
  outputDirectory1 = root_path + '/TP1_TP2/Output/Job1_Filter/'
  #Job1 On
  job1 = Job(inputDirectory1, outputDirectory1, map1Filter, reduce1)
  dictionary = { 'tipoDeEstrellas': 0 } # 0 = Tipo de estrella ENANA.
  job1.setParams(dictionary)
  print(job1.waitForCompletion())

  #JOB_2 - Join
  inputDirectory2 = outputDirectory1
  outputDirectory2 = root_path + '/TP1_TP2/Output/Job2_Join/'
  #Job2 On
  job2 = Job(inputDirectory2, outputDirectory2, map2_Est, reduce2)
  job2.addInputPath(root_path + '/TP1_TP2/Dataset_Observaciones/', map2_Obs)
  job2.setShuffleCmp(shuffle2)
  job2.setSortCmp(sort2)
  print(job2.waitForCompletion())

if __name__ == "__main__":
  main()