# -*- coding: utf-8 -*-
"""RESOLUTION OF QUERIES WITH PYSPARK

Stage 2 of the final project of the Big Data course of
the National University of La Plata. The stage 2 consists
of resolving two queries to the star database using PySpark.

Author: Nahuel Herpo.

Original file is located at
    https://colab.research.google.com/drive/1STlYh5zq2qH42TKBfsk3gDPaxYPz_N04


"""


"""# Imports"""

ROOT_PATH = '/opt/spark-data/'

import math                           # math functions
from pyspark import SparkContext      # spark context
from pyspark.sql import SparkSession  # spark session


"""# Spark Setup"""

# Create Spark Context
sc = SparkContext('local', 'test')

# Create Spark Session
sparkSession = SparkSession.builder.appName('test').getOrCreate()


"""# Query 1

Attached functions
"""

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


"""Query resolution with Spark"""

estrellas = sc.textFile(ROOT_PATH + 'Estrellas/') \
  .map(lambda t: t.split('\t')) \
  .map(lambda t: (int(t[0]), typeNumber(t[2]))).filter(lambda t: t[1] == 0) \
  .partitionBy(7, lambda x: x % 7)

observacionesConTipo = sc.textFile(ROOT_PATH + 'Observaciones/') \
  .map(lambda t: t.split('\t')) \
  .map(lambda t: (int(t[0]), (calcDistInPC(float(t[3]), float(t[4]), calcAbsMag(float(t[5])))))) \
  .partitionBy(7, lambda x: x % 7) \
  .join(estrellas).mapValues(lambda t: t[0]) \
  .aggregateByKey(
      (0, 0),
      (lambda result, original: (result[0] + original, result[1] + 1)),
      (lambda r1, r2: (r1[0] + r2[0], r1[1] + r2[1]))).mapValues(lambda t: t[0] / t[1])

observacionesConTipo.saveAsTextFile(ROOT_PATH + 'output/query1')


"""# Query 2

Attached functions
"""

def activeBit(array, position):
  """
  This function sets an array position to True, depending on the
  observed type. For example if the array is [F, F, F, F] and
  type 0 (zero) was observed then [T, F, F, F] remains.

  Args:
    array (list): Array of observations (elements are booleans).
    position (int): Array position to set to True.

  Returns:
    list: Array modified.

  Raises:
    IndexError: If the index is outside the valid range for the array.
    TypeError: If the 'array' argument is not a list.
  """

  if not isinstance(array, list):
    raise TypeError("The argument 'array' must be a list.")

  if (position < 0) or (position >= len(array)):
    raise IndexError("Index is outside the valid range for the array.")

  array[position] = True

  return array


def mergeBits(array1, array2):
  """
  This merge function does an OR between the two arrays that
  arrive by parameter, in order to total the observed types.

  Args:
    array1 (list): Array of observations (elements are booleans).
    array2 (list): Array of observations (elements are booleans).

  Returns:
    list: Array merged from array1 and array2.
  """

  for i in range(4):
    array1[1][i] = array1[1][i] or array2[1][i]
  return array1


def countBits(array):
  """
  This function, based on the generated array, counts how many
  types the observer has observed.

  Args:
    array (list): Array of observations (e.g. [True, False, True, False]).

  Returns:
    int: number of types observed (e.g. [True, False, True, False] -> 2).
  """

  count = 0
  for i in array:
    if i:
      count += 1
  return count


"""Query resolution with Spark"""

estrellas = sc.textFile(ROOT_PATH + 'Estrellas/') \
  .map(lambda t: t.split('\t')) \
  .map(lambda t: (int(t[0]), typeNumber(t[2]))) \
  .partitionBy(7, lambda x: x % 7)

observacionesConTipo = sc.textFile(ROOT_PATH + 'Observaciones/') \
  .map(lambda t: t.split('\t')) \
  .map(lambda t: (int(t[0]), (t[1], int(t[2])))) \
  .partitionBy(7, lambda x: x % 7) \
  .join(estrellas)

max_observers = observacionesConTipo \
  .mapValues(lambda t: (t[0][0], t[0][1], [False, False, False, False], t[1])) \
  .map(lambda t: (t[1][1], (t[1][0], activeBit(t[1][2], t[1][3])))) \
  .reduceByKey(mergeBits) \
  .map(lambda t: (t[1][0], t[0], countBits(t[1][1]))) \
  .persist()

max = max_observers.aggregate(
      (-1),
      (lambda result, original: result if result > original[2] else original[2]),
      (lambda r1, r2: r1 if r1 > r2 else r2))

max_observers = max_observers.filter(lambda t: t[2] == max).map(lambda t: (t[0], t[1]))

max_observers.saveAsTextFile(ROOT_PATH + 'output/query2')

