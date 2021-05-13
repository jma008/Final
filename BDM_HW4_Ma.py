from pyspark import SparkContext
import datetime
import csv
import functools
import json
import numpy as np
import sys


def main(sc):
    '''
    Transfer our code from the notebook here, however, remember to replace
    the file paths with the ones provided in the problem description.
    '''
    rddPlaces = sc.textFile('/data/share/bdm/core-places-nyc.csv')
    rddPattern = sc.textFile('/data/share/bdm/weekly-patterns-nyc-2019-2020/*')
    OUTPUT_PREFIX = sys.argv[1]

    #===========Step_C=============
    CAT_CODES = {'445210', '445110', '722410', '452311', '722513', '445120', '446110', '445299', '722515', '311811',
                 '722511', '445230', '446191', '445291', '445220', '452210', '445292'}

    CAT_GROUP = {'452210': 0, '452311': 0, '445120': 1, '722410': 2, '722511': 3, '722513': 4, '446110': 5, '446191': 5,
                 '722515': 6, '311811': 6, '445210': 7, '445299': 7, '445230': 7, '445291': 7, '445220': 7, '445292': 7,
                 '445110': 8}

    # ===========Step_D=============
    # 0: placekey
    # 9: naics_code
    def filterPOIs(_, lines):
        reader = csv.reader(lines)
        for row in reader:
            if row[9] in CAT_CODES:
                placekey = row[0]
                group = int(CAT_GROUP[row[9]])  # get group numbers
                yield (placekey, group)

    # only get the needed information
    rddD = rddPlaces.mapPartitionsWithIndex(filterPOIs) \
        .cache()

    # ===========Step_E=============
    storeGroup = dict(rddD.collect())
    groupCount = rddD \
        .map(lambda x: (x[1], 1)) \
        .reduceByKey(lambda x, y: x + y) \
        .sortBy(lambda x: x[0]) \
        .map(lambda x: x[1]) \
        .collect()

    # ===========Step_F=============
    #  0: placekey
    # 12: date_range_start
    # 14: raw_visit_counts
    # 16: visits_by_day

    #def extractVisits(storeGroup, _, lines):
        #if _ == 0:
            #next(lines)

        #reader = csv.reader(lines)
        #for row in reader:
            #if row[0] in storeGroup.keys():
                #group = storeGroup[row[0]]
                #start_date = datetime.datetime.strptime(row[12][:10], "%Y-%m-%d")
                #dates = [str(start_date + datetime.timedelta(days=day))[:10] for day in range(7)]

                #for i, date in enumerate(dates):
                    #if date[:4] in ['2019', '2020']:
                        #visits = json.loads(row[16])
                        #yield ((group, date), visits[i])

    #rddF = rddPattern \
        #.mapPartitionsWithIndex(functools.partial(extractVisits, storeGroup))

    # ===========Step_G=============
    #  0: placekey
    # 12: date_range_start
    # 14: raw_visit_counts
    # 16: visits_by_day
    def extractVisits(storeGroup, _, lines):
        if _ == 0:
            next(lines)

        reader = csv.reader(lines)
        for row in reader:
            if row[0] in storeGroup.keys():
                group = storeGroup[row[0]]
                start_date = datetime.datetime.strptime(row[12][:10], "%Y-%m-%d")
                dates = [str(start_date + datetime.timedelta(days=day))[:10] for day in range(7)]

                for i, date in enumerate(dates):
                    if date[:4] in ['2019', '2020']:
                        delta = (datetime.datetime.strptime(date, "%Y-%m-%d") - datetime.datetime(2019, 1, 1)).days
                        visits_by_day = json.loads(row[16])
                        yield ((group, delta), visits_by_day[i])

    rddG = rddPattern \
        .mapPartitionsWithIndex(functools.partial(extractVisits, storeGroup))

    # ===========Step_H=============
    # Remember to use groupCount to know how long the visits list should be
    #def computeStats(groupCount, _, records):
        #for record in records:
            #group = record[0][0]

            # check how long the visit list should be
            #visit_list = groupCount[group]

            # create list with 0s for the places where no weekly data
            #zeros = np.zeros(visit_list - len(list(record[1])))
            #compute_list = list(record[1]) + list(zeros)
            #median = np.median(compute_list)
            #stdev = np.std(compute_list)
            #low = max(0, median - stdev)
            #high = max(0, median + stdev)
            #yield (record[0], (median, low, high))

    #rddH = rddG.groupByKey() \
        #.mapPartitionsWithIndex(functools.partial(computeStats, groupCount))

    # ===========Step_I=============
    rddI = rddG.groupByKey() \
        .map(lambda x: (x[0], list(x[1]))) \
        .map(lambda x: (x[0], x[1] + ([0] * (groupCount[x[0][0]] - len(x[1]))))) \
        .map(lambda x: (x[0], np.median(x[1]), np.std(x[1]))) \
        .map(lambda x: (x[0][0], str(datetime.datetime(2019, 1, 1) + datetime.timedelta(days=x[0][1]))[:10], int(x[1]),
                        max(0, int(x[1] - x[2])), max(0, int(x[1] + x[2])))) \
        .map(lambda x: (x[0], ','.join((str(x[1][:4]), '2020-' + str(x[1][5:]), str(x[2]), str(x[3]), str(x[4])))))

    # ===========Step_J=============
    rddJ = rddI.sortBy(lambda x: x[1][:15])
    header = sc.parallelize([(-1, 'year,date,median,low,high')]).coalesce(1)
    rddJ = (header + rddJ).coalesce(10).cache()

    # ===========Step_K=============
    #OUTPUT_PREFIX = '/content/output'
    #filename = 'big_box_grocers'
    #rddJ.filter(lambda x: x[0] == 0 or x[0] == -1).values() \
        #.saveAsTextFile(f'{OUTPUT_PREFIX}/{filename}')

    # ===========Step_L=============
    #OUTPUT_PREFIX = '/content/output'

    filenames = ['big_box_grocers',
                 'convenience_stores',
                 'drinking_places',
                 'full_service_restaurants',
                 'limited_service_restaurants',
                 'pharmacies_and_drug_stores',
                 'snack_and_bakeries',
                 'specialty_food_stores',
                 'supermarkets_except_convenience_stores']

    for i, name in enumerate(filenames):
        rddJ.filter(lambda x: x[0] == i or x[0] == -(i + 1)).values() \
            .saveAsTextFile(f'{OUTPUT_PREFIX}/{name}')




if __name__ == '__main__':
    sc = SparkContext()
    main(sc)
