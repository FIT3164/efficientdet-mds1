import json
from pprint import pprint
from statistics import mean

# open annotations json file
with open("prototype/dataset/data/goodbadchili/annotations/instances_train.json") as f:
    file = json.loads("".join(f.readlines()))
    bbox_list = [] # list of all bbox widths and heights
    for ann in file["annotations"]:
        bbox_list.append(ann["bbox"][2:4])

    # print out general statistics about the bboxes
    print('general statistics')
    print(f'min width:  {min(bbox_list, key = lambda x: x[0])[0]}')
    print(f'max width:  {max(bbox_list, key = lambda x: x[0])[0]}')
    print(f'min height: {min(bbox_list, key = lambda x: x[1])[1]}')
    print(f'max height: {max(bbox_list, key = lambda x: x[1])[1]}')
    print(f'avg: ({mean([x[0] for x in bbox_list])}, {mean([x[1] for x in bbox_list])})')
    print(f'length: {len(bbox_list)}')

    print()
    # split the widths and heights into groups of 10 units and count the frequency
    bbox_sizes = [[0 for _ in range(10)] for _ in range(10)] # f(x) = 30x + 30
    for bbox in bbox_list:
        bbox_sizes[(bbox[0] - 30) // 30][(bbox[1] - 30) // 30] += 1
    pprint(bbox_sizes)

    # print()
    freq_threshold = 100
    # filt_bbox_sizes = list(map(lambda y: list(map(lambda x: 1 if x >= freq_threshold else 0, y)), bbox_sizes))
    # pprint(filt_bbox_sizes)

    # print out specific bbox width and height groups with their statistics filtered over a threshold
    print()
    print(f'most common (generalised) anchor box sizes with >= {freq_threshold}  occurences')
    img_size = 512*512
    res = []
    for i in range(len(bbox_sizes)):
        for j in range(10):
            if bbox_sizes[i][j] >= freq_threshold:
                x, y = i*30 + 30, j*30 + 30
                temp = (x, y, round(y/x, 4), x*y, round(x*y/img_size, 4), bbox_sizes[i][j])
                res.append(temp)
                print(f'({temp[0]}, {temp[1]}), \tratio: {temp[2]}, \tsize:  {temp[3]}, \tscale: {temp[4]}, \tfreq: {temp[5]}')

    # print the table based on the order of the ratio, scale, or frequencies
    print('\nsorted by ratio [2]')
    pprint(sorted(res, key = lambda x: x[2]))

    print('\nsorted by scale [4]')
    pprint(sorted(res, key = lambda x: x[4]))

    print('\nsorted by freq [5]')
    pprint(sorted(res, key = lambda x: -x[5]))


# SAMPLE RESULTS FOR THE ORIGINAL COCO-FORMATTED NON-AUGMENTED DATASET

# general statistics
# min width:  37
# max width:  325
# min height: 38
# max height: 315
# avg: (168.59393346379647, 156.43868232224398)
# length: 3066

# [[0, 1, 3, 5, 1, 0, 0, 0, 0, 0],
#  [2, 9, 56, 49, 19, 5, 11, 5, 1, 0],
#  [7, 39, 74, 56, 62, 97, 74, 23, 8, 0],
#  [6, 50, 60, 51, 91, 134, 122, 50, 16, 2],
#  [2, 31, 55, 103, 151, 122, 69, 20, 5, 1],
#  [1, 13, 105, 137, 158, 96, 29, 7, 2, 0],
#  [0, 10, 139, 211, 97, 39, 11, 4, 1, 0],
#  [0, 5, 61, 90, 30, 13, 0, 1, 0, 0],
#  [0, 2, 11, 20, 9, 2, 1, 0, 0, 0],
#  [0, 0, 7, 6, 0, 0, 0, 0, 0, 0]]

# most common (generalised) anchor box sizes with >= 100  occurences
# (120, 180),     ratio: 1.5,     size:  21600,    scale: 0.0824
# (120, 210),     ratio: 1.75,    size:  25200,    scale: 0.0961
# (150, 120),     ratio: 0.8,     size:  18000,    scale: 0.0687
# (150, 150),     ratio: 1.0,     size:  22500,    scale: 0.0858
# (150, 180),     ratio: 1.2,     size:  27000,    scale: 0.103
# (180, 90),      ratio: 0.5,     size:  16200,    scale: 0.0618
# (180, 120),     ratio: 0.6667,  size:  21600,    scale: 0.0824
# (180, 150),     ratio: 0.8333,  size:  27000,    scale: 0.103
# (210, 90),      ratio: 0.4286,  size:  18900,    scale: 0.0721
# (210, 120),     ratio: 0.5714,  size:  25200,    scale: 0.0961

# sorted by ratio [2]
# [(210, 90,  0.4286, 18900, 0.0721),
#  (180, 90,  0.5,    16200, 0.0618),
#  (210, 120, 0.5714, 25200, 0.0961),
#  (180, 120, 0.6667, 21600, 0.0824),
#  (150, 120, 0.8,    18000, 0.0687),
#  (180, 150, 0.8333, 27000, 0.103),
#  (150, 150, 1.0,    22500, 0.0858),
#  (150, 180, 1.2,    27000, 0.103),
#  (120, 180, 1.5,     21600, 0.0824),
#  (120, 210, 1.75,   25200, 0.0961)]

# sorted by scale [4]
# [(180, 90,  0.5,    16200, 0.0618),
#  (150, 120, 0.8,    18000, 0.0687),
#  (210, 90,  0.4286, 18900, 0.0721),
#  (120, 180, 1.5,    21600, 0.0824),
#  (180, 120, 0.6667, 21600, 0.0824),
#  (150, 150, 1.0,    22500, 0.0858),
#  (120, 210, 1.75,   25200, 0.0961),
#  (210, 120, 0.5714, 25200, 0.0961),
#  (150, 180, 1.2,    27000, 0.103),
#  (180, 150, 0.8333, 27000, 0.103)]