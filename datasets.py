
EVT_datasets = [("20240208_160037", "roli-8", i) for i in range(0, 20)] + \
                [("20240209_162809", "roli-8", i) for i in range(0, 20)] + \
                [("20240212_115931", "roli-8", i) for i in range(0, 20)]

BASLER_datasets = [("20240820_150832", "roli-7", i) for i in range(0, 10)] + \
                   [("20240821_160240", "roli-7", i) for i in range(0, 10)] + \
                   [("20240903_134102", "roli-7", i) for i in range(0, 10)] + \
                   [("20240904_142340", "roli-7", i) for i in range(0, 10)]

OMR_datasets = [("20240821_095224", "roli-7", i) for i in range(0, 18)] + \
    [("20240819_095241", "roli-7", i) for i in range(0, 18)] + \
    [("20240816_095721", "roli-7", i) for i in range(0, 18)] + \
    [("20240810_095807", "roli-7", i) for i in range(0, 18)] + \
    [("20240802_100046", "roli-7", i) for i in range(0, 18)]

datasets = EVT_datasets + OMR_datasets + BASLER_datasets