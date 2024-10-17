## Create a list of datasets to choose from when loading data
## For now, this is all EVT / Basler data. Could consider adding standard dasher data
## in the future

datasets = [("20240821_095224", "roli-7", i) for i in range(0, 18)] + \
    [("20240819_095241", "roli-7", i) for i in range(0, 18)] + \
    [("20240816_095721", "roli-7", i) for i in range(0, 18)] + \
    [("20240810_095807", "roli-7", i) for i in range(0, 18)] + \
    [("20240802_100046", "roli-7", i) for i in range(0, 18)]