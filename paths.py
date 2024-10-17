import socket
import glob

def locate_all(timestamp, server, filename, raw=False):
    root_dir = "/data/"
    if server != socket.gethostname():
        sidx = server.split("-")[-1]
        root_dir = f"/nfs/data{sidx}/"
    else:
        root_dir = "/data/"
    if raw:
        datadir = "data_raw"
    else:
        datadir = "data"
    
    return glob.glob(f"{root_dir}*/{datadir}/{timestamp}/{filename}")    


def locate(timestamp, server, filename, raw=False):
    files = locate_all(timestamp, server, filename, raw=raw)
    if len(files) > 1:
        raise ValueError("More than one match for $filename found on server $server: \n $files")
    
    return files[0]