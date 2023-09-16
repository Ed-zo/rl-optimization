def write_to_file(log, filename):
    f = open(filename, "w")
    f.write(log)
    f.close()