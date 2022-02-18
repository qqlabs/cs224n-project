import sys
import tarfile

def main():
    fname = sys.argv[1]
    tar = tarfile.open(fname)
    tar.extractall()
    tar.close()
    
if __name__ == '__main__':
    main()