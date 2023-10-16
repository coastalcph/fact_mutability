import json
import os

def main():
    d = './data/enrichment'
    files = os.listdir(d)
    files = [f for f in files if 'json' in f]
    for f in files:
        print(f)
        try:
            data = json.load(open(os.path.join(d, f)))
            for d in data:
                if len(d['objects']) > 1:
                    print(d)
        except:
            pass

if __name__ == '__main__':
    main()