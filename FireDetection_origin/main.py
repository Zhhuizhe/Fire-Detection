import sys
import firepoints_timemerge_main
import main_fire_detection
import time

if './' not in sys.path:
    sys.path.insert(0, './')

if __name__ == '__main__':
    while True:
        main_fire_detection.main()
        firepoints_timemerge_main.main()
        time.sleep(10)
