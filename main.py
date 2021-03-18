import firepoints_timemerge
import main_fire_detection
import time

if __name__ == '__main__':
    while True:
        main_fire_detection.main()
        firepoints_timemerge.main()
        time.sleep(10)
