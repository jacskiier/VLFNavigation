# -*- coding: utf-8 -*-
"""
Created on Thu Dec 03 08:13:37 2015

@author: jacsk
"""
import time


class tictoc:
    def __init__(self):
        self.ticMessages = []
        self.ticStartTimes = []

    def tic(self, message=""):
        targ = time.time()
        print("{message}".format(message=message))
        self.ticMessages.append(message)
        self.ticStartTimes.append(targ)
        return

    def toc(self):
        targ = time.time()
        start = self.ticStartTimes.pop()
        message = self.ticMessages.pop()
        elapsedTime = targ - start
        if elapsedTime < 1:
            print("{message} completed in: {elapsedTime:5.5} milliseconds".format(message=message, elapsedTime=elapsedTime * 1000.0))
        elif elapsedTime < 60:
            print("{message} completed in: {elapsedTime:5.4} seconds".format(message=message, elapsedTime=elapsedTime))
        elif elapsedTime < 3600:
            print("{message} completed in: {elapsedTime:5.4} minutes".format(message=message, elapsedTime=elapsedTime / 60.0))
        else:
            print("{message} completed in: {elapsedTime:5.4} hours".format(message=message, elapsedTime=elapsedTime / 3600.0))
        return elapsedTime


if __name__ == '__main__':
    t = tictoc()
    t.tic("Loop")
    y = 1
    for x in xrange(5):
        t.tic("insideloop")
        for z in xrange(1000000):
            y = (y * x + x) / (z + 1)
        time.sleep(5)
        t.toc()
    t.toc()
