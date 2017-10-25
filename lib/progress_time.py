import time
class ProgressTime():
    def getStart(self,str=""):
        print(str + " time set")
        return time.time()

    def progresstime(self,start,str=""):
        elapsed_time = time.time() - start
        print (str + " time:{0}".format(elapsed_time) + "[sec]")
