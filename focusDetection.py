import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import glob
import re
from numpy import unravel_index
from scipy.optimize import curve_fit
from scipy.misc import factorial        #for poisson distr

class AutoFocus:

    def __init__(self):
        self.parentDir = None
        self.filePath = None
        self.saveToFile = False
        self.pearson = False
        self.gbs = np.empty(0)   # used for x axis
        self.f11 = np.empty(0)
        self.f12 = np.empty(0)

        self.gaussianMean = np.empty(0)
        self.gaussianStd = np.empty(0)

        self.pearson = np.empty(0)

        self.GFP_f11Max = np.empty(0)
        self.GFP_f12Max = np.empty(0)
        self.RFP_f11Max = np.empty(0)
        self.RFP_f12Max = np.empty(0)

        self.xVal = np.empty(0)

        self.refFrame = np.empty(0)

        self.stackCountArray = np.empty(0)
        self.stackCount = 0

        self.stackFileNames = np.empty(0)

        np.seterr(all='raise')

    def getF11Max(self):
        return self.gbs[np.argmax(self.f11)]

    def getF12Max(self):
        return self.gbs[np.argmax(self.f12)]

    def getPearsonMax(self):
        return self.gbs[np.argmax(self.pearson)]

    def clearArrays(self):
        self.gbs = np.empty(0)
        self.f11 = np.empty(0)
        self.f12 = np.empty(0)

        self.pearson = np.empty(0)

    def setRefFrame(self, imagePath):
        aux = sitk.ReadImage(imagePath)
        self.refFrame = sitk.GetArrayFromImage(aux).reshape(-1)
        self.pearson = True

    def calcPearson(self, imageArray):
        pearsonR = np.corrcoef(self.refFrame, imageArray.reshape(-1))
        return pearsonR[0][1]

    def setSaveToFile(self, savePix):
        self.saveToFile = savePix

    def setFilePath(self, filePath):
        self.filePath = filePath

    def setParentDir(self, parentDir):
        self.parentDir = parentDir

    def analyseFolder(self):
        picfiles = glob.glob(self.filePath + '/*.tiff')

        if len(picfiles) == 0:
            picfiles = glob.glob(self.filePath + '/*.tif')

        sreader = sitk.ImageSeriesReader()
        sreader.SetFileNames(picfiles)
        images = sreader.Execute()

        for index, infile in enumerate(picfiles):
            print("processing file:", infile)

            try:
                #m = re.search(r"pic\d+", infile)
                m = re.search(r"M00L515.cam0.focus 0.000\d+", infile)
                #m = re.search(r"691_good-", infile)
                #aux = m.group(0).replace("pic", '')
                aux = m.group(0).replace("M00L515.cam0.focus 0.000", '')
                #aux = m.group(0).replace("691_good-", '')

                # skipping very first pic
                if aux == '160':
                    print("skipping first image...")
                    continue

                self.gbs = np.append(self.gbs, int(aux))

                #self.gbs = np.append(self.gbs, count)
            except AttributeError:
                aux = os.path.basename(infile)
                m = re.search('(\\d+)', aux)
                self.gbs = np.append(self.gbs, int(m.group(0)))

            #m = re.search('[+-](\\d+)', infile)
            #self.gbs = np.append(self.gbs, int(m.group(0)))

            #m = re.search(r"\d+", infile)
            #self.gbs = np.append(self.gbs, int(m.group(1)))

            #self.gbs = np.append(self.gbs, initialFocus)
            #initialFocus += 5

            print("index", index)
            nda = sitk.GetArrayFromImage(images[:, :, index])
            nda = nda.astype(np.int32)

            if (self.pearson):
                pears = self.calcPearson(nda)
                self.pearson = np.append(self.pearson, pears)
                print("pearson R", pears)

            # print("max: ", nda.max())
            # print("min: ", nda.min())
            # print("mean", nda.mean())
            # print(unravel_index(nda.argmax(), nda.shape))
            # print(nda.shape)

            # Normalized Variance (Groen et al., 1985; Yeo et al., 1993)
            mymean = nda.mean()
            F11 = np.sum(np.square(nda - mymean)) / (nda.size * mymean)
            self.f11 = np.append(self.f11, F11)
            print("F11", F11)

            F12 = np.sum(np.apply_along_axis(self.calcF12, 1, nda))
            self.f12 = np.append(self.f12, F12)
            print("F12", F12)

        print("max f11 at timepoint", self.getF11Max())
        print("max f12 at timepoint ", self.getF12Max())

        if (self.pearson):
            print("max pearson at timepoint", self.getPearsonMax())


    def plotF11(self):
        plt.clf()
        plt.title("focal length using F11")
        plt.xlabel("focal length [um]")
        plt.ylabel("F11 value")
        plt.plot(self.gbs, self.f11, '.r')

        if(self.saveToFile):
            aux = os.path.splitext(self.filePath)[0] + "_F11.png"
            plt.savefig(os.path.join(self.parentDir, aux))
            print("saved F11 diagram in ", aux)
        else:
            plt.show()

        #if os.path.splitext(os.path.basename(self.filePath))[0] == '391':
        # self.fitGaussian()

    def plotF12(self):
        plt.clf()
        plt.title("focal length using F12")
        plt.xlabel("focal length [um]")
        plt.ylabel("F12 value")
        plt.plot(self.gbs, self.f12, '.r')
        if self.saveToFile:
            aux = os.path.splitext(self.filePath)[0] + "_F12.png"
            plt.savefig(os.path.join(self.parentDir, aux))
            print("saved F12 diagram in ", aux)
        else:
            plt.show()

    def plotPearson(self):
        plt.clf()
        plt.title("focal length using pearson")
        plt.xlabel("focal length [um]")
        plt.ylabel("pearson R")
        plt.plot(self.gbs, self.pearson, '.r')
        if self.saveToFile:
            aux = os.path.splitext(self.filePath)[0] + "_pearson.png"
            plt.savefig(os.path.join(self.parentDir, aux))
            print("saved pearson diagram in ", aux)
        else:
            plt.show()

    def gauss(self, x, *p):
        A, mu, sigma = p
        return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))

    # def poisson(self):
    #     fitfunc = lambda p, x: p[0] * pow(p[1], x) * pow(e, -p[1]) / factorial(x)

    def fitGaussian(self, fileName):
        # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
        p0 = [10., 180., 10.]

        try:
            coeff, var_matrix = curve_fit(self.gauss, self.gbs, self.f11, p0=p0, maxfev=10000)
        except RuntimeError:
            print('exception thrown')
            print(RuntimeError)
            return

        # Get the fitted curve
        hist_fit = self.gauss(self.gbs, *coeff)

        # Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
        print('Fitted A = ', coeff[0])
        print('Fitted mean = ', coeff[1])
        print('Fitted standard deviation = ', coeff[2])

        self.gaussianMean = np.append(self.gaussianMean, coeff[1])
        self.gaussianStd = np.append(self.gaussianStd, coeff[2])
        self.stackCountArray = np.append(self.stackCountArray, self.stackCount)
        self.stackCount += 1

        self.stackFileNames = np.append(self.stackFileNames, fileName)

        plt.clf()
        plt.title("raw data + gaussian fit, mean= {0:.2f}, std= {1:.2f}".format(coeff[1], coeff[2]))

        plt.plot(self.gbs, self.f11, label='Test data')
        plt.plot(self.gbs, hist_fit, label='Fitted data')

        plt.xlabel("focal length [um]")
        plt.ylabel("F11 value")

        plt.tight_layout()

        if self.saveToFile:
            aux = os.path.splitext(self.filePath)[0] + "_gaussFitNew.png"
            plt.savefig(os.path.join(self.parentDir, aux))
            print("saved gaussian fit in ", aux)
        else:
            plt.show()

    def plotGaussianMean(self):
        plt.clf()
        plt.title("Gaussian Means using F11")
        plt.xlabel("testStack")
        plt.ylabel("F11 value")

        plt.xticks(self.stackCountArray, self.stackFileNames)
        plt.errorbar(self.stackCountArray, self.gaussianMean, yerr=self.gaussianStd, fmt='o')

        plt.xticks(range(len(self.stackFileNames)), self.stackFileNames, rotation=90)

        plt.tight_layout()

        if self.saveToFile:
            #aux = os.path.splitext(self.filePath)[0] + "_Gaussian_Means.png"
            aux = "/Users/alexander.bindeus/Desktop/ER_testsets/fs_160_200_in_30/all_Gaussian_Means.png"
            plt.savefig(os.path.join(self.parentDir, aux))
            print("saved gaussian Mean diagram in ", aux)
        else:
            plt.show()

    def analyseParentFolder(self):

        subfolders = [f.path for f in os.scandir(self.parentDir) if f.is_dir()]

        count = 0
        xpoint = 0

        for sampleDir in subfolders:
            if sampleDir[0].isdigit:

                print("running folder", sampleDir)
                self.setFilePath(sampleDir)

                self.analyseFolder()
                self.plotF11()
                self.plotF12()
                self.fitGaussian(sampleDir.rsplit('/', 1)[-1])

                # if count % 2 == 0:
                #     print("GFP folder")
                #     self.GFP_f11Max = np.append(self.GFP_f11Max, self.getF11Max())
                #     self.GFP_f12Max = np.append(self.GFP_f12Max, self.getF12Max())
                # else:
                #     print("RFP folder")
                #     self.RFP_f11Max = np.append(self.RFP_f11Max, self.getF11Max())
                #     self.RFP_f12Max = np.append(self.RFP_f12Max, self.getF12Max())
                #     xpoint += 1
                #     self.xVal = np.append(self.xVal, xpoint)

                self.clearArrays()

                count += 1

    def plotF11F12(self):
        plt.clf()

        plt.plot(self.xVal, self.GFP_f11Max, '--g', label='GPF_F11')
        plt.plot(self.xVal, self.RFP_f11Max, '--r', label='RPF_F11')
        plt.plot(self.xVal, self.GFP_f12Max, '-g', label='GPF_F12')
        plt.plot(self.xVal, self.RFP_f12Max, '-r', label='RPF_F12')

        plt.title("GPF/ RPF focal length using F11 and F12 algo")
        plt.xlabel("measured positions")
        plt.ylabel("best focal length [um]")

        plt.legend(loc='upper left')
        plt.show()


    # AutoCorrelation (Vollath, 1987, 1988)
    def calcF12(self, row):
        f = int(0)
        for i in range(0, len(row) - 2):
            f += row[i] * (row[i + 1] - row[i + 2])

        return f

    def sitk_show(self, img, title=None, margin=0.05, dpi=40):
        nda = sitk.GetArrayFromImage(img)

        print("max: ", nda.max())
        print("min: ", nda.min())
        print("mean", nda.mean())

        print(unravel_index(nda.argmax(), nda.shape))

        spacing = img.GetSpacing()
        figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
        extent = (0, nda.shape[1] * spacing[1], nda.shape[0] * spacing[0], 0)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

        plt.set_cmap("gray")
        ax.imshow(nda, extent=extent, interpolation=None)

        if title:
            plt.title(title)

        plt.show()

    def analyseStack(self, stackFile):

        stack = sitk.ReadImage(stackFile)
        stackSize = stack.GetSize()[2]

        #self.sitk_show(sitk.Tile(stack[:, :, 1], stack[:, :, 2], (2, 1, 0)))

        for x in range(0, stackSize-1):

            print("processing stack:", x)
            self.gbs = np.append(self.gbs, x)

            nda = sitk.GetArrayFromImage(stack[:, :, x])
            nda = nda.astype(np.int32)

            # Normalized Variance (Groen et al., 1985; Yeo et al., 1993)
            mymean = nda.mean()
            F11 = np.sum(np.square(nda - mymean)) / (nda.size * mymean)
            self.f11 = np.append(self.f11, F11)
            print("F11", F11)

            F12 = np.sum(np.apply_along_axis(self.calcF12, 1, nda))
            self.f12 = np.append(self.f12, F12)
            print("F12", F12)

        print("max f11 at timepoint", self.getF11Max())
        print("max f12 at timepoint ", self.getF12Max())

        self.plotF11()
        self.plotF12()


if __name__ == '__main__':
    print("detecting focus")
    AutoFocus = AutoFocus()

    #AutoFocus.analyseStack('/Users/alexander.bindeus/Desktop/ER/691_good-2.tif')

    #parentDir = '/Users/alexander.bindeus/Desktop/170502_dual_f'
    parentDir = '/Users/alexander.bindeus/Desktop/ER_testsets/fs_160_200_in_30'

    AutoFocus.setSaveToFile(True)
    #AutoFocus.setFilePath('/Users/alexander.bindeus/Desktop/ER')

    AutoFocus.setParentDir(parentDir)
    AutoFocus.analyseParentFolder()

    AutoFocus.plotGaussianMean()

    # AutoFocus.plotF11F12()

    # AutoFocus.setFilePath('/Users/alexander.bindeus/Desktop/15C')
    # AutoFocus.setRefFrame('/Users/alexander.bindeus/Desktop/15C/106.tif')

    #AutoFocus.setFilePath('/Users/alexander.bindeus/Desktop/T2')
    #AutoFocus.setRefFrame('/Users/alexander.bindeus/Desktop/T1/pic0020.tif')

    #AutoFocus.setFilePath('/Users/alexander.bindeus/Desktop/root1/kr/2017518-113145/tp1/Region_0/Cam0')

    #AutoFocus.analyseFolder()
    #AutoFocus.plotF11()
    #AutoFocus.plotF12()
    #AutoFocus.plotPearson()





