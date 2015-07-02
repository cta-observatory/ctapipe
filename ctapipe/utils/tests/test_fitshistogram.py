"""Unit test for fitshistogram.py"""

from ctapipe.utils import fitshistogram
import unittest
import numpy as np


class TestFITSHistogram(unittest.TestCase):

    """
    Test filling histograms
    """

    def setUp(self):
        pass

    def test_to_string(self):
        H = fitshistogram.Histogram(
            nbins=[5, 10], ranges=[[-2.5, 2.5], [-1, 1]],
            name="testhisto")
        print(H)
    
    def testSimpleFillAndRead(self):
        """
        """

        H = fitshistogram.Histogram(
            nbins=[5, 10], ranges=[[-2.5, 2.5], [-1, 1]])

        pa = (0.1, 0.1)
        pb = (-0.55, 0.55)

        a = np.ones((100, 2)) * pa  # point at 0.1,0.1
        b = np.ones((10, 2)) * pb  # 10 points at -0.5,0.5

        H.fill(a)
        H.fill(b)

        va = H.getValue(pa)[0]
        vb = H.getValue(pb)[0]

        self.assertEqual(va, 100)
        self.assertEqual(vb, 10)

    def testRangeFillAndRead(self, ):
        """
        Check that the correct bin is read and written for multiple
        binnings and fill positions
        """

        N = 100

        for nxbins in np.arange(1, 50, 1):
            for xx in np.arange(-2.0, 2.0, 0.1):
                pp = (xx + 0.01829384, 0.1)
                coords = np.ones((N, 2)) * pp
                H = fitshistogram.Histogram(nbins=[nxbins, 10],
                                            ranges=[[-2.5, 2.5], [-1, 1]])
                H.fill(coords)
                val = H.getValue(pp)[0]
                self.assertEqual(val, N)
                del H

    # def testOutliers(self):
    #     """
    #     Check that out-of-range values work as expected
    #     """
    #     H = fitshistogram.Histogram( nbins=[5,10], range=[[-2.5,2.5],[-1,1]] )
    #     H.fill( np.array( [[1,1],]) )
    #     val1= H.getValue( (100,100), outlierValue = -10000)[0]
    #     val2= H.getValue( (-100,0), outlierValue = None)[0]
    #     self.assertEqual(val1,-10000)
    #     self.assertEqual(val2,0)

    def testFITS(self):
        """
        Write to fits,read back, and check
        """

        pass


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFITSHistogram)
    unittest.TextTestRunner(verbosity=2).run(suite)
