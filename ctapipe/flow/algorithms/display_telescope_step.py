from ctapipe.core import Component
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from traitlets import Bool
from traitlets import Unicode


class DisplayTelescopeStep(Component):
    """DisplayTelescopeStep` class represents a Stage for pipeline.
        it displays calibrated camera events with matplotlib and/or write
        them to a PDF file
    """
    display = Bool(default_value=False,
                   help='Display the camera events').tag(config=True)
    pdf_path = Unicode(default_value=None, allow_none=True,
                       help='Path to store a pdf output of the plots').tag(config=True)

    def init(self):
        self.log.debug("--- DisplayTelescopeStep init ---")
        self.pdfPages = None
        if self.pdf_path:
            self.pdfPages = PdfPages(self.pdf_path)
        return True

    def run(self, figure):
        self.log.debug('%figure' % "DisplayTelescopeStep receive {}")
        if self.display:
            plt.pause(.1)
        if self.pdfPages is not None:
            self.pdfPages.savefig(figure)

    def finish(self):
        self.log.debug("--- DisplayTelescopeStep finish ---")
        if self.pdfPages:
            self.pdfPages.close()
