#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Not titled yet
# GNU Radio version: 3.8.2.0

from distutils.version import StrictVersion

if __name__ == '__main__':
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
        try:
            x11 = ctypes.cdll.LoadLibrary('libX11.so')
            x11.XInitThreads()
        except:
            print("Warning: failed to XInitThreads()")

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio.filter import firdes
import sip
from gnuradio import analog
from gnuradio import blocks
import numpy
from gnuradio import digital
from gnuradio import gr
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation

from gnuradio import qtgui

class test2(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Not titled yet")
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Not titled yet")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except:
            pass
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "test2")

        try:
            if StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
                self.restoreGeometry(self.settings.value("geometry").toByteArray())
            else:
                self.restoreGeometry(self.settings.value("geometry"))
        except:
            pass

        ##################################################
        # Variables
        ##################################################
        self.sr_ch = sr_ch = 100e3
        self.sr = sr = 1e6
        self.q = q = digital.constellation_rect([-1-1j, -1+1j, 1+1j, 1-1j], [0, 1, 3, 2],
        4, 2, 2, 1, 1).base()

        ##################################################
        # Blocks
        ##################################################
        self.qtgui_sink_x_0 = qtgui.sink_c(
            1024, #fftsize
            firdes.WIN_BLACKMAN_hARRIS, #wintype
            0, #fc
            sr, #bw
            "", #name
            True, #plotfreq
            True, #plotwaterfall
            True, #plottime
            True #plotconst
        )
        self.qtgui_sink_x_0.set_update_time(1.0/10)
        self._qtgui_sink_x_0_win = sip.wrapinstance(self.qtgui_sink_x_0.pyqwidget(), Qt.QWidget)

        self.qtgui_sink_x_0.enable_rf_freq(False)

        self.top_grid_layout.addWidget(self._qtgui_sink_x_0_win)
        self.digital_constellation_modulator_0 = digital.generic_mod(
            constellation=q,
            differential=True,
            samples_per_symbol=4,
            pre_diff_code=True,
            excess_bw=0.3,
            verbose=False,
            log=False)
        self.blocks_wavfile_sink_0_1 = blocks.wavfile_sink('/home/fabian/spectrum-sensing-salsat/code/samples/gnu_psk_signal.wav', 2, int(sr), 8)
        self.blocks_wavfile_sink_0_0 = blocks.wavfile_sink('/home/fabian/spectrum-sensing-salsat/code/samples/gnu_psk_both.wav', 2, int(sr), 8)
        self.blocks_wavfile_sink_0 = blocks.wavfile_sink('/home/fabian/spectrum-sensing-salsat/code/samples/gnu_psk_noise.wav', 2, int(sr), 8)
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex*1, sr,True)
        self.blocks_complex_to_float_0_1 = blocks.complex_to_float(1)
        self.blocks_complex_to_float_0_0 = blocks.complex_to_float(1)
        self.blocks_complex_to_float_0 = blocks.complex_to_float(1)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.analog_random_source_x_0 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 255, 1000))), True)
        self.analog_noise_source_x_0 = analog.noise_source_c(analog.GR_GAUSSIAN, 1, 0)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_complex_to_float_0, 0))
        self.connect((self.analog_random_source_x_0, 0), (self.digital_constellation_modulator_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.blocks_complex_to_float_0_0, 0))
        self.connect((self.blocks_add_xx_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.blocks_complex_to_float_0, 0), (self.blocks_wavfile_sink_0, 0))
        self.connect((self.blocks_complex_to_float_0, 1), (self.blocks_wavfile_sink_0, 1))
        self.connect((self.blocks_complex_to_float_0_0, 0), (self.blocks_wavfile_sink_0_0, 0))
        self.connect((self.blocks_complex_to_float_0_0, 1), (self.blocks_wavfile_sink_0_0, 1))
        self.connect((self.blocks_complex_to_float_0_1, 1), (self.blocks_wavfile_sink_0_1, 1))
        self.connect((self.blocks_complex_to_float_0_1, 0), (self.blocks_wavfile_sink_0_1, 0))
        self.connect((self.blocks_throttle_0, 0), (self.qtgui_sink_x_0, 0))
        self.connect((self.digital_constellation_modulator_0, 0), (self.blocks_add_xx_0, 0))
        self.connect((self.digital_constellation_modulator_0, 0), (self.blocks_complex_to_float_0_1, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "test2")
        self.settings.setValue("geometry", self.saveGeometry())
        event.accept()

    def get_sr_ch(self):
        return self.sr_ch

    def set_sr_ch(self, sr_ch):
        self.sr_ch = sr_ch

    def get_sr(self):
        return self.sr

    def set_sr(self, sr):
        self.sr = sr
        self.blocks_throttle_0.set_sample_rate(self.sr)
        self.qtgui_sink_x_0.set_frequency_range(0, self.sr)

    def get_q(self):
        return self.q

    def set_q(self, q):
        self.q = q





def main(top_block_cls=test2, options=None):

    if StrictVersion("4.5.0") <= StrictVersion(Qt.qVersion()) < StrictVersion("5.0.0"):
        style = gr.prefs().get_string('qtgui', 'style', 'raster')
        Qt.QApplication.setGraphicsSystem(style)
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    def quitting():
        tb.stop()
        tb.wait()

    qapp.aboutToQuit.connect(quitting)
    qapp.exec_()

if __name__ == '__main__':
    main()
