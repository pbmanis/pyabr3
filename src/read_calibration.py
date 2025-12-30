import numpy as np
import datetime
from pathlib import Path
import scipy.io
from pyqtgraph import configfile
import matplotlib.pyplot as mpl
import pyqtgraph as pg
#

SPLCAL_Tone_Voltage = 10.0  # V from matlab calibration (hardware_initialization.m)
SPLCAL_Click_Voltage = 10.0 # V for clicks
use_matplotlib = False

"""File structure is: 
CAL.RefSPL = CALIBRATION.SPLCAL.maxtones;
CAL.Freqs = spkr_freq; [1]
CAL.maxdB = maxdB; [2]
CAL.dBSPL = dbspl_cs; [3]
CAL.dBSPL_bp = dbspl_bp;
CAL.dBSPL_nf = dbspl_nf; [4]
CAL.Vmeas = Vrms_cs; [5]
CAL.Vmeas_bp = Vrms_bp; [6]
CAL.Gain = 20; % dB setting microphone amplifier gain
CAL.CalAttn = CALIBRATION.SPKR.CalAttn; % attenuator setting at which calibration was done
CAL.Speaker = Speaker;
CAL.Microphone = Mic;
CAL.Date = date;
CAL.DateTime = datetime();

cs refers to cosinor measurements.
bp refers to bandpass (narrow band filter)
nf refers to noise floor.
maxdB refers to max sound pressure level at 0 dB attenuation.

"""

def get_calibration_data(fn):
    if not Path(fn).is_file():
        raise ValueError(f'Calibration file: {str(fn):s} not found')
        exit()
    dm = scipy.io.loadmat(fn, appendmat=False,  squeeze_me=True)
    print(fn)
    d = dm["CAL"].item()
    caldata = {}
    caldata['refspl'] = d[0]
    # print("Ref SPL: ", caldata['maxspl'])
    caldata['freqs'] = d[1]
    caldata['maxdb'] = d[2]
    caldata['db_cs'] = d[3]
    caldata['db_bp'] = d[4]
    caldata['db_nf'] = d[5]
    caldata['vm_cs'] = d[6]
    caldata['vm_bp'] = d[7]
    caldata['gain'] = d[8]
    caldata['calattn'] = d[9]
    caldata['spkr'] = d[10]
    caldata['mic'] = d[11]
    caldata['date'] = d[12]
    caldata['filename'] = fn

    return caldata

def get_microphone_calibration(fn):
    dm = scipy.io.loadmat(fn, appendmat=False,  squeeze_me=True)
    # print(fn)
    """
     MIC is a structure with the following fields:

          Gain: 20  # amplifier gain
        RefSig: 94  # level used for the calibration in this file
          Vrms: 0.0395  # raw rms microphone voltage, in V
        Vref_c: 0.0286  # cosinor measure of microphone voltage, in Vrms
       Vref_bp: 0.0389  # bandpassed measure of microphone voltae, in Vrms
    Microphone: '7016#10252'  # identiy of the the microphone
          Date: '11-Jan-2022'  # date of calibration
      dBPerVPa: -48.2029   # sensitivity
       mVPerPa: 3.8892   # calibration factor
    """
    d = dm["MIC"].item()
    micdata = {}
    micdata['cal_gain'] = d[0]
    micdata['cal_ref'] = d[1]
    micdata['Vrms'] = d[2]
    micdata['Vref_c'] = d[3]
    micdata['Vref_bp'] = d[4]
    micdata["microphone"] = d[5]
    micdata["date"] = d[6]
    micdata["dBPerVPa"] = d[7]
    micdata["mVPerPa"] = d[8]
    # print(micdata)
    return micdata

def plot_calibration(caldata, plot_target = None):
    txt = f"Gain: {caldata['gain']:.1f}  Cal attn: {caldata['calattn']:.1f} dB, "
    txt += f"Speaker: {caldata['spkr']:s}, Mic: {caldata['mic']:s}, date: {caldata['date']:s}\nFile: {str(caldata['filename']):s}"
    if use_matplotlib:
        f, ax = mpl.subplots(1,1)
        freqs = caldata['freqs']
        ax.semilogx(freqs, caldata['maxdb'], 'ro-')
        ax.semilogx(freqs, caldata['db_cs'], 'k--')
        ax.semilogx(freqs, caldata['db_bp'], 'g--')
        ax.semilogx(freqs, caldata['db_nf'], 'b-')
        ax.set_xlabel ("F, Hz")
        ax.set_ylabel("dB SPL")
        fn = caldata['filename']
        txt = f"Gain: {caldata['gain']:.1f}  Cal attn: {caldata['calattn']:.1f} dB, "
        txt += f"Speaker: {caldata['spkr']:s}, Mic: {caldata['mic']:s}, date: {caldata['date']:s}\nFile: {str(fn):s}"
        mpl.suptitle(txt, fontsize=7)
        ax.grid(True, which="both")
        f.tight_layout()
        mpl.show()
    else:
        if plot_target is None:
            app = pg.mkQApp("Calibration Data Plot")
            win = pg.GraphicsLayoutWidget(show=True, title="ABR Data Plot")
            win.resize(500, 500)
            win.setWindowTitle(f"File: {caldata['filename']}")
            pl = win.addPlot(title=f"Calibration")  # type:ignore
        else:
            pl = plot_target
        freqs = caldata['freqs']
        pl.addLegend()
        pl.setLogMode(x=True, y=False)
        pl.plot(freqs, caldata['maxdb'], pen='r', name="Max SPL (0 dB Attn)")
        pl.plot(freqs, caldata['db_cs'], pen='w', name=f"Measured dB SPL, attn={caldata['calattn']:.1f} cosinor")
        pl.plot(freqs, caldata['db_bp'], pen='g', name=f"Measured dB SPL, attn={caldata['calattn']:.1f}, bandpass")
        pl.plot(freqs, caldata['db_nf'], pen='b', name="Noise Floor")
        # pl.setLogMode(x=True, y=False)
        pl.setLabel("bottom", "Frequency", units="Hz")
        pl.setLabel("left", "dB SPL")
        pl.showGrid(x=True, y=True)
        text_label = pg.LabelItem(txt, size="8pt", color=(255, 255, 255))
        text_label.setParentItem(pl)
        text_label.anchor(itemPos=(0.5, 0.05), parentPos=(0.5, 0.05))

        if plot_target is None:
            pg.exec()
def main():
    import sys
    configtype = "lab"
    cmd = sys.argv[1:]
    print("cmd: ", cmd)

    if cmd == "test":
        configfilename = "config/abrs_test.cfg"
    else:
        configtype = "lab"
        configfilename = "config/abrs.cfg"
    assert configtype in ["test", "lab"]
    # get the latest calibration file:
    cfg = configfile.readConfigFile(configfilename)
    print(cfg)
    fn = Path(cfg['calfile'])
    # fn = Path("E:/Users/Experimenters/Desktop/ABR_Code/frequency_MF1.cal")
    print(fn.is_file())

    d = get_calibration_data(fn)
    plot_calibration(d)

            
if __name__ == "__main__":

    calfiles = list(Path('calfiles/').glob('microphone*.cal'))
    f, ax = mpl.subplots(2,1, figsize=(6,8))
    day = []
    cal = []
    db_pA = []
    format_string = "%d-%b-%Y" #  %H:%M:%S"
    for calfile in calfiles:
        mic_cal = get_microphone_calibration(calfile)
        if 0.5 >= mic_cal["mVPerPa"] or mic_cal["mVPerPa"] >= 10:
            continue
        cal.append(mic_cal["mVPerPa"])
        db_pA.append(mic_cal["dBPerVPa"])
        datetime_object = datetime.datetime.strptime(mic_cal["date"], format_string)
        day.append(datetime_object)
    i_dsort = np.argsort(day)
    day = np.array(day)[i_dsort]
    cal = np.array(cal)[i_dsort]
    db_pA = np.array(db_pA)[i_dsort]
    ax[0].plot(day, cal, 'o-')
    ax[0].set_xlabel("Date", fontsize=10   )
    ax[0].tick_params("x", rotation=45)
    ax[0].set_ylabel("mV/Pa", fontsize=10)
    ax[0].set_title("Microphone Calibration Over Time", fontsize=11)
    ax[0].grid(True, linestyle='--', linewidth=0.5)
    ax[0].set_ylim(3.0,4.5)
    ax[1].plot(day, db_pA, 'o-')
    ax[1].set_xlabel("Date", fontsize=10)
    ax[1].tick_params("x", rotation=45)
    ax[1].set_ylabel("dB/V/Pa", fontsize=10)
    ax[1].grid(True, linestyle='--', linewidth=0.5)
    ax[1].set_ylim([-52,-42])
    f.tight_layout()
    mpl.show()
    # get_microphone_calibration("calfiles/microphone_7016#10252.cal")