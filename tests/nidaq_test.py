#!/cygdrive/c/Python25/python.exe -i
## Workaround for symlinks not working in windows
import sys
import time


# sys.path.append("..\\cheader")
from nidaq import cheader
from nidaq import NIDAQ
import nidaq
import nidaqmx
from nidaqmx.constants import AcquisitionType, Edge, VoltageUnits
import numpy as np
import pyqtgraph as pg
import win32com.client


print("Assert num devs > 0:")
assert(len(NIDAQ.listDevices()) > 0)
print("  OK")
print("devices: %s" % NIDAQ.listDevices())

# print("getDevice:")
dev0 = NIDAQ.getDevice(b'Dev1')
# print("  ", dev0)
# # print(dir(dev0))
# print("\nAnalog Channels:")
# # print "  AI: ", dev0.listAIChannels()
# print("  AO: ", dev0.listAOChannels())

# print("\nDigital ports:")
# print("  DI: ", dev0.listDIPorts())
# print("  DO: ", dev0.listDOPorts())

# print("\nDigital lines:")
# print("  DI: ", dev0.listDILines())
# print("  DO: ", dev0.listDOLines())

def finiteReadTest():
  task = dev0.createTask()
  task.CreateAIVoltageChan("/Dev1/ai0", "chan0", nidaq.Val_RSE, -1., 1., nidaq.Val_Volts, None)
  print('called this function')
  task.CreateAIVoltageChan("/Dev1/ai1", "chan1", nidaq.Val_Cfg_Default, -10., 10., nidaq.Val_Volts, None)
  
  task.CfgSampClkTiming(None, 10000.0, nidaq.Val_Rising, nidaq.Val_FiniteSamps, 1000)
  task.start()
  data = task.read()
  task.stop()
  
  return data



def contReadTest():
  task = dev0.createTask()
  task.CreateAIVoltageChan("/Dev1/ai0", "chan0", nidaq.Val_RSE, -10., 10., nidaq.Val_Volts, None)
  task.CfgSampClkTiming(None, 10000.0, nidaq.Val_Rising, nidaq.Val_ContSamps, 4000)
  task.start()
  t = time.time()
  for i in range(0, 10):
    data, size = task.read(1000)
    print("Cont read %d - %d samples, %fsec" % (i, size, time.time() - t))
    t = time.time()
  task.stop()

def initialize_attnenuator():
  PA5 = win32com.client.Dispatch("PA5.x")
  PA5.ConnectPA5("USB", 1)
  PA5.SetAtten(120.0)
  PA5.ConnectPA5("USB", 2)
  PA5.SetAtten(120.0)
  return PA5

## Output task

def outputTest(duration:float=0.1, freq:float=1000.0, isi:float=0.2, USB_number:int=1, plotwave:bool=True):
  with nidaqmx.Task() as task:
    # print(dir(task))
    #   task = dev0.createTask()
    # task.CreateAOVoltageChan("/Dev1/ao0", "ao0", -10., 10., nidaq.Val_Volts, None)
    # print('HERE!!!')
    PA5 = initialize_attnenuator()
    task.ao_channels.add_ao_voltage_chan("/dev1/ao0", min_val=-10., max_val=10.)
    task.ao_channels.add_ao_voltage_chan("/dev1/ao1", min_val=-10., max_val=10.)
    clock = 100000.0
    # duration = 0.5 # seconds
    ndata = int(duration*clock)
    # freq = 1000.0
    task.timing.cfg_samp_clk_timing(clock, source="",
                                    active_edge=Edge.RISING, 
                                    sample_mode=AcquisitionType.FINITE, samps_per_chan=ndata)
    t = np.arange(0, duration, 1./clock)
    if t.size > ndata:
      t = t[:ndata]
    data = np.zeros((2, ndata))
    data[0, :] = np.ones((ndata), dtype=np.float64)*np.sin(2.0*np.pi*freq*t)
    data[1, :] = np.zeros(ndata)
    data[1, 0] = 5.0
  
    if plotwave:
       pg.plot(data)
       if (sys.flags.interactive != 1) or not hasattr(pg.QtCore, "PYQT_VERSION"):
            pg.QtGui.QGuiApplication.instance().exec()
    PA5.ConnectPA5("USB", USB_number)
    PA5.SetAtten(10.0)
    # print(len(data), data.max(), data.min())
    # data[200:400] = 5.0
    # data[600:800] = 5.0
    for i in range(10):
      task.write(data)
      task.start()
      time.sleep(duration)
      task.stop()
      time.sleep(isi)
  
    PA5.SetAtten(120.0)


## Synchronized tasks

def syncADTest():
  task1 = dev0.createTask()
  task1.CreateAIVoltageChan("/Dev1/ai0", "ai0", nidaq.Val_Cfg_Default, -10., 10., nidaq.Val_Volts, None)
  task1.CfgSampClkTiming(None, 10000.0, nidaq.Val_Rising, nidaq.Val_FiniteSamps, 1000)
  task2 = dev0.createTask()
  task2.CreateDIChan("/Dev1/port0", "di0", nidaq.Val_ChanForAllLines)
  task2.CfgSampClkTiming("/Dev1/ai/SampleClock", 10000.0, nidaq.Val_Rising, nidaq.Val_FiniteSamps, 1000)
  
  data1 = np.zeros((1000,), dtype=np.float64)
  data1[200:400] = 5.0
  data1[600:800] = 5.0
  task2.start()
  print("Wrote samples:", task2.write(data1))
  task1.start()
  data2 = task1.read()
  task2.stop()
  task1.stop()
  
  print(data2)


def syncIOTest():
  task1 = dev0.createTask()
  task1.CreateAIVoltageChan("/Dev1/ai0", "", nidaq.Val_RSE, -10., 10., nidaq.Val_Volts, None)
  task1.CfgSampClkTiming(None, 40000.0, nidaq.Val_Rising, nidaq.Val_FiniteSamps, 1000)

  task2 = dev0.createTask()
  task2.CreateAOVoltageChan("/Dev1/ao0", "", -10., 10., nidaq.Val_Volts, None)
  #task2.CfgSampClkTiming(None, 10000.0, nidaq.Val_Rising, nidaq.Val_FiniteSamps, 1000)
  task2.CfgSampClkTiming("ai/SampleClock", 40000.0, nidaq.Val_Rising, nidaq.Val_FiniteSamps, 1000)
  task2.CfgDigEdgeStartTrig("ai/StartTrigger", nidaq.Val_Rising)
  #task1.SetRefClkSrc("PXI_Clk10")
  #task2.SetRefClkSrc("PXI_Clk10")
  print(task1.GetSampClkTimebaseSrc())
  print(task2.GetSampClkTimebaseSrc())
  task2.SetSampClkTimebaseSrc("SampleClockTimebase")
  #task2.SetSyncPulseSrc("/Dev1/SyncPulse")


  data1 = np.zeros((1000,), dtype=np.float64)
  data1[200:400] = 5.0
  data1[600:800] = 5.0
  print("Wrote samples:", task2.write(data1))
  task2.start()
  task1.start()
  data2 = task1.read()
  #time.sleep(1.0)
  task1.stop()
  task2.stop()
  
  return data2













## here is how it should work
#def analogTest(dev0):
  #task = dev0.addTask()
  #max_num_samples = 1000
  #assert(len(dev0.listAnalogInputChannels()) > 0)
  #print "channels: %s" % dev0.listAnalogInputChannels()
  #chan0 = nidaq.Channel("AIVoltage", "ai0")
  #chan0.minVal = -10.0
  #chan0.maxVal = 10.0
  #chan0.terminalConfig = nidaq.DAQmx_Val_Cfg_Default
  #chan0.units = nidaq.DAQmx_Val_Volts
  #task.addChannel(chan0)
  #timing0 = nidaq.Timing("SampClk")
  #timing0.source = ""
  #timing0.rate = 10000.0
  #timing0.activeEdge = nidaq.DAQmx_Val_Rising
  #timing0.sampleMode = nidaq.DAQmx_Val_FiniteSamps
  #timing0.samplesPerChanToAcquire = max_num_samples
  #task.addTiming(timing0)
  #data = task.read()
  #assert(len(data) == max_num_samples)
  #moreData = task.read(blocking=False)
  #assert(len(moreData) <= max_num_samples)
  #evenMore = task.read(max_num_samples * 2)
  #assert(len(evenMore) == (max_num_samples * 2))

#def digitalTest():
  #pass

##analogTest(dev0)
##digitalTest()

if __name__ == "__main__":
   outputTest(plotwave=False)